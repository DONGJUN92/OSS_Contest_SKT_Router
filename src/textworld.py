"""World F "textworld" — 실텍스트 예행 세계 (Phase 8, 심사 권고 1+2 통합).

합성 세계(A~E)와 달리 실제 텍스트가 흐른다:
  - 프롬프트: 난이도 등급이 내장된 4개 태스크(산술/정렬/문자세기/JSON 포맷)의 실문장
  - 후보 모델: 능력 의존 오류 과정을 가진 프로그램적 답변기 — 실제 출력 텍스트 생성
    (오답 왜곡, 포맷 파괴, 반복, 얼버무림, 장황함이 모델 능력에 따라 분포)
  - 품질 라벨: 출력에서 답을 추출해 정답과 대조 (챌린지의 '출력별 품질 평가'에 대응)
  - 다수결 샘플: 모델당 S=5개 실출력 → 투표 상자가 실제 다수결로 동작

ScoringVerifier(출력 텍스트→품질 신호)와 TextEncoder(프롬프트→특성)가 이 세계에서
실물로 검증된다. 정답(truth)은 라벨링에만 쓰이고 verifier/encoder는 접근 불가.
"""
import json as jsonlib
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

TASKS = ["arith", "sort", "strcount", "jsonfmt"]
FILLERS = [
    "Let me work through this carefully.", "First, I will restate the problem.",
    "Considering the given values step by step.", "Here is my reasoning process.",
    "After double-checking the intermediate results.", "To be thorough, I re-derived it.",
]
HEDGES = ["I am not fully sure, but", "This might be wrong, but", "Possibly,"]
REFUSAL = "I cannot reliably determine the result for this request."


def graded_score(task: str, ans: str | None, truth: str, scale: str = "strict") -> float:
    """부분점수 채점 (B4 — Q4=연속 분기의 검증 데이터).

    이진 라벨 `maj == truth` 를 태스크별 **연속 근접도**로 대체한다. 점수는 실제 출력
    텍스트에서 유도되며 Beta 분포에서 표집하지 않는다 — 신념 모형(Beta 반응모형)이
    자기가 만든 데이터를 되맞히는 자기확증을 차단하기 위함이다 (World A/B 설계 원칙).
    정답이면 정확히 1.0, 추출 실패·거부는 0.0 — 이진 라벨의 양 끝을 보존한다.

    척도 2종 (**결론은 양쪽에서 유지되어야 한다**):
      strict  : 위치 일치율·지수 감쇠 — 오답에 인색한 채점자
      lenient : 국소 순서·상대 오차 — 오답에 관대한 채점자
    두 척도를 두는 이유: 부분점수 규칙은 주최측이 정하며 우리가 모른다. 한 척도에서만
    성립하는 결론은 척도를 우리에게 유리하게 고른 것과 구분되지 않는다. 실제로 초안의
    관대 척도는 평균 0.92·0점 0.9%로 변별력이 붕괴해(어떤 정책이든 ~0.95) A/B가
    무의미해졌다 — 그 사실 자체를 여기 남긴다.
    """
    if ans is None:
        return 0.0
    if _norm_answer(ans) == _norm_answer(truth):
        return 1.0
    strict = scale == "strict"
    try:
        if task == "arith":
            a, t = float(ans), float(truth)
            if strict:                       # 자릿수만 맞으면 소액 부분점수 (루브릭식)
                same_mag = (a != 0 and t != 0
                            and np.sign(a) == np.sign(t)
                            and abs(np.log10(abs(a)) - np.log10(abs(t))) < 1.0)
                return 0.25 if same_mag else 0.0
            return float(np.clip(1.0 - abs(a - t) / max(abs(t), 1.0), 0.0, 0.9))
        if task == "sort":
            xs = [int(x) for x in ans.split(",")]
            ts = [int(x) for x in truth.split(",")]
            if not xs:
                return 0.0
            if strict:                       # 정확한 위치에 놓인 원소 비율
                hit = sum(1 for a, b in zip(xs, ts) if a == b)
                return float(np.clip(hit / len(ts), 0.0, 0.9))
            n_ok = sum(1 for a, b in zip(xs, xs[1:]) if a <= b)      # 국소 순서 보존율
            order = n_ok / max(len(xs) - 1, 1)
            keep = len(set(xs) & set(ts)) / len(ts)
            return float(np.clip(0.5 * order + 0.5 * keep, 0.0, 0.9))
        if task == "strcount":
            d = abs(int(ans) - int(truth))
            return float(np.clip(np.exp(-d / (1.0 if strict else 2.0)), 0.0, 0.9))
        obj, tobj = jsonlib.loads(ans), jsonlib.loads(truth)         # jsonfmt
        if not isinstance(obj, dict):
            return 0.0
        hit = sum(1 for k, v in tobj.items() if obj.get(k) == v)
        prec, rec = hit / max(len(obj), 1), hit / max(len(tobj), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        return float(np.clip(f1 ** 2 if strict else f1, 0.0, 0.9))   # 엄격: F1 제곱
    except Exception:
        return 0.0                                                  # 파싱 불가 = 0점


def _norm_answer(s: str) -> str:
    s = s.strip().strip(".").strip()
    s = re.sub(r"\s+", " ", s)
    try:  # JSON이면 정규화
        return jsonlib.dumps(jsonlib.loads(s), sort_keys=True, separators=(",", ":"))
    except Exception:
        return s.lower()


def extract_answer(text: str) -> str | None:
    """출력 텍스트에서 답 추출 — 라벨링과 verifier가 공유 (truth 미사용)."""
    m = re.search(r"final answer:\s*(.+?)\s*$", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:it is|answer is|result is)\s*(.+?)\s*(?:maybe)?\.?\s*$",
                  text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


# ---------------- 태스크 생성기 (난이도 d ∈ [0,1] 내장) ----------------

def _gen_arith(rng, d):
    k = 1 + int(d * 2.4)                                   # 연산 단계 1~3
    hi = int(10 ** (1 + 2.2 * d))                          # 피연산자 크기
    vals = rng.integers(2, max(hi, 3), size=k + 1)
    ops = rng.choice(["+", "-", "*"], size=k)
    expr = str(vals[0])
    for o, v in zip(ops, vals[1:]):
        expr = f"({expr} {o} {v})"
    truth = str(eval(expr))
    return f"Compute the value of {expr}. End with 'Final answer: <number>'.", truth


def _gen_sort(rng, d):
    n = 4 + int(d * 14)
    xs = rng.choice(np.arange(1, 999), size=n, replace=False)
    truth = ", ".join(map(str, sorted(xs)))
    return (f"Sort the following {n} numbers in ascending order and output them "
            f"comma-separated: {', '.join(map(str, xs))}. "
            f"End with 'Final answer: <sorted list>'."), truth


def _gen_strcount(rng, d):
    n = 6 + int(d * 34)
    letters = rng.choice(list("abcdefg"), size=n)
    word = "".join(letters)
    target = str(rng.choice(list("abcdefg")))
    truth = str(word.count(target))
    return (f"How many times does the letter '{target}' appear in the string "
            f"'{word}'? End with 'Final answer: <count>'."), truth


def _gen_jsonfmt(rng, d):
    k = 2 + int(d * 4)
    keys = [f"field_{c}" for c in "abcdefg"[:k]]
    vals = [int(v) for v in rng.integers(0, 100, size=k)]
    obj = dict(zip(keys, vals))
    desc = ", ".join(f"'{a}' set to {b}" for a, b in obj.items())
    truth = jsonlib.dumps(obj, sort_keys=True, separators=(",", ":"))
    return (f"Return a JSON object with {desc}. "
            f"End with 'Final answer: <json>'."), truth


_GEN = {"arith": _gen_arith, "sort": _gen_sort, "strcount": _gen_strcount,
        "jsonfmt": _gen_jsonfmt}


# ---------------- 모의 후보 모델 (능력 의존 오류 과정 → 실출력 텍스트) ----------------

def _perturb(rng, task, truth):
    """그럴듯한 오답 생성 (실제 LLM 오답 패턴 모사)."""
    if task == "arith":
        v = int(truth)
        return str(v + int(rng.integers(1, 9)) * (1 if rng.random() < 0.5 else -1))
    if task == "sort":
        xs = [int(x) for x in truth.split(",")]
        if rng.random() < 0.5 and len(xs) > 4:             # 원소 누락 (오름차순 유지)
            xs.pop(int(rng.integers(0, len(xs))))
        else:                                              # 인접 교환 (순서 파괴)
            j = int(rng.integers(0, len(xs) - 1))
            xs[j], xs[j + 1] = xs[j + 1], xs[j]
        return ", ".join(map(str, xs))
    if task == "strcount":
        return str(max(0, int(truth) + int(rng.choice([-2, -1, 1, 2]))))
    obj = jsonlib.loads(truth)                             # jsonfmt
    keys = list(obj)
    if rng.random() < 0.5:
        obj[keys[int(rng.integers(0, len(keys)))]] = int(rng.integers(0, 100))
    else:
        obj.pop(keys[int(rng.integers(0, len(keys)))])
    return jsonlib.dumps(obj, sort_keys=True, separators=(",", ":"))


def _render_output(rng, cap, ans, correct):
    """능력 의존 스타일: 포맷 파괴·반복·얼버무림·거부·장황함."""
    fill = " ".join(FILLERS[int(rng.integers(0, len(FILLERS)))]
                    for _ in range(1 + int(cap * 3)))
    r = rng.random()
    if r < 0.06 * (1 - cap) and not correct:               # 거부 (답 없음)
        return REFUSAL
    if rng.random() < 0.12 * (1 - cap):                    # 반복 오염
        fill += " " + " ".join(["The result should follow."] * int(rng.integers(3, 8)))
    if rng.random() < 0.22 * (1 - cap) + 0.02:             # 포맷 파괴 (마커 누락)
        hedge = HEDGES[int(rng.integers(0, len(HEDGES)))] + " " \
            if (not correct and rng.random() < 0.5) else ""
        return f"{fill} {hedge}it is {ans} maybe."
    return f"{fill} Final answer: {ans}"


@dataclass
class TextWorld:
    prompts: list[str]
    domains: np.ndarray
    difficulty: np.ndarray
    truth: list[str]
    samples: list  # samples[i][m] = [S개 출력 텍스트]
    caps: np.ndarray
    seed: int = 42
    extras: dict = field(default_factory=dict)


def build_textworld(cfg, seed: int = 42, S: int = 5) -> TextWorld:
    rng = np.random.default_rng(seed)
    n = cfg["synth"]["n_queries"]
    # 모델별 능력치. 5모델 기본값. 모델 수가 다르면(A.X 3모델) config synth.textworld_caps로 교체.
    caps = np.array(cfg["synth"].get("textworld_caps", [0.28, 0.44, 0.58, 0.75, 0.88]))
    assert len(caps) == len(cfg["models"]), \
        f"textworld_caps 길이({len(caps)})가 모델 수({len(cfg['models'])})와 불일치 — config 확인"
    domains = rng.integers(0, len(TASKS), size=n)
    difficulty = rng.beta(2, 2, size=n)
    prompts, truth, samples = [], [], []
    for i in range(n):
        task = TASKS[domains[i]]
        p, t = _GEN[task](rng, difficulty[i])
        prompts.append(p)
        truth.append(t)
        row = []
        for c in caps:
            p_ok = 0.03 + 0.94 / (1 + np.exp(-7 * (c - difficulty[i])))
            outs = []
            for _ in range(S):
                ok = rng.random() < p_ok
                ans = t if ok else _perturb(rng, task, t)
                outs.append(_render_output(rng, c, ans, ok))
            row.append(outs)
        samples.append(row)
    return TextWorld(prompts, domains, difficulty, truth, samples, caps, seed)


# ---------------- Dataset 조립 (투표 상자 = 실제 다수결) ----------------

def to_dataset(tw: TextWorld, cfg, Ns=(1, 3, 5), graded: bool = False,
               scale: str = "strict"):
    """확장 Dataset(15 가상 모델) + verifier용 텍스트 메타(열별 대표출력·vote_share).

    verifier 행렬은 fold별로 적합해야 하므로 여기서는 0으로 두고 메타만 반환.

    graded=True (B4, World F-c): 품질 라벨을 이진 대신 **부분점수**로 매긴다
    (graded_score 참조). 투표 상자는 다수결 답의 부분점수를 취한다 — 다수결 구조는
    그대로 두고 라벨 척도만 연속으로 바꾸므로, 이진 대비 A/B가 공정하다.
    """
    from .schema import Dataset, ModelMeta
    from .cost_mirror import vote_prefill_mult
    n, M = len(tw.prompts), len(tw.caps)
    base_ids = list(cfg["models"].keys())
    model_ids, models = [], {}
    quality = np.zeros((n, M * len(Ns)))
    out_tokens = np.zeros((n, M * len(Ns)), dtype=int)
    rep_text = [[None] * (M * len(Ns)) for _ in range(n)]
    vote_share = np.ones((n, M * len(Ns)))
    truth_norm = [_norm_answer(t) for t in tw.truth]

    for m, mid in enumerate(base_ids):
        meta = cfg["models"][mid]
        for j, N in enumerate(Ns):
            vid = mid if N == 1 else f"{mid}#v{N}"
            model_ids.append(vid)
            models[vid] = ModelMeta(vid, meta["prefill_price"] * vote_prefill_mult(cfg, N),
                                    meta["decode_price"])
    tasks = [TASKS[d] for d in tw.domains]
    for i in range(n):
        for m in range(M):
            outs = tw.samples[i][m]
            answers = [extract_answer(o) for o in outs]
            norms = [(_norm_answer(a) if a is not None else "__none__") for a in answers]
            for j, N in enumerate(Ns):
                col = m * len(Ns) + j
                cnt = Counter(norms[:N])
                maj, c_maj = cnt.most_common(1)[0]
                idx = norms[:N].index(maj)
                rep_text[i][col] = outs[idx]
                vote_share[i, col] = c_maj / N
                quality[i, col] = (
                    graded_score(tasks[i], answers[idx], tw.truth[i], scale) if graded
                    else float(maj != "__none__" and maj == truth_norm[i]))
                out_tokens[i, col] = sum(max(len(o) // 4, 8) for o in outs[:N])

    in_tokens = np.array([max(len(p) // 4, 12) for p in tw.prompts])
    ds = Dataset(model_ids=model_ids, models=models, domains=tw.domains,
                 features=np.zeros((n, 1)), in_tokens=in_tokens, out_tokens=out_tokens,
                 quality=quality, verifier=np.zeros_like(quality),
                 world="textworld-graded" if graded else "textworld",
                 extras={"difficulty": tw.difficulty, "true_p": None})
    meta = {"rep_text": rep_text, "vote_share": vote_share, "prompts": tw.prompts,
            "tasks": tasks}
    return ds, meta
