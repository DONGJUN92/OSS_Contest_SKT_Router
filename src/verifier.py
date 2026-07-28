"""ScoringVerifier — 출력 텍스트 → 품질 신호 v ∈ [0,1] (Phase 8, 심사 권고 1).

파이프라인 접속 계약: Dataset.verifier[i, col] 행렬을 생성한다 (기존 하네스·엔진·
NoiseModel이 무변경으로 소비). fold별로 train 라벨로만 적합 → 전 행 점수화.

특성 3계층 (진짜 정답 접근 불가):
  ① 구조 검사 (HeuristicVerifier 흡수): 답 마커/추출 성공, 태스크별 검증 가능 속성
     (정렬-오름차순, JSON-스키마, 세기-범위 타당성) — 충분조건이 아닌 필요조건들
  ② 자기일관성: 다수결 샘플 내 합의율 (vote_share)
  ③ 통계 신호: 길이·반복률·얼버무림·프롬프트 반향(echo)
헤드: 로지스틱 회귀 (numpy, 기존 스택과 동일) — 출력이 곧 캘리브레이션된 확률.
"""
import json as jsonlib
import re

import numpy as np

from .textworld import extract_answer, TASKS

_HEDGE_RE = re.compile(r"not (fully )?sure|might be wrong|possibly|maybe|cannot reliably",
                       re.IGNORECASE)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def _repetition_ratio(text: str) -> float:
    words = text.lower().split()
    if len(words) < 4:
        return 0.0
    return 1.0 - len(set(words)) / len(words)


def _echo_ratio(prompt: str, text: str) -> float:
    pw, tw = set(prompt.lower().split()), text.lower().split()
    if not tw:
        return 0.0
    return sum(w in pw for w in tw) / len(tw)


# ---------------- 구조 검사 (태스크별 특화 + 미지 태스크 폴백) ----------------
#
# 구조 검사만이 태스크 특화다 (나머지 특성은 태스크 무관). 실데이터의 도메인은 다를
# 것이므로, 알려진 태스크는 특화 검사로 (textworld 결과 비트 단위 보존), **미지의
# 태스크는 타입-일반 폴백**으로 우아하게 퇴화시킨다 — 라우터의 use_domain 퇴화와
# 같은 철학. 도메인별 특화 검사는 실데이터 수령 시 여기 dict에 등록하면 된다
# (Phase 8이 확정한 유일한 남은 개선 경로, docs/branch_decision_table.md §미해결).

def _struct_sort(prompt: str, ans: str) -> float:
    try:
        xs = [int(x) for x in ans.split(",")]
        asked = re.search(r"following (\d+) numbers", prompt)
        len_ok = (asked is None) or (len(xs) == int(asked.group(1)))
        return float(all(a <= b for a, b in zip(xs, xs[1:])) and len_ok)
    except Exception:
        return 0.0


def _struct_jsonfmt(prompt: str, ans: str) -> float:
    try:
        obj = jsonlib.loads(ans)
        keys_asked = set(re.findall(r"'(field_\w)'", prompt))
        return float(isinstance(obj, dict) and keys_asked.issubset(obj.keys()))
    except Exception:
        return 0.0


def _struct_strcount(prompt: str, ans: str) -> float:
    try:
        m = re.search(r"'([a-g]+)'\?", prompt)
        return float(0 <= int(ans) <= (len(m.group(1)) if m else 10 ** 6))
    except Exception:
        return 0.0


def _struct_arith(prompt: str, ans: str) -> float:
    return float(bool(re.fullmatch(r"-?\d+", ans.strip())))


# ---------------- 실도메인 구조 검사 (Phase 18) ----------------
#
# Phase 8·13·14 가 세 번 반복해 확인한 결론: 검증기의 유일한 남은 개선 경로는 **실제 도메인의
# 구조 검사**다(태스크-불가지 특성 추가는 ΔAUC 0.0012 로 기각됐다). 자작 textworld 4태스크
# (sort/jsonfmt/strcount/arith) 밖의 도메인에서는 `_generic_struct` 폴백만 돌고 있었다.
#
# 실데이터(SKT) 도메인은 아직 모르지만, **실물 RouterBench 의 도메인은 안다** — 그리고 그것들은
# 실제 LLM 벤치마크의 대표 형태다: 객관식(mmlu/arc/hellaswag/winogrande) · 정수 최종답(gsm8k) ·
# 코드(mbpp). 이 세 형태의 검사기를 등록하고 실응답에서 기여를 측정한다. SKT 도메인이 오면
# 같은 자리에 이름만 추가하면 된다.

_CHOICE_RE = re.compile(r"^\(?([A-Ea-e1-5])[).:]?$")
_NUM_RE = re.compile(r"^[-+]?\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?%?$|^[-+]?\d+(?:\.\d+)?$")
_CODE_HINT_RE = re.compile(r"\b(def|return|import|class|for|while|lambda)\b")


def _unwrap(a: str) -> str:
    """리스트/따옴표/마크다운 포장을 벗긴다.

    ★ Phase 18 진단에서 발견: 실물 RouterBench 의 객관식 응답은 `['D']` 형태로 온다.
    포장을 안 벗기면 검사기가 **모든 행에서 0** 을 내고(mean 0.000, std 0.000) 특성이
    통째로 죽는다. 처음엔 이걸 "구조 검사는 실텍스트에서 무용"이라고 오독할 뻔했다 —
    실제로는 파서 버그였다. 형식 가정은 항상 데이터로 확인할 것.
    """
    s = ans_strip = a.strip()
    for pair in ("[]", "()", "''", '""', "``"):
        while len(s) >= 2 and s[0] == pair[0] and s[-1] == pair[1]:
            s = s[1:-1].strip()
    return s or ans_strip


def _struct_choice(prompt: str, ans: str) -> float:
    """객관식 — 답이 보기 문자 하나인가, 그리고 프롬프트에 있는 보기인가."""
    a = _unwrap(ans.strip().strip("*` "))
    m = _CHOICE_RE.match(a)
    if not m:
        # 보기 문자로 시작하는 짧은 답("A) Paris")도 형식은 맞다고 본다
        return 0.6 if re.match(r"^\(?[A-Ea-e1-5][).:]\s+\S", a) and len(a) < 80 else 0.0
    letter = m.group(1).upper()
    opts = set(x.upper() for x in re.findall(r"(?:^|\n)\s*\(?([A-Ea-e1-5])[).:]\s", prompt))
    return 1.0 if (not opts or letter in opts) else 0.3


def _struct_number(prompt: str, ans: str) -> float:
    """수치 최종답 — 깔끔한 수 하나인가 (통화·천단위 구분·퍼센트 허용)."""
    a = _unwrap(ans.strip().strip("*`. "))
    if _NUM_RE.match(a):
        return 1.0
    # 문장 끝에 수가 하나만 있는 형태도 부분 인정
    nums = re.findall(r"-?\d+(?:\.\d+)?", a)
    if len(nums) == 1 and len(a.split()) <= 8:
        return 0.6
    return 0.0 if not nums else 0.2


def _struct_code(prompt: str, ans: str) -> float:
    """코드 — 괄호가 균형 잡혔고 코드 토큰이 있는가 (정답성은 판정하지 않는다)."""
    a = ans.strip()
    if not a:
        return 0.0
    bal = all(a.count(o) == a.count(c) for o, c in (("(", ")"), ("[", "]"), ("{", "}")))
    has_code = bool(_CODE_HINT_RE.search(a))
    if bal and has_code:
        return 1.0
    if has_code:
        return 0.5
    return 0.2 if bal and len(a.split()) <= 40 else 0.0


# task 이름 → 구조 검사기. **실데이터 도메인 검사기를 여기에 추가한다.**
STRUCT_CHECKS = {
    # 자작 textworld (하위호환 — 비트 동일)
    "sort": _struct_sort, "jsonfmt": _struct_jsonfmt,
    "strcount": _struct_strcount, "arith": _struct_arith,
    # 실 벤치마크 도메인 (Phase 18 신설) + 형태 기반 별칭
    "mmlu": _struct_choice, "arc": _struct_choice, "hellaswag": _struct_choice,
    "winogrande": _struct_choice, "choice": _struct_choice, "multichoice": _struct_choice,
    "gsm8k": _struct_number, "math": _struct_number, "number": _struct_number,
    "mbpp": _struct_code, "humaneval": _struct_code, "code": _struct_code,
}


def _generic_struct(ans: str) -> float:
    """타입-일반 형식 타당성 — 태스크 라벨/특화 검사가 없을 때의 폴백.

    정답 여부(정렬됐는가·키가 맞는가)는 태스크 지식이 필요하므로 포기하고, 답이
    **탐지된 타입에 대해 잘 형성됐는가**만 잡는다: 파싱 가능한 JSON/수/수열은 1.0,
    짧고 깨끗한 토큰은 0.7, 장황한 산문(얼버무림·형식 파괴의 신호)은 0.3, 공백은 0.0.
    합성·실데이터 공통으로 '오답은 형식 파괴를 동반하는 경향'을 신호로 쓴다."""
    a = ans.strip()
    if not a:
        return 0.0
    try:
        jsonlib.loads(a)
        return 1.0
    except Exception:
        pass
    if re.fullmatch(r"-?\d+(?:\.\d+)?", a):                 # 깨끗한 수
        return 1.0
    if re.fullmatch(r"-?\d+(?:\s*,\s*-?\d+)+", a):          # 수열 (콤마 구분)
        return 1.0
    return 0.7 if len(a.split()) <= 3 else 0.3             # 짧은 토큰 vs 장황/산문


_BASE_FEATURES = ["has_marker", "parse_ok", "struct", "vote_share", "out_len",
                  "repetition", "hedge", "echo", "ans_len"]

# ---------------- 증강 특성 (Phase 17, 레드팀 지적 #1) ----------------
#
# ★ 왜 필요한가 — 실측이 이유다. 위 9특성을 **실물 자유형식 LLM 응답**(RouterBench 11모델
# × 3,000프롬프트 = 33,000셀)에 붙여 재보니 held-out AUC 0.610 으로, 자작 textworld 기준값
# (0.873~0.90)에서 붕괴했다. 특성별 진단:
#     has_marker  mean 0.002   ← 실제 모델은 "Final answer:" 를 쓰지 않는다
#     parse_ok    mean 0.048   ← 응답 95%가 마커 기반 답 추출 실패
#     struct      std  0.044   ← 거의 항상 "판정불가 0.5" 상수 = 정보 0
#     vote_share  상수 1.000   ← 단일 샘플이면 자기일관성 신호가 없다
#     hedge       mean 0.000
# 즉 9특성 중 5개가 실텍스트에서 죽는다. 결정적 대조: 텍스트를 **완전히 무시**하고 모델
# 원-핫만 쓴 검증기가 AUC 0.699 로 현행(0.611)보다 높다 — 현행 검증기는 실텍스트에서
# IRT 가 이미 갖고 있는 사전확률 p̄_m 보다 적은 정보를 뽑아낸다.
#
# 증강 블록은 세 축을 더한다 (`eval/probe_verifier_real.py` 가 각각의 기여를 측정):
#   ① 마커 비의존 tail 파싱·형식 신호 (task-불가지)
#   ② 응답 텍스트 hashing 인코딩  — "무엇을 답했는가"
#   ③ 프롬프트 hashing + 상자 원-핫 — "어떤 문항을, 어떤 모델이"
# 기존 13특성 경로는 **비트 단위 보존**된다 (기본 OFF, `tests/test_regression.py` 고정).

_TAIL_FEATURES = ["tail_parse_ok", "tail_struct", "tail_len", "digit_ratio",
                  "n_lines", "nonalnum_ratio", "uniq_char_ratio", "prompt_len_ratio",
                  # Phase 18: 마커 파싱이 실패해도 **도메인 구조 검사**를 tail 답에 적용한다.
                  # 기존 `struct`(index 2)는 parse_ok=False 면 0.5 상수로 죽어 실텍스트에서
                  # 정보가 0 이었다 — 실측 std 0.044. 이 특성이 그 경로를 되살린다.
                  "tail_struct_task"]


def feature_names(domain_vocab=TASKS) -> list[str]:
    """특성 이름 — 도메인 원-핫이 데이터의 도메인 어휘를 따른다 (기본=textworld TASKS)."""
    return _BASE_FEATURES + [f"dom_{d}" for d in domain_vocab]


def augmented_feature_names(domain_vocab=TASKS, text_dim: int = 64,
                            n_cols: int = 0, use_prompt: bool = True,
                            use_agreement: bool = True) -> list[str]:
    names = feature_names(domain_vocab) + list(_TAIL_FEATURES)
    if use_agreement:
        names += ["agree_ref"]
    names += [f"rtxt_{i}" for i in range(text_dim)]
    if use_prompt:
        names += [f"ptxt_{i}" for i in range(text_dim)]
    names += [f"box_{i}" for i in range(n_cols)]
    return names


def _tail_answer(text: str) -> str:
    """마커 비의존 답 추출 — 마지막 비어있지 않은 줄.

    `extract_answer` 는 라벨링과 공유되므로 건드리면 textworld 품질 라벨이 바뀐다.
    그래서 검증기 전용 완화 파서를 따로 둔다 (실측: 실응답의 95%가 마커 파싱 실패).
    """
    for line in reversed(text.strip().splitlines()):
        s = line.strip()
        if s:
            return s[:200]
    return ""


def _tail_features(prompt: str, output: str, task: str = "") -> np.ndarray:
    tail = _tail_answer(output)
    body = output.strip()
    digits = sum(c.isdigit() for c in body)
    alnum = sum(c.isalnum() or c.isspace() for c in body)
    checker = STRUCT_CHECKS.get(task)
    try:
        struct_task = checker(prompt, tail) if checker else _generic_struct(tail)
    except Exception:                                           # 검사기는 절대 죽지 않는다
        struct_task = 0.5
    return np.array([
        float(bool(tail)),                                      # tail_parse_ok
        _generic_struct(tail),                                  # tail_struct
        min(len(tail) / 60.0, 2.0),                             # tail_len
        digits / max(len(body), 1),                             # digit_ratio
        min(body.count("\n") / 10.0, 2.0),                      # n_lines
        1.0 - alnum / max(len(body), 1),                        # nonalnum_ratio
        len(set(body.lower())) / max(len(body), 1),             # uniq_char_ratio
        min(len(body) / max(len(prompt), 1), 4.0),              # prompt_len_ratio
        float(struct_task),                                     # tail_struct_task
    ])


def extract_features(prompt: str, output: str, vote_share: float, task: str,
                     domain_vocab=TASKS) -> np.ndarray:
    ans = extract_answer(output)
    has_marker = float("final answer:" in output.lower())
    parse_ok = float(ans is not None)
    ans = ans or ""
    # 구조 검사 (0=위반, 0.5=판정불가, 1=통과). 알려진 태스크는 특화 검사, 미지는 폴백.
    if not parse_ok:
        struct = 0.5
    elif task in STRUCT_CHECKS:
        struct = STRUCT_CHECKS[task](prompt, ans)
    else:
        struct = _generic_struct(ans)

    dom = [float(task == t) for t in domain_vocab]
    return np.array([
        has_marker, parse_ok, struct, vote_share,
        min(len(output) / 400.0, 2.0),
        _repetition_ratio(output),
        float(bool(_HEDGE_RE.search(output))),
        _echo_ratio(prompt, output),
        min(len(ans) / 60.0, 2.0),
        *dom,
    ])


FEATURE_NAMES = feature_names()               # textworld 기본 (하위호환)


class ScoringVerifier:
    """로지스틱 헤드. fit은 train 라벨만 사용, score는 특성만 사용."""

    def __init__(self, l2: float = 1e-4):
        self.l2 = l2

    def fit(self, X: np.ndarray, y: np.ndarray, steps: int = 3000, lr: float = 0.3):
        X1 = np.hstack([X, np.ones((len(X), 1))])
        self.w = np.zeros(X1.shape[1])
        m = np.zeros_like(self.w)
        v = np.zeros_like(self.w)
        for t in range(1, steps + 1):
            p = _sigmoid(X1 @ self.w)
            g = X1.T @ ((p - y) / len(y)) + self.l2 * self.w
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            self.w -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X1 = np.hstack([X, np.ones((len(X), 1))])
        return _sigmoid(X1 @ self.w)


def feature_matrix(meta: dict, domain_vocab=None) -> np.ndarray:
    """(N, cols) 셀 전체의 특성 행렬 — fold 불변이므로 1회 계산·캐시.

    domain_vocab: 도메인 원-핫 어휘. None이면 모든 task가 textworld TASKS에 속할 때
    TASKS로 고정(하위호환·비트동일)하고, 아니면 데이터의 유일 도메인으로 자동 도출한다
    (실데이터 — 미지 도메인에서도 원-핫이 살아있게).
    """
    prompts, tasks = meta["prompts"], meta["tasks"]
    if domain_vocab is None:
        domain_vocab = TASKS if set(tasks) <= set(TASKS) else sorted(set(map(str, tasks)))
    rep, vs = meta["rep_text"], meta["vote_share"]
    n, cols = vs.shape
    F = np.zeros((n, cols, len(feature_names(domain_vocab))))
    for i in range(n):
        for c in range(cols):
            F[i, c] = extract_features(prompts[i], rep[i][c], vs[i, c], tasks[i],
                                       domain_vocab)
    return F


def augmented_feature_matrix(meta: dict, domain_vocab=None, text_dim: int = 64,
                             use_prompt: bool = True, use_agreement: bool = True,
                             ref_col: int = 0, encoder=None) -> np.ndarray:
    """(N, cols, d') 증강 특성 행렬 — 실텍스트에서 신호를 되찾기 위한 경로 (Phase 17).

    `feature_matrix` 의 13특성 경로는 **비트 단위 보존**되고(기본), 이 함수는 그 위에
    다음을 덧붙인다:
      · `_TAIL_FEATURES`  — 마커 비의존 tail 파싱 + task-불가지 형식 신호
      · `agree_ref`       — **교차모델 답 일치**: 이 상자의 tail 답이 기준 상자(ref_col,
        보통 최저가 = 실제로 가장 먼저 열리는 상자)의 답과 같은가. 규칙상 무료인 신호이며
        (이미 호출한 출력만 본다) 자기일관성(`vote_share`)이 단일 샘플에서 상수로 죽는
        체제의 대체물이다. ref_col 자신은 판정 불가라 0.5.

        ★★ **런타임 가용성 조건 (반드시 읽을 것 — Phase 18 자체 감사)**
        이 특성은 **기준 상자를 이미 열었을 때만** 존재한다. 정적 `Dataset.verifier` 행렬은
        모든 셀을 미리 채우므로 오프라인 AUC 는 "기준 상자를 항상 열었다"고 **가정**한다.
        상자를 **1회만** 여는 쿼리에서는 그 1회가 보통 기준 상자 자신이고 `agree_ref = 0.5`
        (무정보)다. 따라서 전 셀 평균 AUC 개선(0.670 → 0.848)이 그대로 전이되지는 않는다.
        ★ 그런데 이 조건은 **λ 가 만든다**: `FINE_MULTS` 격자가 σ<0 비율을 0.70 → 0.58 로
        낮춰 일부 쿼리를 2회 이상 열리게 하면 거기서 이 특성이 살아난다. 실측
        (`eval/results/probe_redteam_closure.json`, λ 레버 적용 후):
            fast     현행 13특성 0.3596 → 증강 0.3638  (+0.0042)
            premium  현행 13특성 0.7331 → 증강 0.7427  (+0.0096)
        λ 레버 **적용 전** 실행에서는 fast 가 오히려 −0.0069 였다 — 즉 두 레버는 상보적이고
        따로 재면 각각을 과소평가한다 (상세: `docs/reflections/phase18.md` §3 caveat 3).
        ※ 완전한 형태는 호출 이력에 따라 **동적으로** 재계산하는 것이고 `submission.py` 의
        배포 경로는 그렇게 할 수 있다 — 다만 1회 호출 쿼리에서는 여전히 정보가 없다.
      · 응답/프롬프트 hashing 인코딩 + 상자 원-핫 — 실측 기여는
        `eval/probe_verifier_real.py` 가 축별로 분해해 보고한다.
    """
    from .text_encoder import HashingEncoder
    prompts, tasks = meta["prompts"], meta["tasks"]
    if domain_vocab is None:
        domain_vocab = TASKS if set(tasks) <= set(TASKS) else sorted(set(map(str, tasks)))
    rep, vs = meta["rep_text"], meta["vote_share"]
    n, cols = vs.shape
    enc = encoder or HashingEncoder(dim=text_dim - 4)     # +4 수치특성 = text_dim
    d = len(augmented_feature_names(domain_vocab, text_dim, cols, use_prompt, use_agreement))
    F = np.zeros((n, cols, d))
    memo: dict[str, np.ndarray] = {}

    def _enc(text: str) -> np.ndarray:
        if text not in memo:
            memo[text] = enc.encode([text])[0]
        return memo[text]

    for i in range(n):
        pv = _enc(prompts[i]) if use_prompt else None
        ref_ans = _norm_tail(rep[i][ref_col], str(tasks[i])) if use_agreement else None
        for c in range(cols):
            parts = [extract_features(prompts[i], rep[i][c], vs[i, c], tasks[i], domain_vocab),
                     _tail_features(prompts[i], rep[i][c], str(tasks[i]))]
            if use_agreement:
                parts.append(np.array([0.5 if c == ref_col else float(
                    _norm_tail(rep[i][c], str(tasks[i])) == ref_ans and bool(ref_ans))]))
            parts.append(_enc(rep[i][c]))
            if use_prompt:
                parts.append(pv)
            oh = np.zeros(cols)
            oh[c] = 1.0
            parts.append(oh)
            F[i, c] = np.concatenate(parts)
    return F


def _norm_tail(text: str, task: str = "") -> str:
    """일치 판정용 정규화 — 공백·대소문자·말미 구두점·포장 무시 + **태스크별 정규형**.

    ★ Phase 18: 교차모델 일치는 두 응답의 답이 **같은가**를 보는데, 문자열을 통째로 비교하면
    `['D']` · `D` · `The answer is D.` 가 전부 다른 답으로 잡힌다. 실물 RouterBench 진단에서
    객관식 응답이 `['D']` 로 온다는 것을 발견했고, 그렇다면 일치 판정도 **정규형**으로 해야
    한다. 객관식은 보기 문자, 수치형은 수 하나로 축약하고, 그 외에는 기존 문자열 정규화.
    이 신호는 규칙상 무료이고(이미 호출한 출력만 본다) 단일호출 정책은 쓸 수 없다.
    """
    raw = _unwrap(_tail_answer(text).strip()).lower().rstrip(".!?").strip()
    s = " ".join(raw.split())
    checker = STRUCT_CHECKS.get(task)
    if checker is _struct_choice:
        # ⚠ 단순히 `\b[a-e1-5]\b` 를 훑으면 산문의 관사 "a" 를 보기 문자로 오인해 **서로 틀린
        # 두 응답이 같은 답으로 잡힌다**(허위 일치 = 검증기 오염). 그래서 세 단계로 좁힌다:
        #   ① 토큰 하나면 그것이 답  ② 답 신호어 뒤의 문자  ③ 그 외에는 문자열 그대로
        if re.fullmatch(r"[a-e1-5]", s):
            return s
        m = re.search(r"(?:answer|option|choice|정답|답)\s*(?:is|was|:|=|은|는)?\s*"
                      r"\(?([a-e1-5])\)?\b", s)
        if m:
            return m.group(1)
        m = re.fullmatch(r"\(?([a-e1-5])\)?[.)\]:]?\s*\S{0,40}", s)   # "b) paris" 류
        if m:
            return m.group(1)
        return s
    if checker is _struct_number:
        nums = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
        return nums[-1] if nums else s
    return s


def canonical_matrix(meta: dict) -> list:
    """(N, cols) 정규형 답 문자열 — 교차모델 일치 판정의 원재료 (Phase 19)."""
    rep, tasks = meta["rep_text"], meta["tasks"]
    return [[_norm_tail(rep[i][c], str(tasks[i])) for c in range(len(rep[i]))]
            for i in range(len(rep))]


def agreement_fraction(canon: list) -> np.ndarray:
    """(N, cols) **대칭 일치율** — 이 상자의 답이 다른 상자들 중 몇 %와 같은가.

    `agree_ref`(기준 상자 1개 대비)의 일반화이며 학습용 통계다. 배포에서는 같은 양을
    **이미 호출한 상자들** 안에서 계산한다(`DynamicAgreementVerifier`) — 같은 추정량이고
    표본 수만 다르다. 정적 행렬 계약을 지키면서 동적 신호를 쓰기 위한 다리다.
    """
    n, cols = len(canon), len(canon[0])
    out = np.zeros((n, cols))
    for i in range(n):
        row = canon[i]
        for c in range(cols):
            a = row[c]
            if not a:
                out[i, c] = 0.0
                continue
            others = [row[k] for k in range(cols) if k != c]
            out[i, c] = sum(1.0 for o in others if o == a) / max(len(others), 1)
    return out


class DynamicAgreementVerifier:
    """호출 이력에 따라 관측 v 를 재계산하는 검증기 (Phase 19).

    왜: 상한 진단에서 남은 **달성 가능** 여지 +0.0674 중 88%(+0.0593)가 관측 채널에 있었고,
    그 여지는 2회 이상 여는 tier(balanced·premium)에 집중돼 있다. 정적 행렬은 일치 신호를
    "기준 상자 대비"로 고정해 그 여지에 접근하지 못한다.

    학습: 특성의 일치 슬롯에 **대칭 일치율**(전 상자 대비)을 넣고 로지스틱 헤드를 적합한다.
    배포: 같은 슬롯에 **이미 호출한 상자들 안에서의 일치율**을 넣는다.

    ★ **비교 대상이 없을 때 무엇을 넣는가 — 초판의 실패 (Phase 19 자체 발견)**
    초판은 0.5("판정 불가")를 넣었다. 그런데 이 슬롯의 학습 분포 평균이 데이터에 따라 매우
    작을 수 있고(textworld 실측 0.003), 그러면 0.5 는 **학습 분포 바깥의 값**이라 로지스틱
    헤드가 외삽해 관측 v 전체를 오염시킨다 — 실측 종합 −0.0217. 즉 "판정 불가"를 중간값으로
    표현한 것이 근본 오류였다.
    수정: 비교 대상이 없으면 **슬롯을 건드리지 않는다**(정적 학습값을 그대로 둔다). 그러면
    1회 호출 tier 에서 정적 경로와 정확히 일치하고, 2회 이상에서만 신호가 갱신된다.
    """

    def __init__(self, F_base: np.ndarray, agree_slot: int, canon: list, sv):
        self.F = F_base                     # (N, cols, d) — agree 슬롯은 학습값이 들어있음
        self.slot = int(agree_slot)
        self.canon = canon
        self.sv = sv

    def row_fn(self, i: int):
        """`Session.observe_fn` 규약 — (m, called) → v."""
        base = self.F[i]
        row = self.canon[i]

        def observe(m: int, called) -> float:
            peers = [k for k in called if k != m]
            a = row[m]
            if not peers or not a:
                # 비교 대상 없음 → 정적 학습값 유지 (분포 밖 값을 주입하지 않는다)
                return float(self.sv.score(base[m][None, :])[0])
            f = base[m].copy()
            f[self.slot] = sum(1.0 for k in peers if row[k] == a) / len(peers)
            return float(self.sv.score(f[None, :])[0])
        return observe


def fold_verifier_matrix(F: np.ndarray, quality: np.ndarray, train_idx: np.ndarray):
    """train 행으로 적합 → 전체 행 점수화한 verifier 행렬과 적합기 반환."""
    n, cols, d = F.shape
    sv = ScoringVerifier().fit(F[train_idx].reshape(-1, d),
                               quality[train_idx].ravel())
    V = sv.score(F.reshape(-1, d)).reshape(n, cols)
    return V, sv
