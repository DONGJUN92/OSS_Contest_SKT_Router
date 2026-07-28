"""검증기의 **실텍스트** 예리함 측정 + 손익분기 게이트 (Phase 17 — 레드팀 CRITICAL #1).

왜 이 프로브가 이 저장소의 가장 중요한 측정인가
--------------------------------------------------
챌린지 런타임은 품질 점수를 주지 않는다. 호출하면 **출력 텍스트**가 오고, 그 출력이 맞았는지는
라우터가 스스로 판정해야 한다. 그 판정값 v 가 σ(예약값)·정지 규칙·최종 답 선택 **전부**에
들어가므로, v 가 정보를 잃으면 순차 개봉 구조 전체가 비용만 남는다.

측정 결과 (Phase 17 진입 시점): 자작 `ScoringVerifier` 의 13특성을 실물 RouterBench 자유형식
응답에 붙이면 held-out AUC **0.610** 으로, 자작 textworld 기준값 0.873~0.90 에서 붕괴했다.
특성별로 `has_marker` 0.002 · `parse_ok` 0.048 · `struct` std 0.044 · `vote_share` 상수 ·
`hedge` 0.000 — 9개 중 5개가 죽는다. 텍스트를 무시하고 상자 원-핫만 쓴 검증기가 0.699 로
**더 높다**(= 현행 검증기는 IRT 가 이미 아는 사전확률보다 적은 정보를 준다).

★ 손익분기: `eval/probe_verifier_sensitivity.py` 가 검증기를 단계적으로 열화시켜 측정한
LPB vs 학습형 단일호출의 교차점은 **AUC ≈ 0.845** 다. 그 아래에서 LPB 는 학습형보다 나쁘다
(추가 호출이 정보를 못 주므로 순수 비용). 따라서 이 프로브는 단순 리포트가 아니라 **게이트**다.

사용법: python eval/probe_verifier_real.py [--n 1500] [--dim 100]
"""
import sys, pathlib, json, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from src.verifier import (ScoringVerifier, extract_features, feature_names,
                          augmented_feature_matrix, augmented_feature_names)

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "results"
BREAK_EVEN = 0.845          # probe_verifier_sensitivity.py 가 측정한 LPB=learned 교차점

MODELS = ["gpt-4-1106-preview", "gpt-3.5-turbo-1106", "claude-v1", "claude-v2",
          "claude-instant-v1", "meta/llama-2-70b-chat", "mistralai/mixtral-8x7b-chat",
          "mistralai/mistral-7b-chat", "zero-one-ai/Yi-34B-Chat",
          "meta/code-llama-instruct-34b-chat", "WizardLM/WizardLM-13B-V1.2"]


def auc(scores, labels) -> float:
    s, y = np.asarray(scores).ravel(), np.asarray(labels).ravel()
    pos, neg = s[y > 0.5], s[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _bench(e):
    e = str(e).lower()
    for k in ["mmlu", "hellaswag", "gsm8k", "arc", "winogrande", "mbpp", "mt-bench"]:
        if k in e:
            return k
    return "other"


def load_meta(n_prompts: int):
    """실물 RouterBench → `(meta, quality)` — textworld 의 (Dataset, meta) 계약과 같은 모양."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    df = pd.read_pickle(hf_hub_download("withmartian/routerbench",
                                        "routerbench_0shot.pkl", repo_type="dataset"))
    sub = df.sample(n_prompts, random_state=0).reset_index(drop=True)
    cols = [m for m in MODELS if f"{m}|model_response" in sub.columns]
    n = len(sub)
    rep = [[str(sub[f"{m}|model_response"].iloc[i]) for m in cols] for i in range(n)]
    q = np.clip(sub[cols].to_numpy(dtype=float), 0.0, 1.0)
    meta = {"prompts": [str(p) for p in sub["prompt"]],
            "tasks": [_bench(e) for e in sub["eval_name"]],
            "rep_text": rep,
            # 단일 응답 = 실전(Q2=No) 조건. 자기일관성 신호는 존재하지 않는다.
            "vote_share": np.ones((n, len(cols)))}
    return meta, q, cols


def _split(n_rows, seed=0, frac=0.7):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rows)
    k = int(frac * n_rows)
    return perm[:k], perm[k:]


def main(n_prompts=1500, dim=100):
    t0 = time.time()
    meta, q, cols = load_meta(n_prompts)
    n, M = q.shape
    yb = (q > 0.5).astype(float).ravel()
    vocab = sorted(set(meta["tasks"]))
    print(f"[probe_verifier_real] 실물 RouterBench 응답 {n}프롬프트 × {M}모델 = {n * M}셀, "
          f"양성률 {yb.mean():.3f}", flush=True)

    # ---- 현행 경로 (13특성) ----
    base = np.stack([[extract_features(meta["prompts"][i], meta["rep_text"][i][c],
                                       1.0, meta["tasks"][i], vocab)
                      for c in range(M)] for i in range(n)])
    d0 = base.shape[2]
    n_dom = len(vocab)
    variants = {
        "현행 ScoringVerifier (9특성+도메인원핫)": base.reshape(-1, d0),
        "현행 9특성만 (런타임 task 라벨 없음)": base[:, :, :9].reshape(-1, 9),
    }
    # ---- 증강 경로 ----
    aug = augmented_feature_matrix(meta, domain_vocab=vocab, text_dim=dim,
                                   use_prompt=True, use_agreement=True, ref_col=0)
    names = augmented_feature_names(vocab, dim, M, True, True)
    A = aug.reshape(-1, aug.shape[2])
    from src.verifier import _TAIL_FEATURES
    i_tail0 = 9 + n_dom
    n_tail = len(_TAIL_FEATURES)
    i_agree = i_tail0 + n_tail
    i_rtxt = i_agree + 1
    # tail 블록을 두 단계로 쪼개 **실도메인 구조 검사(`tail_struct_task`)의 순기여**를 분리한다
    j_task = _TAIL_FEATURES.index("tail_struct_task")
    keep = list(range(i_tail0)) + [i_tail0 + k for k in range(n_tail) if k != j_task]
    variants["+ tail/형식 특성 (도메인 검사 제외)"] = A[:, keep]
    variants["+ 실도메인 STRUCT_CHECKS"] = A[:, :i_agree]
    variants["+ 교차모델 답 일치(agree_ref)"] = A[:, :i_rtxt]
    variants["+ 응답 hashing"] = A[:, :i_rtxt + dim]
    variants["+ 프롬프트 hashing"] = A[:, :i_rtxt + 2 * dim]
    variants["+ 상자 원-핫 (= 전체 증강)"] = A
    variants["[대조] 상자 원-핫만 (텍스트 무시)"] = A[:, -M:]

    tr, te = _split(len(yb))
    rows = {}
    print(f"\n  {'AUC':<8}{'Δ':<9}{'게이트':<8}변형   (손익분기 {BREAK_EVEN})")
    prev = None
    for name, X in variants.items():
        sv = ScoringVerifier(l2=1e-3).fit(X[tr], yb[tr], steps=2500, lr=0.2)
        a = auc(sv.score(X[te]), yb[te])
        delta = "" if prev is None or name.startswith("[") else f"{a - prev:+.3f}"
        gate = "PASS" if a >= BREAK_EVEN else "FAIL"
        rows[name] = round(a, 4)
        print(f"  {a:<8.3f}{delta:<9}{gate:<8}{name}", flush=True)
        if not name.startswith("[") and not name.startswith("현행 9"):
            prev = a
    best = max(rows.values())
    verdict = ("PASS — 증강 검증기가 손익분기를 넘겼다" if best >= BREAK_EVEN else
               "FAIL — 증강해도 손익분기 미달. 이 데이터에서는 LPB 의 순차 관측이 "
               "학습형 단일호출 대비 순가치를 내지 못한다고 보고해야 한다")
    print(f"\n  판정: {verdict}  (best {best:.3f} vs gate {BREAK_EVEN})")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"n_prompts": n, "n_models": M, "n_cells": int(len(yb)),
               "positive_rate": round(float(yb.mean()), 4), "text_dim": dim,
               "break_even_auc": BREAK_EVEN, "auc": rows, "best": best,
               "gate": "PASS" if best >= BREAK_EVEN else "FAIL", "verdict": verdict,
               "note": "실물 RouterBench 0-shot 자유형식 응답. 손익분기는 "
                       "eval/probe_verifier_sensitivity.py 측정값."},
              open(OUT / "probe_verifier_real.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"  ({time.time() - t0:.0f}s) → eval/results/probe_verifier_real.json")


if __name__ == "__main__":
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 1500
    d = int(sys.argv[sys.argv.index("--dim") + 1]) if "--dim" in sys.argv else 100
    main(n, d)
