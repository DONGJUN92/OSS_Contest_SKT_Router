"""B6 + Q2=No 리플레이 — 점수표의 두 미검증 가정 해소 (위험 순 권고 1).

현행 6세계 점수표는 두 가지를 **가정**하고 있다. 둘 다 코드 변경 0줄이지만
점수표 재산출을 요구하는 **검증 부채**다 (docs/branch_decision_table.md Q2·Q6).

  b6    : 파산 의미론이 force_cheapest 라면? (Q6가 관대한 쪽으로 확정될 경우)
          현행 zero_quality(무응답=0점, 보수적)는 우리 점수의 하한이다. 관대한 규칙에서는
          우리도 오르지만 **파산하는 베이스라인이 더 크게 오를 수 있으므로**, 격차가
          좁혀지는지 확인해야 한다.
  q2no  : 다중 샘플이 제공되지 않는다면? (Q2=No) 투표 상자는 ablation 최대 기여
          (−0.178/−0.139)인데 현행 점수표는 **전부 투표 상자 포함** 구성이다.
          단일 샘플 조건의 서열 검증은 Phase 3b(World A·B)까지만 있었다.

두 경우 모두 정책 코드는 무변경 — 상자 집합과 채점 규칙만 바뀐다.

사용법: python run_b56.py {b6|q2no}
"""
import sys, pathlib, json

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

from src.cost_mirror import cost_matrix
from src.harness import run_tier, tier_budgets
from src.router import LPBRouter
from src.synth import make_world
from src.synth2 import make_world2
from src.synth_ext import extend_with_votes
from src.text_encoder import get_encoder
from src.textworld import build_textworld, to_dataset
from src.verifier import feature_matrix, fold_verifier_matrix
from baselines.policies import StaticCascade, tune_cascade_tau
from run_phase2 import CFG, OUT

TIERS = ["fast", "balanced", "premium"]
W = {t: CFG["tiers"][t]["weight"] for t in CFG["tiers"]}
SYNTH = ["irt", "specialist", "corr", "crossing", "nosignal"]
# 현행 점수표 (구성: 투표 상자 있음 · zero_quality) — 대조 기준
BASELINE = {"irt": 0.9486, "specialist": 0.9982, "corr": 0.9259,
            "crossing": 0.9907, "nosignal": 0.9615, "textworld": 0.8192}
CASCADE0 = {"irt": 0.6844, "specialist": 0.9985, "corr": 0.8690,
            "crossing": 0.9935, "nosignal": 0.7869, "textworld": 0.6344}


def build(wname, votes=True):
    """(dataset, folds, verifier_hook). votes=False면 투표 상자 없이 (Q2=No)."""
    if wname == "textworld":
        tw = build_textworld(CFG, seed=42)                 # Phase 8과 동일 구성
        ds, meta = to_dataset(tw, CFG, Ns=(1, 3, 5) if votes else (1,))
        ds.features = get_encoder("hashing").encode(meta["prompts"])
        F = feature_matrix(meta)
        folds = ds.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])
        return ds, folds, (lambda tr: fold_verifier_matrix(F, ds.quality, tr)[0])
    base = make_world(CFG, wname, seed=CFG["seed"]) if wname in ("irt", "specialist") \
        else make_world2(CFG, wname, seed=CFG["seed"])
    folds = base.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])
    if not votes:
        return base, folds, None
    conc = 0.6 if wname == "crossing" else None            # D14: 교차 세계는 집중 오답
    return extend_with_votes(base, CFG, conc=conc), folds, None


def evaluate(ds, folds, vhook, bankruptcy="zero_quality"):
    """6세계 공통 평가: (LPB 종합, cascade 종합, tier별 상세)."""
    order = list(np.argsort(cost_matrix(ds).mean(axis=0)))
    rows = {}
    for tier in TIERS:
        lq, cq, un = [], [], []
        for f in range(CFG["eval"]["k_folds"]):
            te = folds[f]
            tr = np.setdiff1d(np.arange(ds.n), te)
            if vhook is not None:
                ds.verifier = vhook(tr)
            router = LPBRouter(CFG, CFG["synth"]["n_domains"], seed=f).fit(ds, tr, tier)
            b_te = tier_budgets(ds, te, CFG)[tier]
            r = run_tier(ds, te, router.policy(), tier, b_te, bankruptcy=bankruptcy)
            lq.append(r.mean_quality)
            un.append(100.0 * r.unanswered / len(te))
            tau = tune_cascade_tau(ds, tr, CFG, tier, tier_budgets(ds, tr, CFG)[tier])
            cq.append(run_tier(ds, te, StaticCascade(order, tau), tier, b_te,
                               bankruptcy=bankruptcy).mean_quality)
        rows[tier] = {"lpb": round(float(np.mean(lq)), 4),
                      "std": round(float(np.std(lq)), 4),
                      "cascade": round(float(np.mean(cq)), 4),
                      "unans_pct": round(float(np.mean(un)), 2)}
    return (sum(W[t] * rows[t]["lpb"] for t in TIERS),
            sum(W[t] * rows[t]["cascade"] for t in TIERS), rows)


def run(tag, votes, bankruptcy):
    head = ("파산 의미론 = force_cheapest (Q6 관대)" if bankruptcy != "zero_quality"
            else "투표 상자 제거 Ns=(1,) (Q2=No)")
    print(f"[{tag}] {head} — 정책 코드 무변경, 6세계 재산출\n")
    print(f"  {'world':<12}{'LPB':>9}{'기준':>9}{'Δ':>9}   "
          f"{'cascade':>9}{'기준':>9}{'Δ':>9}   {'격차':>8}{'기준격차':>10}")
    out = {}
    for w in SYNTH + ["textworld"]:
        ds, folds, vhook = build(w, votes=votes)
        cl, cc, rows = evaluate(ds, folds, vhook, bankruptcy)
        gap, gap0 = cl - cc, BASELINE[w] - CASCADE0[w]
        out[w] = {"lpb": round(cl, 4), "cascade": round(cc, 4),
                  "lpb_ref": BASELINE[w], "cascade_ref": CASCADE0[w],
                  "gap": round(gap, 4), "gap_ref": round(gap0, 4), "tiers": rows}
        print(f"  {w:<12}{cl:>9.4f}{BASELINE[w]:>9.4f}{cl - BASELINE[w]:>+9.4f}   "
              f"{cc:>9.4f}{CASCADE0[w]:>9.4f}{cc - CASCADE0[w]:>+9.4f}   "
              f"{gap:>+8.4f}{gap0:>+10.4f}")
    wins = sum(1 for v in out.values() if v["lpb"] >= v["cascade"] - 0.003)
    print(f"\n  LPB 최상위/동률 세계: {wins}/6"
          f"   (기준 구성에서는 6/6)")
    json.dump(out, open(OUT / f"{tag}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return out


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "b6"
    if stage == "b6":
        run("b6", votes=True, bankruptcy="force_cheapest")
    else:
        run("q2no", votes=False, bankruptcy="zero_quality")
