"""B7: 시드 강건성 — 세계 생성 시드 3종에서 정책 서열(LPB > cascade) 불변 확인.

세계: irt(신호 강) / corr(최난관 합성) / textworld(실텍스트).
합성 세계는 3-fold 축약(서열 확인 목적), textworld는 5-fold 전체.
사용법: python run_b7.py [prize_mode]
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
from src.harness import run_tier, tier_budgets
from src.cost_mirror import cost_matrix
from src.synth import make_world
from src.synth2 import make_world2
from src.synth_ext import extend_with_votes
from src.router import LPBRouter
from run_phase2 import CFG, OUT
from run_phase8 import stage_8b
from baselines.policies import StaticCascade, tune_cascade_tau

TIERS = ["fast", "balanced", "premium"]
W = {t: CFG["tiers"][t]["weight"] for t in CFG["tiers"]}
SEEDS = [42, 123, 777]
PRIZE_MODE = sys.argv[1] if len(sys.argv) > 1 else "calibrated"


def eval_synth(wname, seed, k=3):
    base = (make_world(CFG, wname, seed=seed) if wname in ("irt", "specialist")
            else make_world2(CFG, wname, seed=seed))
    eds = extend_with_votes(base, CFG)
    folds = base.stratified_folds(k, CFG["seed"])
    order = list(np.argsort(cost_matrix(eds).mean(axis=0)))
    comb_l, comb_c = 0.0, 0.0
    for tier in TIERS:
        lq, cq = [], []
        for f in range(k):
            te = folds[f]
            tr = np.setdiff1d(np.arange(eds.n), te)
            router = LPBRouter(CFG, CFG["synth"]["n_domains"], seed=f,
                               prize_mode=PRIZE_MODE).fit(eds, tr, tier)
            b_te = tier_budgets(eds, te, CFG)[tier]
            lq.append(run_tier(eds, te, router.policy(), tier, b_te).mean_quality)
            tau = tune_cascade_tau(eds, tr, CFG, tier, tier_budgets(eds, tr, CFG)[tier])
            cq.append(run_tier(eds, te, StaticCascade(order, tau), tier, b_te).mean_quality)
        comb_l += W[tier] * float(np.mean(lq))
        comb_c += W[tier] * float(np.mean(cq))
    return comb_l, comb_c


if __name__ == "__main__":
    report = {}
    print(f"[B7] 시드 강건성 (prize_mode={PRIZE_MODE})")
    print(f"  {'world':<12}{'seed':<7}{'LPB':<9}{'cascade':<10}{'서열유지':<8}")
    for wname in ["irt", "corr", "textworld"]:
        for seed in SEEDS:
            if wname == "textworld":
                l, c = stage_8b(seed=seed, tag=f"b7-tw-{seed}", quiet=True,
                                prize_mode=PRIZE_MODE)
            else:
                l, c = eval_synth(wname, seed)
            ok = "O" if l >= c - 0.01 else "X"          # 동률 허용 오차 1%p
            report[f"{wname}/{seed}"] = {"lpb": round(l, 4), "cascade": round(c, 4)}
            print(f"  {wname:<12}{seed:<7}{l:<9.4f}{c:<10.4f}{ok:<8}")
    json.dump(report, open(OUT / "b7_seeds.json", "w", encoding="utf-8"), indent=1)
