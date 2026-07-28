"""B9 — 잔여 하이퍼파라미터 탐색 + fold 분산 페널티 (PROJECT_PLAN §9 P2).

**이 실험의 가설은 "튜닝하면 좋아진다"가 아니라 "현재 기본값이 이미 충분하다"** 이다.
v5 설계는 탐색 표면적 축소(Optuna 변수 5+→2)를 분포 이동 방어의 근거로 삼았으므로,
탐색이 기본값을 유의미하게 이기면 그 근거가 약해지고, 이기지 못하면 강해진다.
어느 쪽이든 결과를 그대로 기록한다.

탐색 대상 (수동 기본값이 있는 것만)
    cal_frac       0.30   보정셋 비율 — λ 튜닝·인증서의 표본 배분
    tune_headroom  0.97   λ 튜닝 목표의 안전 여유 (Phase 7 R1)
    k_over         8.0    페이싱 과지출 게인 (Phase 4에서 4~16 둔감 확인)
    k_under        2.0    페이싱 절약 게인

목적함수: mean(fold 종합점수) − penalty × std(fold 종합점수)
  성공 기준 ③(fold 간 분산이 낮을 것, PROJECT_PLAN §1.2)의 직접 구현.

사용법: python run_b9.py [--world irt] [--trials 24] [--penalty 1.0] [--backend optuna]
"""
import sys, pathlib, json

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

from src.engine.pacing import PacedPandora
from src.harness import run_tier, tier_budgets
from src.router import LPBRouter
from src.synth import make_world
from src.synth2 import make_world2
from src.synth_ext import extend_with_votes
from src.tune import SearchSpace, save, search
from run_phase2 import CFG, OUT

TIERS = ["fast", "balanced", "premium"]
W = {t: CFG["tiers"][t]["weight"] for t in CFG["tiers"]}
BASELINE = {"cal_frac": 0.30, "tune_headroom": 0.97, "k_over": 8.0, "k_under": 2.0}


def build(wname):
    base = make_world(CFG, wname, seed=CFG["seed"]) if wname in ("irt", "specialist") \
        else make_world2(CFG, wname, seed=CFG["seed"])
    conc = 0.6 if wname == "crossing" else None
    return extend_with_votes(base, CFG, conc=conc), \
        base.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])


def make_objective(ds, folds):
    """config → fold별 **종합점수** 리스트 (tier 가중 평균)."""

    def objective(cfg):
        per_fold = []
        for f in range(CFG["eval"]["k_folds"]):
            te = folds[f]
            tr = np.setdiff1d(np.arange(ds.n), te)
            tot = 0.0
            for tier in TIERS:
                r = LPBRouter(CFG, CFG["synth"]["n_domains"], seed=f,
                              cal_frac=float(cfg["cal_frac"]),
                              tune_headroom=float(cfg["tune_headroom"])).fit(ds, tr, tier)
                pol = PacedPandora(r._factory, r.lam0, k_over=float(cfg["k_over"]),
                                   k_under=float(cfg["k_under"]), name="lpb-tuned")
                b_te = tier_budgets(ds, te, CFG)[tier]
                tot += W[tier] * run_tier(ds, te, pol, tier, b_te).mean_quality
            per_fold.append(tot)
        return per_fold

    return objective


def main(world="irt", n_trials=18, penalty=1.0, backend="builtin"):
    ds, folds = build(world)
    space = SearchSpace(cont={"cal_frac": (0.15, 0.45), "tune_headroom": (0.90, 1.00),
                              "k_over": (2.0, 24.0), "k_under": (0.5, 8.0)})
    print(f"[b9] 하이퍼 탐색 — world={world} trials={n_trials} penalty={penalty} "
          f"backend={backend}")
    print(f"  목적 = mean(fold 종합) − {penalty} × std(fold 종합)\n")
    rep = search(make_objective(ds, folds), space, n_trials=n_trials, penalty=penalty,
                 seed=CFG["seed"], backend=backend, baseline=BASELINE)

    b = rep["best"]
    scores = [t["score"] for t in rep["trials"]]
    print(f"\n  기본값 점수 : {rep['baseline_score']:.4f}")
    print(f"  탐색 최적   : {max(scores):.4f}  (채택 문턱 = 기본값 표준오차 "
          f"{rep['margin']:.4f})")
    print(f"  전 시행 범위: {min(scores):.4f} ~ {max(scores):.4f} "
          f"(폭 {max(scores) - min(scores):.4f})")
    if rep["keep_baseline"]:
        gain = max(scores) - rep["baseline_score"]
        print(f"  → **기본값 유지**: 최대 이득 +{gain:.4f} 가 fold 잡음(표준오차 "
              f"{rep['margin']:.4f}) 이하다.")
        print(f"     방법이 이 하이퍼들에 둔감하다는 뜻이며, v5의 '탐색 표면적 축소'"
              f" 근거가 강화된다.")
    else:
        gain = b["score"] - rep["baseline_score"]
        print(f"  → 탐색 최적이 +{gain:.4f} 우세 (문턱 초과): {b['config']}")
        print(f"     (실데이터에서 재튜닝 가치가 있음을 시사 — 단 in-dist 결과임에 유의)")
    rep["world"] = world
    save(rep, OUT / f"b9-{world}.json")


if __name__ == "__main__":
    a = sys.argv
    main(world=a[a.index("--world") + 1] if "--world" in a else "irt",
         n_trials=int(a[a.index("--trials") + 1]) if "--trials" in a else 18,
         penalty=float(a[a.index("--penalty") + 1]) if "--penalty" in a else 1.0,
         backend=a[a.index("--backend") + 1] if "--backend" in a else "builtin")
