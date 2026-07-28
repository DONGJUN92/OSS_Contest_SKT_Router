"""프로브: corr/balanced에서 LPB의 무응답 2.7%와 cascade 대비 패배의 원인 규명.

수집: λ0 튜너의 후보별 (품질, 지출) 곡선 / 테스트 리플레이의 λ 궤적·지출 궤적 /
무응답 발생 위치 / 생존 모드 작동 여부 / calls per query
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from src.harness import Session, tier_budgets, run_tier
from src.cost_mirror import cost_matrix
from src.router import LPBRouter
from src.engine.pandora import tune_lambda_replay
from run_phase2 import CFG
from run_phase7 import build_world

base, eds, folds = build_world("corr")
f = 0
test_idx = folds[f]
tr = np.setdiff1d(np.arange(eds.n), test_idx)
router = LPBRouter(CFG, CFG["synth"]["n_domains"], seed=f).fit(eds, tr, "balanced")

# 1) λ 후보 곡선 재현 (튜너 내부와 동일 절차)
rng = np.random.default_rng(f)
perm = rng.permutation(tr)
tr_cal = perm[int(0.7 * len(perm)):]
b_cal = tier_budgets(eds, tr_cal, CFG)["balanced"]
lam_min = tune_lambda_replay(router._factory, eds, tr_cal, b_cal)
print(f"[probe corr/balanced fold0] lam_min={lam_min:.4f}  선택된 lam0={router.lam0:.4f}")
print(f"  {'λ':<10}{'cal품질':<10}{'cal지출%':<10}")
for mult in [1, 2, 4, 8, 16]:
    r = run_tier(eds, tr_cal, router._factory(lam_min * mult), "t", b_cal)
    print(f"  {lam_min * mult:<10.4f}{r.mean_quality:<10.4f}{100 * r.total_cost / b_cal:<10.1f}")

# 2) 테스트 리플레이 상세 궤적
b_te = tier_budgets(eds, test_idx, CFG)["balanced"]
pol = router.policy()
pol.reset(tier="balanced", budget=b_te, n_queries=len(test_idx))
cmat = cost_matrix(eds)
spent = 0.0
unans_pos, lam_hist, calls_hist = [], [], []
surv = 0
for t, i in enumerate(test_idx):
    remaining = b_te - spent
    sess = Session(costs=cmat[i], verifier_row=eds.verifier[i], remaining_budget=remaining)
    pol.route(sess, eds.features[i], int(eds.domains[i]))
    if not sess.called:
        unans_pos.append(t)
    lam_hist.append(pol.inner.lam)
    calls_hist.append(len(sess.called))
    if pol.inner.lam >= router.lam0 * pol.clip * 0.999:
        surv += 1
    spent += sess.spent
    pol.observe_spend(sess.spent)

lam_hist = np.array(lam_hist)
print(f"\n  테스트: 지출 {100 * spent / b_te:.1f}%  무응답 {len(unans_pos)}건 "
      f"위치={unans_pos[:12]}{'...' if len(unans_pos) > 12 else ''}")
print(f"  λ 궤적: λ0={router.lam0:.4f} min={lam_hist.min():.4f} "
      f"max={lam_hist.max():.4f} cap={router.lam0 * pol.clip:.4f} 생존모드쿼리={surv}")
print(f"  calls/q: 평균 {np.mean(calls_hist):.2f}  분포 "
      f"{[int((np.array(calls_hist) == k).sum()) for k in range(0, 6)]} (0..5회)")
# 3) 쿼리별 비용 산포 (c_min 추정의 위험도)
cheap_costs = cmat[test_idx].min(axis=1)
print(f"  쿼리별 최저 호출비용: min={cheap_costs.min():.4f} median={np.median(cheap_costs):.4f} "
      f"max={cheap_costs.max():.4f}  (생존모드 c_min 추정 = 전역 min)")