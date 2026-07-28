"""정밀 프로브: World A fast의 조기정지 후회(4.65%)가 시퀀스 어디서 발생하는가.

가설 H1: 후회가 꼬리(생존 모드 구간)에 집중 → 페이싱 개선(더 이른 절약)으로 재분배 가능
가설 H2: 후회가 균등 분포 → shadow price 배분의 구조적 잔차 (개선 불가에 가까움)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from src.harness import Session, tier_budgets
from src.cost_mirror import cost_matrix
from src.router import LPBRouter
from run_phase2 import CFG
from run_phase4 import ext_worlds

worlds, ext, folds = ext_worlds()
eds = ext["irt"]
cmat = cost_matrix(eds)
quarter_events = np.zeros(4)
quarter_counts = np.zeros(4)
survival_events, survival_queries, total_events = 0, 0, 0

for f in range(CFG["eval"]["k_folds"]):
    test_idx = folds["irt"][f]
    tr = np.setdiff1d(np.arange(eds.n), test_idx)
    router = LPBRouter(CFG, CFG["synth"]["n_domains"], seed=f).fit(eds, tr, "fast")
    budget = tier_budgets(eds, test_idx, CFG)["fast"]
    pol = router.policy()
    pol.reset(tier="fast", budget=budget, n_queries=len(test_idx))
    spent_total = 0.0
    lam_cap = pol.lam0 * pol.clip
    for t, i in enumerate(test_idx):
        remaining = budget - spent_total
        in_survival = pol.inner.lam >= lam_cap * 0.999
        sess = Session(costs=cmat[i], verifier_row=eds.verifier[i],
                       remaining_budget=remaining)
        choice = pol.route(sess, eds.features[i], int(eds.domains[i]))
        quarter = min(int(4 * t / len(test_idx)), 3)
        quarter_counts[quarter] += 1
        survival_queries += in_survival
        if sess.called:
            if choice not in sess.called:
                choice = max(sess.called, key=lambda m: sess.verifier_row[m])
            if eds.quality[i, choice] == 0:
                left = remaining - sess.spent
                unopened = [m for m in range(eds.m) if m not in sess.called]
                if any(eds.quality[i, m] == 1 and cmat[i, m] <= left for m in unopened):
                    quarter_events[quarter] += 1
                    total_events += 1
                    survival_events += in_survival
        spent_total += sess.spent
        pol.observe_spend(sess.spent)

print("[probe] World A / fast — 조기정지 후회의 시퀀스 내 분포 (5-fold 합산)")
for q in range(4):
    rate = 100 * quarter_events[q] / max(quarter_counts[q], 1)
    print(f"  {q+1}사분위: 후회율 {rate:.2f}%  (이벤트 {int(quarter_events[q])}건)")
print(f"  생존 모드 중 발생 비율: {survival_events}/{total_events} "
      f"(생존 모드 쿼리 비중 {100*survival_queries/quarter_counts.sum():.1f}%)")
