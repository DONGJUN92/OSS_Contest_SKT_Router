"""반사실 검증: textworld에서 베이지안 사후 선택의 가치 (Phase 6 기각 건의 재시험).

Phase 6에서 기각된 이유 = 합성 검증기가 완벽해 선택오류가 0이었기 때문.
실텍스트 검증기(AUC 0.90)에서 선택오류 8.3%가 등장 — 기각 조건이 소멸했는지 확인.
diagnose()가 이미 계산하는 quality_bayes_cf(사전 p̄ × 우도 결합 선택)를 출력한다.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from src.harness import tier_budgets
from src.verifier import fold_verifier_matrix
from src.router import LPBRouter
from run_phase2 import CFG
from run_diagnostics import diagnose
from run_phase8 import build_f

tw, ds, meta, F, folds = build_f()
for tier in ["fast", "balanced", "premium"]:
    qa, qb = [], []
    for f, te in enumerate(folds):
        tr = np.setdiff1d(np.arange(ds.n), te)
        V, _ = fold_verifier_matrix(F, ds.quality, tr)
        ds.verifier = V
        router = LPBRouter(CFG, len(np.unique(ds.domains)), seed=f).fit(ds, tr, tier)
        d = diagnose(ds, te, router, tier, tier_budgets(ds, te, CFG)[tier])
        qa.append(d["quality"])
        qb.append(d["quality_bayes_cf"])
    print(f"  {tier:<10} actual(v-최대)={np.mean(qa):.4f}  "
          f"bayes-cf(사전×우도)={np.mean(qb):.4f}  delta={np.mean(qb)-np.mean(qa):+.4f}")
