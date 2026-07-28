"""B4 검증: λ₀ 튜너의 브래킷 웜스타트가 해를 바꾸지 않는가?

연속(Beta) 경로에서 λ₀ 튜너는 보정셋을 20여 회 리플레이하고, **λ가 작을수록 정책이
상자를 많이 열어** 리플레이가 비싸진다. 기본 브래킷은 λ=0.5에서 올라오므로 가장 비싼
구간을 먼저 지난다 (측정: 튜닝 1,800s vs 모형 적합 45s).

LPBRouter는 값싼 평균정합 Bernoulli 튜닝으로 **브래킷 시작점만** 잡고, 탐색과 해는
정확한 Beta 정책으로 구한다. 탐색 순서만 바뀌므로 λ₀는 동일해야 한다 — 그것을 확인한다.

기각된 대안 (정직 기록): 튜닝 **전체**를 Bernoulli로 대체하는 fast_tune 안은
27배 빨랐지만 λ₀ 9.160→12.477, 품질 0.8051→0.7995 로 실제 손실이 나 폐기했다.
"빠르다"가 채택 근거가 될 수 없음을 보여준 사례 (World F-c / fast / fold 0).

사용법: python eval/probe_tune_family.py [--folds 1] [--tiers fast,balanced,premium]
"""
import sys, pathlib, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from src.engine.pandora import tune_lambda_quality
from src.harness import run_tier, tier_budgets
from src.router import LPBRouter
from src.verifier import fold_verifier_matrix
from run_b4 import build_fc, CFG


def main(n_folds=1, tiers=("fast",)):
    ds, meta, F, folds = build_fc(scale="strict")
    print("[probe] λ₀ 브래킷 웜스타트의 해 동일성 (World F-c, 연속 품질)\n")
    print(f"  {'tier':<10}{'fold':<6}{'λ₀ 웜':>10}{'λ₀ 냉':>10}{'품질 웜':>10}"
          f"{'품질 냉':>10}{'Δ품질':>9}{'웜 적합':>9}{'냉 재탐색':>11}")
    for tier in tiers:
        for f in range(n_folds):
            te = folds[f]
            tr = np.setdiff1d(np.arange(ds.n), te)
            ds.verifier, _ = fold_verifier_matrix(F, ds.quality, tr)
            b_te = tier_budgets(ds, te, CFG)[tier]

            t0 = time.perf_counter()
            r = LPBRouter(CFG, CFG["synth"]["n_domains"], seed=f,
                          prize_family="beta").fit(ds, tr, tier)
            dt_warm = time.perf_counter() - t0
            lam_warm = r.lam0
            q_warm = run_tier(ds, te, r.policy(), tier, b_te).mean_quality

            t1 = time.perf_counter()               # 동일 적합물에 λ₀만 힌트 없이 재탐색
            lam_cold = tune_lambda_quality(r._factory, ds, r._tr_cal, r._b_tune)
            dt_cold = time.perf_counter() - t1
            r.lam0 = lam_cold
            q_cold = run_tier(ds, te, r.policy(), tier, b_te).mean_quality

            print(f"  {tier:<10}{f:<6}{lam_warm:>10.4f}{lam_cold:>10.4f}{q_warm:>10.4f}"
                  f"{q_cold:>10.4f}{q_warm - q_cold:>+9.4f}{dt_warm:>8.0f}s"
                  f"{dt_cold:>10.0f}s")
            assert abs(lam_warm - lam_cold) < 1e-9, "웜스타트가 해를 바꿈 — 탐색 로직 결함"
    print("\n  판정: λ₀ 일치 (탐색 순서 최적화가 해를 바꾸지 않음)")


if __name__ == "__main__":
    nf = int(sys.argv[sys.argv.index("--folds") + 1]) if "--folds" in sys.argv else 1
    ts = (sys.argv[sys.argv.index("--tiers") + 1].split(",")
          if "--tiers" in sys.argv else ("fast",))
    main(nf, ts)
