"""3c-stress: 약한 프로파일러(인코더 train=150건) 조건에서 베이지안 갱신 가치 재검증.

가설: 본 실험(3c)에서 갱신 이득이 없던 이유는 인코더가 θ를 거의 정확히 예측해
사전분포가 좁았기 때문. 실데이터처럼 프로파일러가 약하면 관측 기반 갱신이
가치를 가져야 한다. (IRT 곡선(a,b)은 전체 train으로 적합 — 모델 이해는 유지,
쿼리별 사전분포만 약화)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from src.irt.mml import fit_mml
from src.harness import run_tier, tier_budgets
from src.engine.pandora import NoiseModel, tune_lambda_replay
from phase2_stages import IRTEncoder, DiscLR
from run_phase2 import load_worlds_and_folds, CFG
from run_phase3 import make_pandora, make_pandora_disc

N_DOM = CFG["synth"]["n_domains"]
worlds, folds = load_worlds_and_folds()
rng = np.random.default_rng(7)
print("[3c-stress] weak profiler (encoder train=150) / fast tier")
for wname in ["irt", "specialist"]:
    ds = worlds[wname]
    rows = {"pandora-irt": [], "pandora-noupd": [], "pandora-disc": []}
    calls = {k: [] for k in rows}
    for f, test_idx in enumerate(folds[wname]):
        tr = np.setdiff1d(np.arange(ds.n), test_idx)
        tr_enc = rng.choice(tr, size=150, replace=False)
        irt = fit_mml(ds.quality[tr], ds.domains[tr], N_DOM, per_domain_b=True)
        enc = IRTEncoder(irt["a"], irt["b"]).fit(
            ds.features[tr_enc], ds.domains[tr_enc], ds.quality[tr_enc])
        disc = DiscLR().fit(ds.features[tr_enc], ds.domains[tr_enc], ds.quality[tr_enc])
        noise = NoiseModel().fit(ds.verifier[tr].ravel(), ds.quality[tr].ravel())
        sub = rng.choice(tr, size=800, replace=False)
        b_sub = tier_budgets(ds, sub, CFG)["fast"]
        b_te = tier_budgets(ds, test_idx, CFG)["fast"]
        variants = {
            "pandora-irt": lambda l: make_pandora(irt, enc, noise, l),
            "pandora-noupd": lambda l: make_pandora(irt, enc, noise, l, do_update=False),
            "pandora-disc": lambda l: make_pandora_disc(disc, noise, l),
        }
        for name, fac in variants.items():
            lam = tune_lambda_replay(fac, ds, sub, b_sub)
            r = run_tier(ds, test_idx, fac(lam), "fast", b_te)
            rows[name].append(r.mean_quality)
            calls[name].append(r.calls_per_query)
    print(f"  World={wname}")
    for name in rows:
        q = np.array(rows[name])
        print(f"    {name:<15} quality={q.mean():.4f}+-{q.std():.4f}"
              f"  calls/q={np.mean(calls[name]):.2f}")
