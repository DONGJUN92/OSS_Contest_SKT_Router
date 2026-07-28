"""3c-stress2: 약한 검증기(noise 0.35) + 약한 인코더(n=150)에서 갱신 가치 최종 판정."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, copy, yaml
from src.synth import make_world
from src.irt.mml import fit_mml
from src.harness import run_tier, tier_budgets
from src.engine.pandora import NoiseModel, tune_lambda_replay
from phase2_stages import IRTEncoder
from run_phase3 import make_pandora

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG2 = copy.deepcopy(yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8")))
CFG2["synth"]["verifier_noise"] = 0.35

print("[3c-stress2] weak verifier(0.35) + weak encoder(n=150) / World A / fast+balanced")
ds = make_world(CFG2, "irt", seed=CFG2["seed"])
folds = ds.stratified_folds(CFG2["eval"]["k_folds"], CFG2["seed"])
rng = np.random.default_rng(7)
for tier in ["fast", "balanced"]:
    rows = {"pandora-irt": [], "pandora-noupd": []}
    calls = {k: [] for k in rows}
    for f, test_idx in enumerate(folds):
        tr = np.setdiff1d(np.arange(ds.n), test_idx)
        tr_enc = rng.choice(tr, size=150, replace=False)
        irt = fit_mml(ds.quality[tr], ds.domains[tr], 4, per_domain_b=True)
        enc = IRTEncoder(irt["a"], irt["b"]).fit(
            ds.features[tr_enc], ds.domains[tr_enc], ds.quality[tr_enc])
        noise = NoiseModel().fit(ds.verifier[tr].ravel(), ds.quality[tr].ravel())
        sub = rng.choice(tr, size=800, replace=False)
        b_sub = tier_budgets(ds, sub, CFG2)[tier]
        b_te = tier_budgets(ds, test_idx, CFG2)[tier]
        for name, upd in [("pandora-irt", True), ("pandora-noupd", False)]:
            fac = lambda l, u=upd: make_pandora(irt, enc, noise, l, do_update=u)
            lam = tune_lambda_replay(fac, ds, sub, b_sub)
            r = run_tier(ds, test_idx, fac(lam), tier, b_te)
            rows[name].append(r.mean_quality)
            calls[name].append(r.calls_per_query)
    for name in rows:
        q = np.array(rows[name])
        print(f"  {tier:<9} {name:<15} quality={q.mean():.4f}+-{q.std():.4f}"
              f"  calls/q={np.mean(calls[name]):.2f}")
