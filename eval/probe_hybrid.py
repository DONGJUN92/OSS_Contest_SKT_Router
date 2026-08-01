"""Attempt 3 (Phase 15) — 하이브리드 정책 선택으로 한계 극복 시도.

배경: belief-swap(중립)·value-order(D17에 포섭)가 실패. 3모델·실텍스트에서 LPB와 학습형이
대등하고 어느 쪽이 나은지 세계마다 다르다. → **train 내 held-out으로 LPB vs learned를 선택**
하는 메타 라우터(둘 다 저용량이라 선택이 안정적일 것). 가설: 하이브리드가 세계마다 max(LPB,
learned)에 근접해 어느 baseline에도 지지 않는다.

프로토콜(테스트 미접촉): tr을 fit(70%)/sel(30%)로 분할 → 각 정책을 tr_fit 적합 → tr_sel 품질로
선택 → 선택된 정책을 te 평가. LPB는 open_order=auto(σ/value 중 나은 것).

사용법: python eval/probe_hybrid.py [--folds N] [--worlds textworld,irt,specialist]
"""
import sys, pathlib, json, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from src.synth import make_world
from src.synth2 import make_world2
from src.synth_ext import extend_with_votes
from src.textworld import build_textworld, to_dataset
from src.text_encoder import get_encoder
from src.verifier import feature_matrix, fold_verifier_matrix
from src.harness import run_tier, tier_budgets
from src.router import LPBRouter
from baselines.policies import LearnedRouter, tune_learned_lambda
from phase2_stages import DiscLR

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config_ax3.yaml", encoding="utf-8"))
OUT = ROOT / "eval" / "results"
TIERS = ["fast", "balanced", "premium"]
W = {t: CFG["tiers"][t]["weight"] for t in CFG["tiers"]}
SYNTH = ["irt", "specialist", "corr", "crossing", "nosignal"]
# 시도3b: 선택 margin (D21 교훈 — 잡음을 승리로 오인 방지). LPB를 기본으로, learned가 tr_sel
# 에서 이만큼 이상 명확히 이길 때만 전환. thesis 정합(LPB 주 정책, learned는 명확한 대등우위 보험).
MARGIN = 0.01


def _build(wname):
    if wname == "textworld":
        tw = build_textworld(CFG, seed=42)
        ds, meta = to_dataset(tw, CFG)
        _enc = get_encoder("hashing")
        ds.features = _enc.encode(meta["prompts"])
        ds.text_encoder = _enc          # D70: 배포 텍스트 경로 계약
        F = feature_matrix(meta)
        return ds, F, ds.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])
    base = make_world(CFG, wname, seed=CFG["seed"]) if wname in ("irt", "specialist") \
        else make_world2(CFG, wname, seed=CFG["seed"])
    conc = 0.6 if wname == "crossing" else None
    eds = extend_with_votes(base, CFG, conc=conc)
    return eds, None, base.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])


def _fit_lpb(ds, tr, tier, seed):
    return LPBRouter(CFG, len(np.unique(ds.domains)), seed=seed,
                     use_domain=False, open_order="auto").fit(ds, tr, tier)


def _fit_learned(ds, tr, tier):
    disc = DiscLR().fit(ds.features[tr], ds.domains[tr], ds.quality[tr])
    lam = tune_learned_lambda(disc, ds, tr, tier_budgets(ds, tr, CFG)[tier])
    return LearnedRouter(disc, lam)


def _run(wname, nf):
    ds, F, folds = _build(wname)
    res = {t: {"lpb": [], "learned": [], "hybrid": []} for t in TIERS}
    picks = {"lpb": 0, "learned": 0}
    for f in range(nf):
        te = folds[f]
        tr = np.setdiff1d(np.arange(ds.n), te)
        if F is not None:
            V, _ = fold_verifier_matrix(F, ds.quality, tr)
            ds.verifier = V
        rng = np.random.default_rng(1000 + f)
        perm = rng.permutation(tr)
        n_fit = int(0.7 * len(perm))
        tr_fit, tr_sel = perm[:n_fit], perm[n_fit:]
        for tier in TIERS:
            b_te = tier_budgets(ds, te, CFG)[tier]
            b_sel = tier_budgets(ds, tr_sel, CFG)[tier]
            lpb = _fit_lpb(ds, tr_fit, tier, f)
            lrn = _fit_learned(ds, tr_fit, tier)
            q_lpb_sel = run_tier(ds, tr_sel, lpb.policy(), tier, b_sel).mean_quality
            q_lrn_sel = run_tier(ds, tr_sel, lrn, tier, b_sel).mean_quality
            pick_lrn = q_lrn_sel > q_lpb_sel + MARGIN     # learned가 margin 이상 명확히 이길 때만
            chosen = lrn if pick_lrn else lpb.policy()
            picks["learned" if pick_lrn else "lpb"] += 1
            res[tier]["lpb"].append(run_tier(ds, te, lpb.policy(), tier, b_te).mean_quality)
            res[tier]["learned"].append(run_tier(ds, te, lrn, tier, b_te).mean_quality)
            res[tier]["hybrid"].append(run_tier(ds, te, chosen, tier, b_te).mean_quality)
    comp = {p: sum(W[t] * float(np.mean(res[t][p])) for t in TIERS)
            for p in ("lpb", "learned", "hybrid")}
    return comp, picks


def main(nf=5, which=None):
    which = which or ["textworld", "irt", "specialist"]
    t0 = time.time()
    print(f"\n[probe_hybrid] 3모델 — 하이브리드(LPB vs learned 선택) (nf={nf})")
    print(f"  {'world':<12}{'LPB':<9}{'learned':<9}{'hybrid':<9}{'hyb−max':<10}{'picks(lpb/lrn)':<14}")
    summary = {}
    for wname in which:
        comp, picks = _run(wname, nf)
        best_base = max(comp["lpb"], comp["learned"])
        summary[wname] = {**{k: round(v, 4) for k, v in comp.items()},
                          "hybrid_minus_max": round(comp["hybrid"] - best_base, 4),
                          "picks": picks}
        print(f"  {wname:<12}{comp['lpb']:<9.4f}{comp['learned']:<9.4f}{comp['hybrid']:<9.4f}"
              f"{comp['hybrid'] - best_base:<+10.4f}{picks['lpb']}/{picks['learned']}")
    print(f"  ({time.time() - t0:.0f}s)")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"n_folds": nf, "summary": summary,
               "note": "hybrid≈max(LPB,learned)이면 어느 baseline에도 안 짐 = 한계 극복"},
              open(OUT / "probe_hybrid.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("  → eval/results/probe_hybrid.json")
    return summary


if __name__ == "__main__":
    nf = 5
    which = None
    if "--folds" in sys.argv:
        nf = int(sys.argv[sys.argv.index("--folds") + 1])
    if "--worlds" in sys.argv:
        which = sys.argv[sys.argv.index("--worlds") + 1].split(",")
    main(nf, which)
