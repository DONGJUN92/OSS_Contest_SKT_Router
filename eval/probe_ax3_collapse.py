"""5→3 collapse 측정 (권고 #1) — A.X 3모델에서 LPB vs 튜닝 cascade vs 학습형 라우터.

동기(적대적 평가 지적): Pandora/Weitzman 선택탐색은 상자가 많을 때 제값을 한다. 3모델
사다리(또는 3×N 투표=9상자)에서는 2-threshold cascade나 학습형 단일호출 라우터가 대부분의
가치를 이미 잡을 수 있다 — easy세계 동률·IRT 분포내 ~0 ablation이 조기경고다. 정책 형태를
방어하지 말고 **측정**해서, 3상자 체제에서 LPB 구조가 여전히 우위인지/무의미해지는지 확정.

세 정책 (동일 3모델 세계·동일 fold):
  · LPB       : LPBRouter(use_domain=False, 배포 단일집단) — 순차 개봉·정지
  · cascade   : 비용오름 StaticCascade, τ를 train에서 그리드 튜닝 (FrugalGPT류)
  · learned   : DiscLR 예측 + argmax(p̄−λc) 단일호출 (RouteLLM류, baselines.LearnedRouter)

사용법: python eval/probe_ax3_collapse.py [--folds N] [--worlds irt,...,textworld]
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
from src.cost_mirror import cost_matrix
from src.router import LPBRouter
from baselines.policies import (StaticCascade, tune_cascade_tau,
                                LearnedRouter, tune_learned_lambda)
from phase2_stages import DiscLR

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config_ax3.yaml", encoding="utf-8"))
OUT = ROOT / "eval" / "results"
TIERS = ["fast", "balanced", "premium"]
W = {t: CFG["tiers"][t]["weight"] for t in CFG["tiers"]}
SYNTH = ["irt", "specialist", "corr", "crossing", "nosignal"]
NS = (1, 3, 5)          # 투표 상자 크기. --single이면 (1,) = Q2=No 보수적 배포 경로 (#4a)


def _three_policies(ds, tr, te, tier, order, seed):
    """정책들을 train-적합·평가 → (lpb, lpb_auto, cascade, learned)."""
    b_te = tier_budgets(ds, te, CFG)[tier]
    b_tr = tier_budgets(ds, tr, CFG)[tier]
    n_dom = len(np.unique(ds.domains))
    # LPB (배포 단일집단, σ-order 기본)
    router = LPBRouter(CFG, n_dom, seed=seed, use_domain=False).fit(ds, tr, tier)
    lpb = run_tier(ds, te, router.policy(), tier, b_te).mean_quality
    # LPB-auto (개선: open_order=auto, train-cal로 σ/value 선택) — Phase 15
    router_a = LPBRouter(CFG, n_dom, seed=seed, use_domain=False,
                         open_order="auto").fit(ds, tr, tier)
    lpb_auto = run_tier(ds, te, router_a.policy(), tier, b_te).mean_quality
    # tuned cascade
    tau = tune_cascade_tau(ds, tr, CFG, tier, b_tr)
    cas = run_tier(ds, te, StaticCascade(order, tau), tier, b_te).mean_quality
    # learned single-shot (DiscLR)
    disc = DiscLR().fit(ds.features[tr], ds.domains[tr], ds.quality[tr])
    lam = tune_learned_lambda(disc, ds, tr, b_tr)
    lrn = run_tier(ds, te, LearnedRouter(disc, lam), tier, b_te).mean_quality
    return lpb, lpb_auto, cas, lrn


def _run_synth(wname, nf):
    base = make_world(CFG, wname, seed=CFG["seed"]) if wname in ("irt", "specialist") \
        else make_world2(CFG, wname, seed=CFG["seed"])
    conc = 0.6 if wname == "crossing" else None
    eds = extend_with_votes(base, CFG, Ns=NS, conc=conc)
    folds = base.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])
    order = list(np.argsort(cost_matrix(eds).mean(axis=0)))
    out = {t: {"lpb": [], "lpb-auto": [], "cascade": [], "learned": []} for t in TIERS}
    for tier in TIERS:
        for f in range(nf):
            te = folds[f]
            tr = np.setdiff1d(np.arange(eds.n), te)
            lpb, lpb_a, cas, lrn = _three_policies(eds, tr, te, tier, order, f)
            out[tier]["lpb"].append(lpb)
            out[tier]["lpb-auto"].append(lpb_a)
            out[tier]["cascade"].append(cas)
            out[tier]["learned"].append(lrn)
    return out


def _run_textworld(nf):
    tw = build_textworld(CFG, seed=42)
    ds, meta = to_dataset(tw, CFG, Ns=NS)
    ds.features = get_encoder("hashing").encode(meta["prompts"])
    F = feature_matrix(meta)
    folds = ds.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])
    order = list(np.argsort(cost_matrix(ds).mean(axis=0)))
    out = {t: {"lpb": [], "lpb-auto": [], "cascade": [], "learned": []} for t in TIERS}
    for tier in TIERS:
        for f in range(nf):
            te = folds[f]
            tr = np.setdiff1d(np.arange(ds.n), te)
            V, _ = fold_verifier_matrix(F, ds.quality, tr)
            ds.verifier = V
            lpb, lpb_a, cas, lrn = _three_policies(ds, tr, te, tier, order, f)
            out[tier]["lpb"].append(lpb)
            out[tier]["lpb-auto"].append(lpb_a)
            out[tier]["cascade"].append(cas)
            out[tier]["learned"].append(lrn)
    return out


def _composite(world_res):
    return {pol: sum(W[t] * float(np.mean(world_res[t][pol])) for t in TIERS)
            for pol in ("lpb", "lpb-auto", "cascade", "learned")}


def main(nf=5, which=None):
    which = which or (SYNTH + ["textworld"])
    t0 = time.time()
    results = {}
    for wname in which:
        results[wname] = _run_textworld(nf) if wname == "textworld" else _run_synth(wname, nf)

    print(f"\n[probe_ax3_collapse] A.X 3모델 — LPB(σ) vs LPB(auto) vs cascade vs learned "
          f"(nf={nf}, {time.time() - t0:.0f}s)")
    print(f"  {'world':<12}{'LPB':<9}{'LPB-auto':<10}{'cascade':<9}{'learned':<9}"
          f"{'auto−σ':<10}{'auto−learn':<11}")
    summary = {}
    for wname in which:
        c = _composite(results[wname])
        summary[wname] = {k: round(v, 4) for k, v in c.items()}
        summary[wname]["auto_minus_sigma"] = round(c["lpb-auto"] - c["lpb"], 4)
        summary[wname]["auto_minus_learned"] = round(c["lpb-auto"] - c["learned"], 4)
        summary[wname]["auto_minus_cascade"] = round(c["lpb-auto"] - c["cascade"], 4)
        print(f"  {wname:<12}{c['lpb']:<9.4f}{c['lpb-auto']:<10.4f}{c['cascade']:<9.4f}"
              f"{c['learned']:<9.4f}{c['lpb-auto'] - c['lpb']:<+10.4f}"
              f"{c['lpb-auto'] - c['learned']:<+11.4f}")
    OUT.mkdir(parents=True, exist_ok=True)
    # ★ D54: 출력 파일명이 **NS(투표 상자 구성)에 의존해야 한다.** 초판은 `--single` 여부와
    # 무관하게 항상 `probe_ax3_collapse.json` 에 썼다. 그런데 `reproduce.py` 는 이 스크립트를
    # 두 번(기본 / `--single`) 돌리고 산출물을 `probe_ax3_collapse(_q2no).json` 이라 적어 뒀다 —
    # 즉 **두 단계가 같은 파일을 서로 덮어써서** 뒤에 돈 것만 남고, 저장소에 있는
    # `_q2no`·`_q2yes` 아티팩트는 **어떤 명령으로도 재생성되지 않았다.** D46 과 같은 부류다.
    suffix = "_q2no" if tuple(NS) == (1,) else ""
    name = f"probe_ax3_collapse{suffix}.json"
    json.dump({"config": "config_ax3.yaml (A.X 3모델)", "n_folds": nf, "summary": summary,
               "samples_per_cell": list(NS),
               "branch": "Q2=No (투표 상자 없음)" if suffix else "Q2=Yes (투표 상자 포함)",
               "note": "LPB=배포 단일집단, cascade=τ튜닝, learned=DiscLR 단일호출. "
                       "3상자에서 LPB 구조 우위 여부. 상세: docs/reflections/phase13.md"},
              open(OUT / name, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"  → eval/results/{name}")
    return summary


if __name__ == "__main__":
    nf = 5
    which = None
    if "--folds" in sys.argv:
        nf = int(sys.argv[sys.argv.index("--folds") + 1])
    if "--worlds" in sys.argv:
        which = sys.argv[sys.argv.index("--worlds") + 1].split(",")
    if "--single" in sys.argv:               # Q2=No 보수적 배포 경로 (투표 상자 없음)
        NS = (1,)
    main(nf, which)
