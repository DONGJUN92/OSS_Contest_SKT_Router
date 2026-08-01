"""RouterBench 예행연 (Phase 15) — 실데이터 진입점(`src.loader`)으로 RouterBench-형 데이터를
적재해 LPB vs cascade vs learned를 비교. 샌드박스를 실제 문제 형태의 예행연습으로 전환.

이 프로브가 실증하는 것 = "수령 당일 파이프라인": 외부 데이터(long JSONL) → `load_dataset`
→ `validate_dataset`(Q2/Q4 자동판정) → 프롬프트 인코딩 → LPBRouter/cascade/learned 적합·비교.
합성 세계 생성기(synth*)를 우회하고 **로더 경로**만 탄다 — 실 RouterBench 파일도 같은 경로.

주의: 데이터는 RouterBench 구조·현실 프로파일 모사(make_routerbench_style.py)이지 실물이
아니다. 실물은 1.47GB parquet+pyarrow 필요. 목적은 절대 성능이 아니라 **파이프라인 예행 +
정책 간 상대 비교가 실데이터 형태에서 작동함**을 보이는 것.

사용법: python eval/probe_routerbench_rehearsal.py [--folds N]
"""
import sys, pathlib, json, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from src.loader import load_dataset, validate_dataset, format_report
from src.text_encoder import get_encoder
from src.harness import run_tier, tier_budgets
from src.cost_mirror import cost_matrix
from src.router import LPBRouter
from baselines.policies import (StaticCascade, tune_cascade_tau,
                                LearnedRouter, tune_learned_lambda)
from phase2_stages import DiscLR

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config_routerbench.yaml", encoding="utf-8"))
DATA = ROOT / "eval" / "results" / "routerbench_style.jsonl"
OUT = ROOT / "eval" / "results"
TIERS = ["fast", "balanced", "premium"]
W = {t: CFG["tiers"][t]["weight"] for t in CFG["tiers"]}


def main(nf=5):
    t0 = time.time()
    # 1) 수령 당일 파이프라인: 로더 → 검증(분기 자동판정) → 인코딩
    ds, meta = load_dataset(DATA, CFG)
    rep = validate_dataset(ds, meta, CFG)
    print(format_report(rep))
    _enc = get_encoder("hashing")
    ds.features = _enc.encode(meta["prompts"])   # 프롬프트 → 특성
    ds.text_encoder = _enc          # D70: 배포 텍스트 경로 계약
    folds = ds.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])
    order = list(np.argsort(cost_matrix(ds).mean(axis=0)))
    n_dom = len(np.unique(ds.domains))

    # 2) 정책 비교 (LPB 배포 단일집단 vs 튜닝 cascade vs 학습형 단일호출)
    res = {t: {"lpb": [], "cascade": [], "learned": []} for t in TIERS}
    for tier in TIERS:
        for f in range(nf):
            te = folds[f]
            tr = np.setdiff1d(np.arange(ds.n), te)
            b_te = tier_budgets(ds, te, CFG)[tier]
            b_tr = tier_budgets(ds, tr, CFG)[tier]
            router = LPBRouter(CFG, n_dom, seed=f, use_domain=False).fit(ds, tr, tier)
            res[tier]["lpb"].append(run_tier(ds, te, router.policy(), tier, b_te).mean_quality)
            tau = tune_cascade_tau(ds, tr, CFG, tier, b_tr)
            res[tier]["cascade"].append(run_tier(ds, te, StaticCascade(order, tau), tier, b_te).mean_quality)
            disc = DiscLR().fit(ds.features[tr], ds.domains[tr], ds.quality[tr])
            lam = tune_learned_lambda(disc, ds, tr, b_tr)
            res[tier]["learned"].append(run_tier(ds, te, LearnedRouter(disc, lam), tier, b_te).mean_quality)

    comp = {p: sum(W[t] * float(np.mean(res[t][p])) for t in TIERS)
            for p in ("lpb", "cascade", "learned")}
    print(f"\n[probe_routerbench_rehearsal] RouterBench-형 실데이터 경로 (5모델·6태스크, nf={nf}, "
          f"{time.time() - t0:.0f}s)")
    print(f"  {'tier':<10}{'LPB':<10}{'cascade':<10}{'learned':<10}")
    for tier in TIERS:
        print(f"  {tier:<10}{np.mean(res[tier]['lpb']):<10.4f}{np.mean(res[tier]['cascade']):<10.4f}"
              f"{np.mean(res[tier]['learned']):<10.4f}")
    print(f"  {'종합':<10}{comp['lpb']:<10.4f}{comp['cascade']:<10.4f}{comp['learned']:<10.4f}"
          f"   LPB−cascade={comp['lpb'] - comp['cascade']:+.4f}  LPB−learned={comp['lpb'] - comp['learned']:+.4f}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"source": "RouterBench-style (구조 모사, 실물 아님)", "n_folds": nf,
               "branch": rep["branch"], "composite": {k: round(v, 4) for k, v in comp.items()},
               "per_tier": {t: {p: round(float(np.mean(res[t][p])), 4) for p in res[t]} for t in TIERS}},
              open(OUT / "probe_routerbench_rehearsal.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("  → eval/results/probe_routerbench_rehearsal.json")
    return comp


if __name__ == "__main__":
    nf = 5
    if "--folds" in sys.argv:
        nf = int(sys.argv[sys.argv.index("--folds") + 1])
    main(nf)
