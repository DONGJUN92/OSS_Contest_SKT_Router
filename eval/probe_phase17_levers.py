"""Phase 17 신설 레버 2종의 실측 (구현만 하고 재지 않으면 이 저장소의 규칙 위반).

  ① 사전 비용 추정 (`src/cost_model.py`): 호출 전 decode 길이를 모른다는 사실을 반영하면
     점수가 얼마나 떨어지는가, 안전마진이 그걸 되찾는가, 호스트 거부(rejected)는 몇 건인가.
     이전까지 정책은 **오라클 비용**을 봤으므로 이 측정 없이 보고한 점수는 낙관 편향이다.
  ② 무감독 프롬프트 군집 pseudo-domain (`src/cluster.py`): 런타임에 task 라벨이 없어 버린
     다집단 이득을 군집으로 되찾을 수 있는가. RouterBench 에서 다집단은 +0.026 이었다.

구성은 실전 근접(A.X 3모델 · Q2=No · textworld · 증강 검증기).
사용법: python eval/probe_phase17_levers.py [--folds 3] [--n 1000]
"""
import sys, pathlib, json, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from src.textworld import build_textworld, to_dataset
from src.text_encoder import get_encoder
from src.verifier import augmented_feature_matrix, fold_verifier_matrix
from src.harness import run_tier, tier_budgets
from src.router import LPBRouter
from src.cost_model import DecodeLengthEstimator, estimated_cost_matrix, estimation_report

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config_ax3.yaml", encoding="utf-8"))
OUT = ROOT / "eval" / "results"
TIERS = ["fast", "balanced", "premium"]
W = {t: CFG["tiers"][t]["weight"] for t in CFG["tiers"]}


def main(nf=3, n_queries=1000):
    t0 = time.time()
    cfg = dict(CFG, synth=dict(CFG["synth"], n_queries=n_queries))
    tw = build_textworld(cfg, seed=42)
    ds, meta = to_dataset(tw, cfg, Ns=(1,))
    ds.features = get_encoder("hashing").encode(meta["prompts"])
    F = augmented_feature_matrix(meta, text_dim=64, use_prompt=True, use_agreement=True)
    folds = ds.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])
    print(f"[probe_phase17_levers] n={ds.n} M={ds.m} folds={nf}", flush=True)

    # 추정 오차 자체를 먼저 보고 (마진 선택의 근거)
    est_full = estimated_cost_matrix(ds, mode="ridge", train_idx=np.arange(ds.n))
    err = estimation_report(ds, est_full)
    print(f"  비용 추정 오차 (ridge, 전체적합 참고): {err}", flush=True)

    arms = {
        "oracle-cost (기존 암묵 가정)": dict(cost_mode="oracle"),
        "ridge-cost margin 0.00": dict(cost_mode="ridge", cost_margin=0.0),
        "ridge-cost margin 0.05": dict(cost_mode="ridge", cost_margin=0.05),
        "ridge-cost margin 0.15": dict(cost_mode="ridge", cost_margin=0.15),
        "mean-cost margin 0.05": dict(cost_mode="mean", cost_margin=0.05),
        "cluster(k=6) oracle-cost": dict(cost_mode="oracle", use_domain="cluster", n_clusters=6),
        "cluster(k=12) oracle-cost": dict(cost_mode="oracle", use_domain="cluster", n_clusters=12),
    }
    res = {"cost_estimation_error": err, "n": ds.n, "n_folds": nf, "arms": {}}
    print(f"\n  {'구성':<30}{'종합':<10}{'무응답':<9}{'거부':<8}{'호출/쿼리':<10}")
    for name, kw in arms.items():
        use_dom = kw.pop("use_domain", False)
        nclu = kw.pop("n_clusters", 8)
        total, unans, rej, calls = 0.0, 0, 0, 0.0
        for tier in TIERS:
            qs, cs = [], []
            for f in range(nf):
                te = folds[f]
                tr = np.setdiff1d(np.arange(ds.n), te)
                ds.verifier, _ = fold_verifier_matrix(F, ds.quality, tr)
                r = LPBRouter(CFG, 1, seed=f, use_domain=use_dom, n_clusters=nclu,
                              **kw).fit(ds, tr, tier)
                est = r._est_cmat
                out = run_tier(ds, te, r.policy(), tier, tier_budgets(ds, te, CFG)[tier],
                               est_costs=est, cost_margin=kw.get("cost_margin", 0.0))
                qs.append(out.mean_quality)
                cs.append(out.calls_per_query)
                unans += out.unanswered
                rej += out.rejected
            total += W[tier] * float(np.mean(qs))
            calls += W[tier] * float(np.mean(cs))
        res["arms"][name] = {"composite": round(total, 4), "unanswered": unans,
                             "rejected": rej, "weighted_calls": round(calls, 3)}
        print(f"  {name:<30}{total:<10.4f}{unans:<9}{rej:<8}{calls:<10.3f}", flush=True)
        kw["cost_margin"] = kw.get("cost_margin", 0.0)          # 원복 (dict 재사용 방지)

    base = res["arms"]["oracle-cost (기존 암묵 가정)"]["composite"]
    best_cost = max((k for k in res["arms"] if "cost margin" in k),
                    key=lambda k: res["arms"][k]["composite"])
    best_clu = max((k for k in res["arms"] if k.startswith("cluster")),
                   key=lambda k: res["arms"][k]["composite"])
    res["verdict"] = {
        "oracle_cost_baseline": base,
        "honest_cost_best": {"arm": best_cost, "composite": res["arms"][best_cost]["composite"],
                             "delta_vs_oracle": round(res["arms"][best_cost]["composite"] - base, 4)},
        "cluster_best": {"arm": best_clu, "composite": res["arms"][best_clu]["composite"],
                         "delta_vs_single_group": round(res["arms"][best_clu]["composite"] - base, 4)},
    }
    print(f"\n  판정")
    print(f"   · 정직한 비용 추정의 대가: {res['verdict']['honest_cost_best']['delta_vs_oracle']:+.4f} "
          f"({best_cost})")
    print(f"   · 군집 pseudo-domain 의 값: {res['verdict']['cluster_best']['delta_vs_single_group']:+.4f} "
          f"({best_clu})")
    OUT.mkdir(parents=True, exist_ok=True)
    res["note"] = ("Phase 17. A.X 3모델·Q2=No·textworld·증강 검증기. rejected=비용 과소추정으로 "
                   "호스트가 거부한 호출 수. cluster=무감독 프롬프트 군집을 다집단 2PL 의 집단 축으로 사용.")
    json.dump(res, open(OUT / "probe_phase17_levers.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"  ({time.time() - t0:.0f}s) → eval/results/probe_phase17_levers.json")
    return res


if __name__ == "__main__":
    nf = int(sys.argv[sys.argv.index("--folds") + 1]) if "--folds" in sys.argv else 3
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 1000
    main(nf, n)
