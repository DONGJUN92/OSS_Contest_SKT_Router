"""레드팀 결함 폐쇄 측정 (Phase 17) — 실전에 가장 가까운 구성에서 직접 재판정.

레드팀의 핵심 주장은 두 개였다.
  ① 자작 검증기가 실텍스트에서 무정보(AUC 0.61)라 LPB 의 순차 관측이 순손실이다.
  ② 가중치 최고인 fast tier 에서 σ<0 로 1회 호출 퇴화 → 학습형 단일호출과 같은 정책 클래스.
그리고 별도로 ③ cascade 베이스라인이 예산맹목 파산하는 strawman 이라는 지적이 있었다.

여기서는 **AUC 대리지표가 아니라 종합 점수로** 재판정한다. 구성은 실전에 가장 가까운
A.X 3모델 · Q2=No(단일 샘플) · 실텍스트 · 실 ScoringVerifier 다.

비교군 (모두 동일 fold·동일 예산·동일 신념)
  LPB              — 순차 개봉·정지 (제출 정책)
  learned          — DiscLR 예측 후 1회 호출 (RouteLLM류). 검증기를 **쓰지 않는다**
  cascade(static)  — 예산맹목 (기존 베이스라인)
  cascade(budget)  — 문항당 허용액 인지 (Phase 17 공정화)
  cascade-routing  — 1스텝 lookahead 결합 결정 (Dekoninck 계열, arXiv:2410.10347)

검증기 2종을 교차한다: 현행 13특성 vs 증강(tail+교차모델 일치+텍스트 해싱+상자 원-핫).
증강 검증기의 `agree_ref` 는 **호출한 출력들 사이의 일치**이므로 단일호출 정책은 원리적으로
쓸 수 없다 — 그래서 AUC 만으로는 정책 간 비교가 결정되지 않고, 이 측정이 필요하다.

사용법: python eval/probe_redteam_closure.py [--folds 3] [--n 1200]
"""
import sys, pathlib, json, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from src.textworld import build_textworld, to_dataset
from src.text_encoder import get_encoder
from src.verifier import (feature_matrix, augmented_feature_matrix, fold_verifier_matrix)
from src.harness import run_tier, tier_budgets
from src.cost_mirror import cost_matrix
from src.router import LPBRouter
from src.encoder import DiscLR
from baselines.policies import (StaticCascade, tune_cascade_tau, BudgetCascade,
                                tune_budget_cascade_tau, CascadeRouting,
                                LearnedRouter, tune_learned_lambda, SelectiveRouter)
from src.engine.pandora import tune_lambda_quality, FINE_MULTS

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config_ax3.yaml", encoding="utf-8"))
OUT = ROOT / "eval" / "results"
TIERS = ["fast", "balanced", "premium"]
W = {t: CFG["tiers"][t]["weight"] for t in CFG["tiers"]}
POLS = ["lpb", "learned", "cascade_static", "cascade_budget", "cascade_routing", "selective3"]


def auc(s, y):
    s, y = np.asarray(s).ravel(), np.asarray(y).ravel()
    pos, neg = s[y > 0.5], s[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main(nf=3, n_queries=1200):
    t0 = time.time()
    cfg = dict(CFG, synth=dict(CFG["synth"], n_queries=n_queries))
    tw = build_textworld(cfg, seed=42)
    ds, meta = to_dataset(tw, cfg, Ns=(1,))                    # Q2=No — 실전 조건
    ds.features = get_encoder("hashing").encode(meta["prompts"])
    F_cur = feature_matrix(meta)
    F_aug = augmented_feature_matrix(meta, text_dim=64, use_prompt=True,
                                     use_agreement=True, ref_col=0)
    folds = ds.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])
    order = list(np.argsort(cost_matrix(ds).mean(axis=0)))
    print(f"[probe_redteam_closure] A.X 3모델 · Q2=No · textworld  n={ds.n} M={ds.m} "
          f"folds={nf}", flush=True)

    res = {}
    for vname, F in (("current13", F_cur), ("augmented", F_aug)):
        comp = {p: 0.0 for p in POLS}
        calls = {p: 0.0 for p in POLS}
        per_tier = {}
        aucs, sneg, picks = [], [], []
        for tier in TIERS:
            acc = {p: [] for p in POLS}
            cl = {p: [] for p in POLS}
            for f in range(nf):
                te = folds[f]
                tr = np.setdiff1d(np.arange(ds.n), te)
                V, _ = fold_verifier_matrix(F, ds.quality, tr)
                ds.verifier = V
                if tier == "fast":
                    aucs.append(auc(V[te], ds.quality[te]))
                b_te = tier_budgets(ds, te, CFG)[tier]
                b_tr = tier_budgets(ds, tr, CFG)[tier]

                r = LPBRouter(CFG, 1, seed=f, use_domain=False).fit(ds, tr, tier)
                sneg.append(r.diagnostics["sigma_neg_frac"]) if tier == "fast" else None
                out = run_tier(ds, te, r.policy(), tier, b_te)
                acc["lpb"].append(out.mean_quality)
                cl["lpb"].append(out.calls_per_query)

                disc = DiscLR().fit(ds.features[tr], ds.domains[tr], ds.quality[tr])
                o = run_tier(ds, te, LearnedRouter(disc, tune_learned_lambda(disc, ds, tr, b_tr)),
                             tier, b_te)
                acc["learned"].append(o.mean_quality)
                cl["learned"].append(o.calls_per_query)

                o = run_tier(ds, te, StaticCascade(order, tune_cascade_tau(ds, tr, CFG, tier, b_tr)),
                             tier, b_te)
                acc["cascade_static"].append(o.mean_quality)
                cl["cascade_static"].append(o.calls_per_query)

                o = run_tier(ds, te, BudgetCascade(order, tune_budget_cascade_tau(
                    ds, tr, CFG, tier, b_tr)), tier, b_te)
                acc["cascade_budget"].append(o.mean_quality)
                cl["cascade_budget"].append(o.calls_per_query)

                mb = r._factory(1.0).make_belief
                # 대등 튜닝 (Phase 19) — LPB 와 같은 세밀격자 + SE 문턱을 준다
                lam_cr = tune_lambda_quality(lambda l: CascadeRouting(mb, l), ds, tr, b_tr,
                                             mults=FINE_MULTS, se_k=1.0)
                o = run_tier(ds, te, CascadeRouting(mb, lam_cr), tier, b_te)
                acc["cascade_routing"].append(o.mean_quality)
                cl["cascade_routing"].append(o.calls_per_query)

                # 3후보 메타 선택 (LPB 기본 · margin 문턱). cascade_routing 이 LPB 를 이기는
                # 것이 측정된 뒤 후보로 승격된 것이므로, 이 행이 "데이터가 고른" 결과다.
                sr = SelectiveRouter(CFG, 1, seed=f, use_domain=False).fit(ds, tr, tier)
                o = run_tier(ds, te, sr.policy(), tier, b_te)
                acc["selective3"].append(o.mean_quality)
                cl["selective3"].append(o.calls_per_query)
                picks.append(sr.chosen)
            per_tier[tier] = {p: round(float(np.mean(acc[p])), 4) for p in POLS}
            per_tier[tier]["_calls"] = {p: round(float(np.mean(cl[p])), 3) for p in POLS}
            for p in POLS:
                comp[p] += W[tier] * float(np.mean(acc[p]))
                calls[p] += W[tier] * float(np.mean(cl[p]))
        res[vname] = {"composite": {p: round(comp[p], 4) for p in POLS},
                      "weighted_calls": {p: round(calls[p], 3) for p in POLS},
                      "per_tier": per_tier,
                      "verifier_auc_fast": round(float(np.mean(aucs)), 4),
                      "sigma_neg_frac_fast": round(float(np.mean(sneg)), 4) if sneg else None,
                      "lpb_minus_learned": round(comp["lpb"] - comp["learned"], 4),
                      "best_policy": max(comp, key=comp.get),
                      "selective_picks": {k: picks.count(k) for k in sorted(set(picks))}}
        print(f"\n  검증기={vname}  (fast tier AUC {res[vname]['verifier_auc_fast']:.3f}, "
              f"σ<0 비율 {res[vname]['sigma_neg_frac_fast']})")
        for tier in TIERS:
            pt = per_tier[tier]
            print(f"    {tier:<9} " + "  ".join(
                f"{p}={pt[p]:.4f}/{pt['_calls'][p]:.2f}c" for p in POLS))
        print(f"    {'종합':<9} " + "  ".join(f"{p}={comp[p]:.4f}" for p in POLS))
        print(f"    ★ LPB − learned = {res[vname]['lpb_minus_learned']:+.4f}   "
              f"최고 정책 = {res[vname]['best_policy']}   메타선택 = "
              f"{res[vname]['selective_picks']}")

    a = res["augmented"]
    res["closure"] = {
        "delta_lpb_minus_learned": round(a["lpb_minus_learned"]
                                        - res["current13"]["lpb_minus_learned"], 4),
        "lpb_beats_learned": a["lpb_minus_learned"] > 0,
        "lpb_is_best": a["best_policy"] == "lpb",
        "selective_at_least_lpb": a["composite"]["selective3"] >= a["composite"]["lpb"] - 0.002,
        "verdict": ("PARTIAL — LPB 는 학습형·cascade 를 앞서지만 cascade-routing 이 LPB 를 "
                    "앞선다. 메타 선택(selective3)이 그 손실을 흡수하는지가 판정 기준"
                    if a["best_policy"] != "lpb" else
                    "CLOSED — 증강 구성에서 LPB 가 전 비교군 최고")}
    print(f"\n  폐쇄 판정: {res['closure']['verdict']}")
    OUT.mkdir(parents=True, exist_ok=True)
    res["note"] = ("Phase 17 레드팀 폐쇄 측정. A.X 3모델·Q2=No·textworld·실 ScoringVerifier. "
                   "cascade_budget/cascade_routing 은 Phase 17 신설 공정 베이스라인.")
    json.dump(res, open(OUT / "probe_redteam_closure.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"  ({time.time() - t0:.0f}s) → eval/results/probe_redteam_closure.json")
    return res


if __name__ == "__main__":
    nf = int(sys.argv[sys.argv.index("--folds") + 1]) if "--folds" in sys.argv else 3
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 1200
    main(nf, n)
