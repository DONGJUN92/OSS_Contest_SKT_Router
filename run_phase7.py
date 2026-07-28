"""Phase 7 (고도화): 신규 적대 세계 3종에서 파이프라인 약점 탐지·개선.

  7a: World C(corr)/D(crossing)/E(nosignal) 구축 검증 + 최종 라우터 vs 베이스라인
      + 손실 분해로 약점 국소화
  7b: 개선 구현 후 재검증 (개선은 반드시 약점 셀 근거로만)
  7c: 회귀 매트릭스 — 개선판을 5개 세계 전부에서 재평가 (무회귀 확인)

사용법: python run_phase7.py {7a|7c}
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
from src.harness import run_tier, tier_budgets
from src.cost_mirror import cost_matrix
from src.synth import make_world
from src.synth2 import make_world2
from src.synth_ext import extend_with_votes
from src.router import LPBRouter
from run_phase2 import CFG, OUT
from run_diagnostics import diagnose
from baselines.policies import StaticCascade, AllModel, tune_cascade_tau

TIERS = ["fast", "balanced", "premium"]
W = {t: CFG["tiers"][t]["weight"] for t in CFG["tiers"]}
NEW_WORLDS = ["corr", "crossing", "nosignal"]
ALL_WORLDS = ["irt", "specialist"] + NEW_WORLDS


def build_world(wname):
    base = make_world(CFG, wname, seed=CFG["seed"]) if wname in ("irt", "specialist") \
        else make_world2(CFG, wname, seed=CFG["seed"])
    conc = 0.6 if wname == "crossing" else None        # D14: 교차 세계는 집중 오답
    eds = extend_with_votes(base, CFG, conc=conc)
    folds = base.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])
    return base, eds, folds


def world_stats(base):
    q = base.quality
    oracle_any = q.max(axis=1).mean()
    best_single = q.mean(axis=0).max()
    # 오류 상관: 인접 모델 오답 지시자 상관 평균
    err = 1 - q
    cors = [np.corrcoef(err[:, i], err[:, j])[0, 1]
            for i in range(q.shape[1]) for j in range(i + 1, q.shape[1])]
    return oracle_any, best_single, float(np.mean(cors))


def eval_router(eds, folds, tier, router_kwargs=None):
    qs, diags = [], []
    for f in range(CFG["eval"]["k_folds"]):
        test_idx = folds[f]
        tr = np.setdiff1d(np.arange(eds.n), test_idx)
        router = LPBRouter(CFG, CFG["synth"]["n_domains"], seed=f,
                           **(router_kwargs or {})).fit(eds, tr, tier)
        b_te = tier_budgets(eds, test_idx, CFG)[tier]
        d = diagnose(eds, test_idx, router, tier, b_te)
        qs.append(d["quality"])
        diags.append(d)
    cats = {k: round(float(np.mean([d["cats_pct"][k] for d in diags])), 2)
            for k in diags[0]["cats_pct"]}
    return float(np.mean(qs)), float(np.std(qs)), cats, \
        float(np.mean([d["oracle_any"] for d in diags]))


def eval_cascade(eds, folds, tier):
    order = list(np.argsort(cost_matrix(eds).mean(axis=0)))
    qs = []
    for f in range(CFG["eval"]["k_folds"]):
        test_idx = folds[f]
        tr = np.setdiff1d(np.arange(eds.n), test_idx)
        b_tr = tier_budgets(eds, tr, CFG)[tier]
        tau = tune_cascade_tau(eds, tr, CFG, tier, b_tr)
        b_te = tier_budgets(eds, test_idx, CFG)[tier]
        qs.append(run_tier(eds, test_idx, StaticCascade(order, tau), tier, b_te).mean_quality)
    return float(np.mean(qs))


def stage_7a(worlds=None, router_kwargs=None, tag="7a"):
    report = {}
    for wname in (worlds or NEW_WORLDS):
        base, eds, folds = build_world(wname)
        oa, bs, corr = world_stats(base)
        print(f"\n[{tag} | World: {wname}] oracle-any={oa:.4f} best-single={bs:.4f} "
              f"오류상관={corr:.3f}  (escalation 이론 여지={oa - bs:+.4f})")
        print(f"  {'tier':<10}{'LPB(+-std)':<19}{'cascade':<10}{'oracle':<9}"
              f"{'sel.err%':<9}{'regret%':<9}{'unans%':<8}")
        report[wname] = {}
        for tier in TIERS:
            q, sd, cats, oa_t = eval_router(eds, folds, tier, router_kwargs)
            qc = eval_cascade(eds, folds, tier)
            report[wname][tier] = {"lpb": round(q, 4), "std": round(sd, 4),
                                   "cascade": round(qc, 4), "oracle": round(oa_t, 4),
                                   "cats": cats}
            print(f"  {tier:<10}{q:.4f} +-{sd:.4f}    {qc:<10.4f}{oa_t:<9.4f}"
                  f"{cats['selection_error']:<9}{cats['early_stop_regret']:<9}"
                  f"{cats['unanswered']:<8}")
        comb_l = sum(W[t] * report[wname][t]["lpb"] for t in TIERS)
        comb_c = sum(W[t] * report[wname][t]["cascade"] for t in TIERS)
        print(f"  종합점수: LPB={comb_l:.4f}  cascade={comb_c:.4f}")
    json.dump(report, open(OUT / f"phase{tag}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return report


def stage_7c():
    """회귀 매트릭스: (개선 반영된) 최종 라우터를 5개 세계 전부에서."""
    stage_7a(worlds=ALL_WORLDS, tag="7c")


def stage_7b2():
    """페이싱 v3 적용 후 약점 세계 재검증."""
    stage_7a(worlds=["corr", "crossing"], tag="7b2")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "7a"
    {"7a": stage_7a, "7b2": stage_7b2, "7c": stage_7c}[stage]()
