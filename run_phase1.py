"""Phase 1 실행: 두 세계 × 5-fold × 3 tier에서 베이스라인 점수표 산출.

완료 기준(PROJECT_PLAN.md): 베이스라인 3종의 fold별 점수표.
이후 모든 Phase의 개선 주장은 이 표 대비로만 측정한다.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import yaml
from src.synth import make_world
from src.harness import run_tier, tier_budgets, combined_score
from src.cost_mirror import cost_matrix
from baselines.policies import AllModel, StaticCascade, tune_cascade_tau

ROOT = pathlib.Path(__file__).resolve().parent
CFG = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))


def evaluate_world(world: str) -> dict:
    ds = make_world(CFG, world, seed=CFG["seed"])
    folds = ds.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])
    cheap_idx = int(np.argmin(cost_matrix(ds).mean(axis=0)))
    large_idx = int(np.argmax(cost_matrix(ds).mean(axis=0)))
    order = list(np.argsort(cost_matrix(ds).mean(axis=0)))

    rows = []
    for f, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(ds.n), test_idx)
        budgets = tier_budgets(ds, test_idx, CFG)
        for tier, budget in budgets.items():
            tau = tune_cascade_tau(ds, train_idx, CFG, tier, tier_budgets(ds, train_idx, CFG)[tier])
            policies = [
                AllModel(large_idx, "all-largest"),
                AllModel(cheap_idx, "all-cheapest"),
                StaticCascade(order, tau, "static-cascade"),
            ]
            for pol in policies:
                r = run_tier(ds, test_idx, pol, tier, budget)
                rows.append({
                    "world": world, "fold": f, "tier": tier, "policy": pol.name,
                    "tau": tau if pol.name == "static-cascade" else None,
                    "quality": round(r.mean_quality, 4),
                    "cost_used_pct": round(100 * r.total_cost / r.budget, 1),
                    "unanswered": r.unanswered,
                    "calls_per_q": round(r.calls_per_query, 2),
                    # 오라클은 반드시 동일 test fold에서 산출 (Phase 1 reflection: 기준 불일치 수정)
                    "oracle_any": round(float(ds.quality[test_idx].max(axis=1).mean()), 4),
                })
    return {"rows": rows, "oracle": oracle_reference(ds)}


def oracle_reference(ds) -> dict:
    """참고 상한: 예산 무제한 + 진짜 품질을 아는 오라클 (달성 불가능한 천장)."""
    best_any = ds.quality.max(axis=1).mean()          # 어떤 모델이든 정답이면 정답
    best_single = ds.quality.mean(axis=0).max()       # 최고 단일 모델
    return {"oracle_any_model": round(float(best_any), 4),
            "best_single_model": round(float(best_single), 4)}


def summarize(all_results: dict):
    print(f"\n{'='*100}")
    for world, res in all_results.items():
        print(f"\n[World: {world}]  oracle(any)={res['oracle']['oracle_any_model']}"
              f"  best-single={res['oracle']['best_single_model']}")
        print(f"{'tier':<10}{'policy':<18}{'quality(+-std)':<20}{'budget%':<10}"
              f"{'calls/q':<9}{'unans':<7}{'oracle(fold)':<12}")
        rows = res["rows"]
        keys = sorted({(r["tier"], r["policy"]) for r in rows},
                      key=lambda k: (["fast", "balanced", "premium"].index(k[0]), k[1]))
        for tier, pol in keys:
            sub = [r for r in rows if r["tier"] == tier and r["policy"] == pol]
            q = np.array([r["quality"] for r in sub])
            c = np.mean([r["cost_used_pct"] for r in sub])
            calls = np.mean([r["calls_per_q"] for r in sub])
            u = sum(r["unanswered"] for r in sub)
            orc = np.mean([r["oracle_any"] for r in sub])
            print(f"{tier:<10}{pol:<18}{q.mean():.4f} +-{q.std():.4f}    {c:>6.1f}%   "
                  f"{calls:>5.2f}    {u:<7}{orc:.4f}")
    print(f"\n{'='*100}")


if __name__ == "__main__":
    results = {w: evaluate_world(w) for w in ["irt", "specialist"]}
    out = ROOT / "eval" / "results"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out / "phase1.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    summarize(results)
    print(f"저장: {out / 'phase1.json'}")
