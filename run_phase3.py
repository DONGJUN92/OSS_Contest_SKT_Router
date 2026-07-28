"""Phase 3 실행기: M2 베이지안 재지수 Weitzman 엔진.

  3a: 엔진 정확성 — 동적계획법(DP) 최적해와 대조 + λ 단조성 검증
  3b: 전면 비교 — static-cascade / single-shot(2c) / pandora-irt (강화 기준)
  3c: ablation — θ 갱신 on/off, IRT vs 판별식 신념 (독창성 입증 책임)

데이터: Phase 1과 동일 (seed 42 두 세계 + 동일 층화 5-fold).
사용법: python run_phase3.py {3a|3b|3c}
"""
import sys, pathlib, json
from functools import lru_cache
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
from src.irt.mml import fit_mml
from src.harness import run_tier, tier_budgets, Session
from src.cost_mirror import cost_matrix
from src.engine.pandora import (NoiseModel, ExactNoise, GridBelief, FixedBelief,
                                PandoraPolicy, tune_lambda_replay)
from phase2_stages import IRTEncoder, DiscLR, SingleShotIndex, _tune_lambda
from run_phase2 import load_worlds_and_folds, CFG, OUT
from baselines.policies import StaticCascade, tune_cascade_tau

N_DOM = CFG["synth"]["n_domains"]
_CACHE: dict = {}


def fold_models(ds, wname, f, folds):
    """(world, fold)별 MML+인코더+잡음모델 적합 결과 캐시 (스테이지 간 재사용)."""
    key = (wname, f)
    if key not in _CACHE:
        test_idx = folds[wname][f]
        tr = np.setdiff1d(np.arange(ds.n), test_idx)
        irt = fit_mml(ds.quality[tr], ds.domains[tr], N_DOM, per_domain_b=True)
        enc = IRTEncoder(irt["a"], irt["b"]).fit(ds.features[tr], ds.domains[tr], ds.quality[tr])
        disc = DiscLR().fit(ds.features[tr], ds.domains[tr], ds.quality[tr])
        noise = NoiseModel().fit(ds.verifier[tr].ravel(), ds.quality[tr].ravel())
        _CACHE[key] = (tr, test_idx, irt, enc, disc, noise)
    return _CACHE[key]


def make_pandora(irt, enc, noise, lam, do_update=True, name="pandora-irt",
                 stop_margin=0.0):
    def make_belief(x, dom):
        mu, s = enc.belief_row(x)
        return GridBelief(mu, s, irt["a"], irt["b"][dom], noise, do_update=do_update)
    return PandoraPolicy(make_belief, lam, name, stop_margin=stop_margin)


def make_pandora_disc(disc, noise, lam, name="pandora-disc"):
    def make_belief(x, dom):
        return FixedBelief(disc.predict_row(x, dom), noise)
    return PandoraPolicy(make_belief, lam, name)


# ---------------- 3a: DP 대조 + λ 단조성 ----------------

def dp_value(p, c, lam):
    """독립 상자·정확 관측·must-open≥1 의 전역 최적 기대 순가치 (완전 탐색 DP)."""
    M = len(p)

    @lru_cache(maxsize=None)
    def V(S, best):
        vals = [float(best)]
        for m in range(M):
            if not (S >> m) & 1:
                S2 = S | (1 << m)
                vals.append(-lam * c[m] + p[m] * V(S2, 1.0) + (1 - p[m]) * V(S2, best))
        return max(vals)

    return max(-lam * c[m] + p[m] * V(1 << m, 1.0) + (1 - p[m]) * V(1 << m, 0.0)
               for m in range(M))


def engine_value(p, c, lam):
    """엔진의 기대 순가치: 상금 조합 전수 열거 × 결정적 리플레이."""
    M = len(p)
    exact = ExactNoise()
    ev = 0.0
    for mask in range(2 ** M):
        y = np.array([(mask >> m) & 1 for m in range(M)], dtype=float)
        prob = float(np.prod([p[m] if y[m] else 1 - p[m] for m in range(M)]))
        pol = PandoraPolicy(lambda x, d: FixedBelief(p, exact), lam)
        sess = Session(costs=np.array(c), verifier_row=y, remaining_budget=1e9)
        choice = pol.route(sess, None, 0)
        ev += prob * (y[choice] - lam * sess.spent)
    return ev


def stage_3a():
    rng = np.random.default_rng(0)
    gaps = []
    for _ in range(300):
        M = int(rng.integers(2, 5))
        p = rng.uniform(0.2, 0.9, size=M)
        c = rng.uniform(0.01, 0.3, size=M)
        lam = float(rng.uniform(0.5, 2.0))
        gaps.append(dp_value(tuple(p), tuple(c), lam) - engine_value(p, c, lam))
    gaps = np.array(gaps)
    print(f"[3a-1] DP 최적해 대비 엔진 기대가치 격차 (300 무작위 인스턴스, M=2..4)")
    print(f"       mean={gaps.mean():.2e}  max={gaps.max():.2e}  "
          f"(0에 가까울수록 Weitzman 최적성 재현)")

    worlds, folds = load_worlds_and_folds()
    print(f"\n[3a-2] λ 단조성: λ 증가 ⇒ 호출수·지출 단조 감소 (fold 0)")
    for wname, ds in worlds.items():
        tr, test_idx, irt, enc, disc, noise = fold_models(ds, wname, 0, folds)
        budget = tier_budgets(ds, test_idx, CFG)["balanced"]
        lams = [0.05, 0.2, 0.8, 3.2, 12.8]
        calls, spends = [], []
        for lam in lams:
            r = run_tier(ds, test_idx, make_pandora(irt, enc, noise, lam), "x", budget * 1e6)
            calls.append(round(r.calls_per_query, 3))
            spends.append(round(r.total_cost, 1))
        mono = all(calls[i] >= calls[i + 1] for i in range(len(calls) - 1))
        print(f"  {wname:<12} calls/q={calls}  단조감소={'OK' if mono else 'FAIL'}")


# ---------------- 3b: 전면 비교 (강화 기준) ----------------

def stage_3b():
    worlds, folds = load_worlds_and_folds()
    report = {}
    for wname, ds in worlds.items():
        rows = []
        order = list(np.argsort(cost_matrix(ds).mean(axis=0)))
        for f in range(CFG["eval"]["k_folds"]):
            tr, test_idx, irt, enc, disc, noise = fold_models(ds, wname, f, folds)
            sub = np.random.default_rng(f).choice(tr, size=800, replace=False)  # λ 튜닝 표본
            budgets_te = tier_budgets(ds, test_idx, CFG)
            budgets_sub = tier_budgets(ds, sub, CFG)
            budgets_tr = tier_budgets(ds, tr, CFG)
            for tier in ["fast", "balanced", "premium"]:
                tau = tune_cascade_tau(ds, tr, CFG, tier, budgets_tr[tier])
                lam_ss = _tune_lambda(enc, ds, tr, budgets_tr[tier])
                lam_pd = tune_lambda_replay(
                    lambda l: make_pandora(irt, enc, noise, l), ds, sub, budgets_sub[tier])
                pols = [
                    StaticCascade(order, tau, "static-cascade"),
                    SingleShotIndex(enc, lam_ss, "single-shot"),
                    make_pandora(irt, enc, noise, lam_pd, name="pandora-irt"),
                ]
                for pol in pols:
                    r = run_tier(ds, test_idx, pol, tier, budgets_te[tier])
                    rows.append({"fold": f, "tier": tier, "policy": pol.name,
                                 "quality": round(r.mean_quality, 4),
                                 "cost_pct": round(100 * r.total_cost / r.budget, 1),
                                 "calls_q": round(r.calls_per_query, 2),
                                 "unans": r.unanswered})
        report[wname] = rows
    json.dump(report, open(OUT / "phase3b.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    _print_table(report, ["static-cascade", "single-shot", "pandora-irt"],
                 ["fast", "balanced", "premium"], "3b")


# ---------------- 3c: ablation ----------------

def stage_3c():
    worlds, folds = load_worlds_and_folds()
    report = {}
    for wname, ds in worlds.items():
        rows = []
        for f in range(CFG["eval"]["k_folds"]):
            tr, test_idx, irt, enc, disc, noise = fold_models(ds, wname, f, folds)
            sub = np.random.default_rng(100 + f).choice(tr, size=800, replace=False)
            budgets_te = tier_budgets(ds, test_idx, CFG)
            budgets_sub = tier_budgets(ds, sub, CFG)
            variants = {
                "pandora-irt": lambda l: make_pandora(irt, enc, noise, l),
                "pandora-noupd": lambda l: make_pandora(irt, enc, noise, l, do_update=False,
                                                        name="pandora-noupd"),
                "pandora-disc": lambda l: make_pandora_disc(disc, noise, l),
            }
            for tier in ["fast", "balanced"]:
                for name, factory in variants.items():
                    lam = tune_lambda_replay(factory, ds, sub, budgets_sub[tier])
                    r = run_tier(ds, test_idx, factory(lam), tier, budgets_te[tier])
                    rows.append({"fold": f, "tier": tier, "policy": name,
                                 "quality": round(r.mean_quality, 4),
                                 "cost_pct": round(100 * r.total_cost / r.budget, 1),
                                 "calls_q": round(r.calls_per_query, 2),
                                 "unans": r.unanswered})
        report[wname] = rows
    json.dump(report, open(OUT / "phase3c.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    _print_table(report, ["pandora-irt", "pandora-noupd", "pandora-disc"],
                 ["fast", "balanced"], "3c")


def _print_table(report, policies, tiers, tag):
    for wname, rows in report.items():
        print(f"\n[{tag} | World: {wname}]")
        print(f"  {'tier':<10}{'policy':<16}{'quality(+-std)':<20}{'budget%':<10}"
              f"{'calls/q':<9}{'unans':<6}")
        for tier in tiers:
            for pol in policies:
                sub = [r for r in rows if r["tier"] == tier and r["policy"] == pol]
                if not sub:
                    continue
                q = np.array([r["quality"] for r in sub])
                c = np.mean([r["cost_pct"] for r in sub])
                k = np.mean([r["calls_q"] for r in sub])
                u = sum(r["unans"] for r in sub)
                print(f"  {tier:<10}{pol:<16}{q.mean():.4f} +-{q.std():.4f}    "
                      f"{c:>6.1f}%   {k:>5.2f}    {u:<6}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "3a"
    {"3a": stage_3a, "3b": stage_3b, "3c": stage_3c}[stage]()
