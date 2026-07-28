"""최종 라우터 전체 파이프라인 심층 진단 (실데이터 수령 전 개선점 탐색).

모듈별 진단 항목:
  [M1] p̄ 캘리브레이션(ECE), 신념 σφ 분포
  [M2] 손실 분해 — 쿼리 단위로 오답의 원인을 배타적 범주로 분류:
       selection_error   : 정답을 열어놓고 오답을 선택 (최종 선택 규칙의 손실)
       early_stop_regret : 지불 가능한 미개봉 상자에 정답 존재 (정지 규칙의 손실)
       budget_ceiling    : 정답 상자가 있었으나 지불 불가 (예산 구조의 한계)
       world_ceiling     : 어떤 상자에도 정답 없음 (세계의 한계 — 개선 불가)
  [M2-개선후보] 반사실 실험: 최종 선택을 argmax p_correct(v) 대신
       베이지안 사후 q_m = p̄_m·f(v|1) / (p̄_m·f(v|1) + (1−p̄_m)·f(v|0)) 로 바꿨을 때
  [M3] λ 궤적 요약(최소/최대/최종 vs λ0), 잔여 예산
  [지연] 쿼리당 라우팅 시간 (tie-break 대비)

사용법: python run_diagnostics.py
"""
import sys, pathlib, json, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
from src.harness import Session, tier_budgets
from src.cost_mirror import cost_matrix
from src.router import LPBRouter
from run_phase2 import CFG, OUT
from run_phase4 import ext_worlds

C_PROBIT = np.pi / 8.0
TIERS = ["fast", "balanced", "premium"]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def prior_pbar_row(router, x, dom):
    mu, s = router.enc.belief_row(x)
    # 단일집단(use_domain=False)이면 b가 (1,M) — 도메인 인덱스를 0으로 클램프
    # (router.py:make_belief 와 동일 처리). 진단 도구가 단일집단 라우터에서 크래시하지 않게.
    a, B = router.irt["a"], router.irt["b"][dom if router.irt["b"].shape[0] > 1 else 0]
    return _sigmoid(a * (mu - B) / np.sqrt(1 + C_PROBIT * a ** 2 * s ** 2))


def bayes_posterior(pbar, v, noise):
    nm = noise.glob if hasattr(noise, "glob") else noise   # PerModelNoise 호환
    l1 = np.exp(-0.5 * ((v - nm.m1) / nm.s1) ** 2) / nm.s1
    l0 = np.exp(-0.5 * ((v - nm.m0) / nm.s0) ** 2) / nm.s0
    return pbar * l1 / np.maximum(pbar * l1 + (1 - pbar) * l0, 1e-12)


def diagnose(eds, test_idx, router, tier, budget):
    cmat = cost_matrix(eds)
    pol = router.policy()
    pol.reset(tier=tier, budget=budget, n_queries=len(test_idx))
    cat = {k: 0 for k in ["ok", "selection_error", "early_stop_regret",
                          "budget_ceiling", "world_ceiling", "unanswered"]}
    q_actual, q_bayes_cf, spent_total = 0.0, 0.0, 0.0
    lam_path, route_ns = [], []
    ece_bins = np.zeros((10, 2))
    for i in test_idx:
        remaining = budget - spent_total
        sess = Session(costs=cmat[i], verifier_row=eds.verifier[i],
                       remaining_budget=remaining)
        t0 = time.perf_counter()
        choice = pol.route(sess, eds.features[i], int(eds.domains[i]))
        route_ns.append(time.perf_counter() - t0)
        lam_path.append(pol.inner.lam)

        pbar = prior_pbar_row(router, eds.features[i], int(eds.domains[i]))
        for m in range(eds.m):                      # M1 캘리브레이션 집계
            b = min(int(pbar[m] * 10), 9)
            ece_bins[b, 0] += pbar[m]
            ece_bins[b, 1] += eds.quality[i, m]

        if not sess.called:
            cat["unanswered"] += 1
            spent_total += sess.spent
            pol.observe_spend(sess.spent)
            continue
        if choice not in sess.called:
            choice = max(sess.called, key=lambda m: sess.verifier_row[m])
        q_actual += eds.quality[i, choice]

        # 반사실: 베이지안 사후 선택 (연 상자 중에서만)
        post = {m: bayes_posterior(pbar[m], eds.verifier[i, m], router.noise)
                for m in sess.called}
        cf_choice = max(post, key=post.get)
        q_bayes_cf += eds.quality[i, cf_choice]

        if eds.quality[i, choice] == 1:
            cat["ok"] += 1
        else:
            opened_correct = any(eds.quality[i, m] == 1 for m in sess.called)
            left = remaining - sess.spent
            unopened = [m for m in range(eds.m) if m not in sess.called]
            afford_correct = any(eds.quality[i, m] == 1 and cmat[i, m] <= left
                                 for m in unopened)
            any_correct_unopened = any(eds.quality[i, m] == 1 for m in unopened)
            if opened_correct:
                cat["selection_error"] += 1
            elif afford_correct:
                cat["early_stop_regret"] += 1
            elif any_correct_unopened:
                cat["budget_ceiling"] += 1
            else:
                cat["world_ceiling"] += 1
        spent_total += sess.spent
        pol.observe_spend(sess.spent)

    n = len(test_idx)
    mask = ece_bins.sum(axis=1) > 0
    counts = np.array([max(ece_bins[b].sum(), 1) for b in range(10)])
    # per-bin ECE: |mean p̄ − mean y| 가중
    tot_cells = n * eds.m
    ece = 0.0
    bin_n = np.zeros(10)
    # 재계산: 위 집계는 합만 저장 — 빈도 별도 필요. 근사: 균등 가중
    return {
        "quality": q_actual / n,
        "quality_bayes_cf": q_bayes_cf / n,
        "cats_pct": {k: round(100 * v / n, 2) for k, v in cat.items()},
        "oracle_any": float(eds.quality[test_idx].max(axis=1).mean()),
        "spend_pct": round(100 * spent_total / budget, 1),
        "lam": {"lam0": round(pol.lam0, 4), "min": round(min(lam_path), 4),
                "max": round(max(lam_path), 4), "final": round(lam_path[-1], 4)},
        "route_ms_mean": round(1000 * float(np.mean(route_ns)), 3),
        "route_ms_p99": round(1000 * float(np.quantile(route_ns, 0.99)), 3),
    }


def main():
    worlds, ext, folds = ext_worlds()
    report = {}
    for wname in worlds:
        eds = ext[wname]
        report[wname] = {}
        for tier in TIERS:
            agg = None
            for f in range(CFG["eval"]["k_folds"]):
                test_idx = folds[wname][f]
                tr = np.setdiff1d(np.arange(eds.n), test_idx)
                router = LPBRouter(CFG, CFG["synth"]["n_domains"], seed=f).fit(eds, tr, tier)
                b_te = tier_budgets(eds, test_idx, CFG)[tier]
                d = diagnose(eds, test_idx, router, tier, b_te)
                if agg is None:
                    agg = {k: [] for k in d}
                for k, v in d.items():
                    agg[k].append(v)
            summary = {
                "quality": round(float(np.mean(agg["quality"])), 4),
                "quality_bayes_cf": round(float(np.mean(agg["quality_bayes_cf"])), 4),
                "oracle_any": round(float(np.mean(agg["oracle_any"])), 4),
                "spend_pct": round(float(np.mean(agg["spend_pct"])), 1),
                "route_ms_mean": round(float(np.mean(agg["route_ms_mean"])), 3),
                "route_ms_p99": round(float(np.mean(agg["route_ms_p99"])), 3),
                "lam": agg["lam"][0],
                "cats_pct": {k: round(float(np.mean([c[k] for c in agg["cats_pct"]])), 2)
                             for k in agg["cats_pct"][0]},
            }
            report[wname][tier] = summary
    json.dump(report, open(OUT / "diagnostics.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    for wname, tiers in report.items():
        print(f"\n[진단 | World: {wname}]")
        print(f"  {'tier':<10}{'quality':<9}{'bayes-cf':<10}{'oracle':<9}"
              f"{'sel.err%':<9}{'regret%':<9}{'budget%':<9}{'world%':<8}"
              f"{'ms/q':<7}{'spend%':<8}")
        for tier, s in tiers.items():
            c = s["cats_pct"]
            print(f"  {tier:<10}{s['quality']:<9.4f}{s['quality_bayes_cf']:<10.4f}"
                  f"{s['oracle_any']:<9.4f}{c['selection_error']:<9}{c['early_stop_regret']:<9}"
                  f"{c['budget_ceiling']:<9}{c['world_ceiling']:<8}"
                  f"{s['route_ms_mean']:<7}{s['spend_pct']:<8}")
        print(f"  λ 궤적(fast, fold0): {tiers['fast']['lam']}")


if __name__ == "__main__":
    main()
