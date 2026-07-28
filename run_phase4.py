"""Phase 4 실행기: M3 쌍대가격 페이싱 + 투표 상자 (Phase 3 보류 해제).

  4a: 투표 상자 — 시뮬레이터 확장 검증 (연속성 assert + Wu et al. 포화 구조)
  4b: 페이싱 — 정적 λ vs mirror descent (전 tier + 도착 순서 스트레스 + η 강건성)
  4c: 풀스택 — 투표 상자 15개 + 페이싱 통합 vs 베이스라인 (종합 점수)

데이터: Phase 1과 동일 세계 (첫 샘플 보존 확장). 사용법: python run_phase4.py {4a|4b|4c}
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
from src.irt.mml import fit_mml
from src.harness import run_tier, tier_budgets
from src.cost_mirror import cost_matrix
from src.engine.pandora import NoiseModel, tune_lambda_replay, tune_lambda_quality
from src.engine.pacing import PacedPandora
from src.synth_ext import extend_with_votes
from phase2_stages import IRTEncoder
from run_phase2 import load_worlds_and_folds, CFG, OUT
from run_phase3 import make_pandora, fold_models
from baselines.policies import StaticCascade, tune_cascade_tau

N_DOM = CFG["synth"]["n_domains"]
_EXT_CACHE: dict = {}


def ext_worlds():
    worlds, folds = load_worlds_and_folds()
    ext = {w: extend_with_votes(ds, CFG) for w, ds in worlds.items()}
    return worlds, ext, folds


def ext_fold_models(eds, wname, f, folds):
    key = (wname, f)
    if key not in _EXT_CACHE:
        test_idx = folds[wname][f]
        tr = np.setdiff1d(np.arange(eds.n), test_idx)
        irt = fit_mml(eds.quality[tr], eds.domains[tr], N_DOM, per_domain_b=True)
        enc = IRTEncoder(irt["a"], irt["b"]).fit(eds.features[tr], eds.domains[tr],
                                                 eds.quality[tr])
        noise = NoiseModel().fit(eds.verifier[tr].ravel(), eds.quality[tr].ravel())
        _EXT_CACHE[key] = (tr, test_idx, irt, enc, noise)
    return _EXT_CACHE[key]


# ---------------- 4a ----------------

def stage_4a():
    worlds, ext, folds = ext_worlds()
    Ns = (1, 3, 5)
    for wname, ds in worlds.items():
        eds = ext[wname]
        # (1) 연속성: N=1 열 = Phase 1 데이터와 완전 동일
        for m in range(ds.m):
            assert (eds.quality[:, m * 3] == ds.quality[:, m]).all()
            assert (eds.verifier[:, m * 3] == ds.verifier[:, m]).all()
        print(f"[4a | {wname}] 연속성 OK (N=1 열 == Phase 1 quality/verifier)")

        # (2) Wu et al. Thm 1의 실제 예언 검증 (D8 수정): 투표의 포화 상한은
        #     I[정답질량 p > 최대 오답질량 (1−p)/4] — p>0.2 쿼리에서만 투표가 이득,
        #     p<0.2 쿼리에서는 투표가 해로움. 그룹별로 방향이 갈리는지 확인.
        from src.synth_ext import _true_p
        p_true = _true_p(ds, CFG)
        thresh = 1.0 / (1.0 + 4)                     # p > (1−p)/4  ⇔  p > 0.2
        print(f"  {'model':<12}{'HIGH(p>0.2): v1→v5':<24}{'LOW(p<0.2): v1→v5':<24}"
              f"{'Thm1방향OK':<8}")
        for m, mid in enumerate(ds.model_ids):
            hi = p_true[:, m] > thresh
            v1h, v5h = eds.quality[hi, m * 3].mean(), eds.quality[hi, m * 3 + 2].mean()
            if (~hi).sum() > 20:
                v1l, v5l = eds.quality[~hi, m * 3].mean(), eds.quality[~hi, m * 3 + 2].mean()
                low_txt, low_ok = f"{v1l:.3f}→{v5l:.3f}", v5l <= v1l + 0.02
            else:
                low_txt, low_ok = "(표본<20)", True
            ok = (v5h >= v1h - 0.02) and low_ok
            print(f"  {mid:<12}{f'{v1h:.3f}→{v5h:.3f}':<24}{low_txt:<24}"
                  f"{'O' if ok else 'X':<8}")

        # (3) 가성비 포인트: 상위권 모델의 vote 상자가 최상위 모델 1회를 넘는가
        cmat = cost_matrix(eds)
        c_mean, q_mean = cmat.mean(axis=0), eds.quality.mean(axis=0)
        for pair in [("large-34b#v3", "xl-70b"), ("mid-7b#v5", "large-34b")]:
            a_, b_ = eds.model_ids.index(pair[0]), eds.model_ids.index(pair[1])
            print(f"  [Wu-check] {pair[0]}: q={q_mean[a_]:.3f} c={c_mean[a_]:.4f}  vs  "
                  f"{pair[1]}×1: q={q_mean[b_]:.3f} c={c_mean[b_]:.4f}")


# ---------------- 4b ----------------

def _hard_first(ds, idx):
    if ds.world.startswith("irt"):
        key = ds.extras["theta"][idx]              # θ 낮음 = 어려움 → 오름차순
        return idx[np.argsort(key)]
    return idx[np.argsort(-ds.extras["difficulty"][idx])]


def stage_4b():
    worlds, ext, folds = ext_worlds()
    report = {}
    for wname, ds in worlds.items():
        rows = []
        for f in range(CFG["eval"]["k_folds"]):
            tr, test_idx, irt, enc, disc, noise = fold_models(ds, wname, f, folds)
            sub = np.random.default_rng(f).choice(tr, size=800, replace=False)
            b_te = tier_budgets(ds, test_idx, CFG)
            b_sub = tier_budgets(ds, sub, CFG)
            for tier in ["fast", "balanced", "premium"]:
                fac = lambda l: make_pandora(irt, enc, noise, l)
                lam0 = tune_lambda_replay(fac, ds, sub, b_sub[tier])
                pols = {"static": fac(lam0),
                        "paced": PacedPandora(fac, lam0)}
                for order_tag, order_idx in [("natural", test_idx),
                                             ("hard-first", _hard_first(ds, test_idx))]:
                    for pname, pol in pols.items():
                        r = run_tier(ds, order_idx, pol, tier, b_te[tier])
                        rows.append({"fold": f, "tier": tier, "order": order_tag,
                                     "policy": pname, "quality": round(r.mean_quality, 4),
                                     "cost_pct": round(100 * r.total_cost / r.budget, 1),
                                     "unans": r.unanswered})
        report[wname] = rows
    json.dump(report, open(OUT / "phase4b.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    for wname, rows in report.items():
        print(f"\n[4b | World: {wname}]  (5-fold 평균)")
        print(f"  {'tier':<10}{'order':<12}{'policy':<9}{'quality(+-std)':<20}"
              f"{'budget%':<10}{'unans':<6}")
        for tier in ["fast", "balanced", "premium"]:
            for order in ["natural", "hard-first"]:
                for pol in ["static", "paced"]:
                    sub = [r for r in rows if r["tier"] == tier
                           and r["order"] == order and r["policy"] == pol]
                    q = np.array([r["quality"] for r in sub])
                    c = np.mean([r["cost_pct"] for r in sub])
                    u = sum(r["unans"] for r in sub)
                    print(f"  {tier:<10}{order:<12}{pol:<9}{q.mean():.4f} +-{q.std():.4f}"
                          f"    {c:>6.1f}%   {u:<6}")


def stage_4b_eta():
    """게인 강건성 스윕: k_over ∈ {4, 8, 16} (fold 0, fast, natural)."""
    worlds, ext, folds = ext_worlds()
    for wname, ds in worlds.items():
        tr, test_idx, irt, enc, disc, noise = fold_models(ds, wname, 0, folds)
        sub = np.random.default_rng(0).choice(tr, size=800, replace=False)
        b_te = tier_budgets(ds, test_idx, CFG)["fast"]
        b_sub = tier_budgets(ds, sub, CFG)["fast"]
        fac = lambda l: make_pandora(irt, enc, noise, l)
        lam0 = tune_lambda_replay(fac, ds, sub, b_sub)
        line = f"[4b-gain | {wname} fast] "
        for k in [4.0, 8.0, 16.0]:
            r = run_tier(ds, test_idx, PacedPandora(fac, lam0, k_over=k), "fast", b_te)
            line += f"k_over={k}: q={r.mean_quality:.4f} unans={r.unanswered}  "
        print(line)


# ---------------- 4c ----------------

def stage_4c():
    worlds, ext, folds = ext_worlds()
    report = {}
    for wname in worlds:
        eds = ext[wname]
        rows = []
        order = list(np.argsort(cost_matrix(eds).mean(axis=0)))
        for f in range(CFG["eval"]["k_folds"]):
            tr, test_idx, irt, enc, noise = ext_fold_models(eds, wname, f, folds)
            sub = np.random.default_rng(f).choice(tr, size=800, replace=False)
            b_te = tier_budgets(eds, test_idx, CFG)
            b_sub = tier_budgets(eds, sub, CFG)
            b_tr = tier_budgets(eds, tr, CFG)
            for tier in ["fast", "balanced", "premium"]:
                fac = lambda l: make_pandora(irt, enc, noise, l)
                lam0 = tune_lambda_quality(fac, eds, sub, b_sub[tier])   # D11: 품질 우선
                tau = tune_cascade_tau(eds, tr, CFG, tier, b_tr[tier])
                paced = PacedPandora(fac, lam0, name="pandora-paced-ext")
                paced.stats = {}
                pols = [StaticCascade(order, tau, "cascade-ext"),
                        make_pandora(irt, enc, noise, lam0, name="pandora-static-ext"),
                        paced]
                for pol in pols:
                    r = run_tier(eds, test_idx, pol, tier, b_te[tier])
                    row = {"fold": f, "tier": tier, "policy": pol.name,
                           "quality": round(r.mean_quality, 4),
                           "cost_pct": round(100 * r.total_cost / r.budget, 1),
                           "calls_q": round(r.calls_per_query, 2),
                           "unans": r.unanswered}
                    if pol is paced:
                        total = max(sum(paced.stats.values()), 1)
                        vote_calls = sum(v for k, v in paced.stats.items()
                                         if "#v" in eds.model_ids[k])
                        row["vote_call_pct"] = round(100 * vote_calls / total, 1)
                    rows.append(row)
        report[wname] = rows
    json.dump(report, open(OUT / "phase4c.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    p3 = json.load(open(OUT / "phase3b.json", encoding="utf-8"))
    W = {t: CFG["tiers"][t]["weight"] for t in CFG["tiers"]}
    for wname, rows in report.items():
        print(f"\n[4c | World: {wname}]  (확장 상자 15개, 5-fold 평균)")
        print(f"  {'tier':<10}{'policy':<20}{'quality(+-std)':<20}{'budget%':<10}"
              f"{'calls/q':<9}{'unans':<6}")
        combined = {}
        for tier in ["fast", "balanced", "premium"]:
            for pol in ["cascade-ext", "pandora-static-ext", "pandora-paced-ext"]:
                sub = [r for r in rows if r["tier"] == tier and r["policy"] == pol]
                q = np.array([r["quality"] for r in sub])
                combined[pol] = combined.get(pol, 0.0) + W[tier] * q.mean()
                c = np.mean([r["cost_pct"] for r in sub])
                k = np.mean([r["calls_q"] for r in sub])
                u = sum(r["unans"] for r in sub)
                vote = ""
                if pol == "pandora-paced-ext":
                    vp = np.mean([r.get("vote_call_pct", 0) for r in sub])
                    vote = f"  vote호출={vp:.0f}%"
                print(f"  {tier:<10}{pol:<20}{q.mean():.4f} +-{q.std():.4f}    "
                      f"{c:>6.1f}%   {k:>5.2f}    {u:<6}{vote}")
        p3q = {t: np.mean([r["quality"] for r in p3[wname]
                           if r["tier"] == t and r["policy"] == "pandora-irt"])
               for t in W}
        p3_combined = sum(W[t] * p3q[t] for t in W)
        print(f"  종합점수: " + "  ".join(f"{k}={v:.4f}" for k, v in combined.items())
              + f"  | Phase3 pandora(5상자)={p3_combined:.4f}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "4a"
    {"4a": stage_4a, "4b": stage_4b, "4b-eta": stage_4b_eta, "4c": stage_4c}[stage]()
