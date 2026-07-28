"""Phase 5 실행기: M4 Conformal 정지 보증.

위험 정의 (조기 정지 후회, early-stop regret):
  쿼리에서 정책이 자발적으로 정지했을 때,
  L = 1[선택 답이 오답 ∧ 미개봉 상자 중 정답이 존재 ∧ 그 상자가 지불 가능했음]
τ(정지 마진)↑ ⇒ 더 늦게 정지 ⇒ L 단조 감소 → conformal risk control(CRC) 적용 가능:
  τ̂ = min{ τ : (n·R̂(τ) + 1)/(n + 1) ≤ α }   (유한표본, 분포무관 보증)

  5a: CRC 구현 + 분포 내 유효성 (위험 단조성, 보증 준수, 품질 영향)
  5b: 분포 이동 스트레스 — 검증기 잡음 ×2 / 도메인 편중 / 미지 도메인(LODO)
  5c: 풀스택 종합 영향

데이터: Phase 4 확장 세계 (Phase 1 데이터 보존). 사용법: python run_phase5.py {5a|5b|5c}
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
from src.irt.mml import fit_mml
from src.harness import Session, tier_budgets
from src.cost_mirror import cost_matrix
from src.engine.pandora import NoiseModel, tune_lambda_quality
from src.engine.pacing import PacedPandora
from phase2_stages import IRTEncoder
from run_phase2 import load_worlds_and_folds, CFG, OUT
from run_phase3 import make_pandora
from run_phase4 import ext_worlds

N_DOM = CFG["synth"]["n_domains"]
TAU_GRID = [0.0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.22, 0.30, 0.40]


def audit_tier(eds, idx, policy, budget, verifier_override=None):
    """run_tier와 동일 의미론 + 조기 정지 후회 이벤트 계측 (평가 전용: 진짜 품질 사용)."""
    cmat = cost_matrix(eds)
    ver = verifier_override if verifier_override is not None else eds.verifier
    q_sum, spent_total, unans, events, n = 0.0, 0.0, 0, 0, len(idx)
    if hasattr(policy, "reset"):
        policy.reset(tier="audit", budget=budget, n_queries=n)
    for i in idx:
        remaining = budget - spent_total
        sess = Session(costs=cmat[i], verifier_row=ver[i], remaining_budget=remaining)
        choice = policy.route(sess, eds.features[i], int(eds.domains[i]))
        if not sess.called:
            unans += 1
            spent_total += sess.spent
            if hasattr(policy, "observe_spend"):
                policy.observe_spend(sess.spent)
            continue
        if choice not in sess.called:
            choice = max(sess.called, key=lambda m: sess.verifier_row[m])
        # 조기 정지 후회: 오답 선택 ∧ 지불 가능했던 미개봉 상자에 정답 존재
        if eds.quality[i, choice] == 0:
            left = remaining - sess.spent
            unopened = [m for m in range(eds.m) if m not in sess.called]
            if any(eds.quality[i, m] == 1 and cmat[i, m] <= left for m in unopened):
                events += 1
        q_sum += eds.quality[i, choice]
        spent_total += sess.spent
        if hasattr(policy, "observe_spend"):
            policy.observe_spend(sess.spent)
    return {"quality": q_sum / n, "risk": events / n, "unans": unans,
            "spend_pct": 100 * spent_total / budget}


def fit_split(eds, tr_fit):
    irt = fit_mml(eds.quality[tr_fit], eds.domains[tr_fit], N_DOM, per_domain_b=True)
    enc = IRTEncoder(irt["a"], irt["b"]).fit(eds.features[tr_fit], eds.domains[tr_fit],
                                             eds.quality[tr_fit])
    noise = NoiseModel().fit(eds.verifier[tr_fit].ravel(), eds.quality[tr_fit].ravel())
    return irt, enc, noise


def crc_tau(eds, cal_idx, factory, budget, alpha):
    """CRC: 보정셋 리플레이로 R̂(τ) 곡선을 얻고 τ̂ = min{τ: (nR̂+1)/(n+1) ≤ α}."""
    n = len(cal_idx)
    curve = []
    for tau in TAU_GRID:
        r = audit_tier(eds, cal_idx, factory(tau), budget)
        curve.append((tau, r["risk"]))
    for tau, risk in curve:
        if (n * risk + 1) / (n + 1) <= alpha:
            return tau, curve
    return TAU_GRID[-1], curve


def build(eds, wname, f, folds, tier, seed=0):
    """fold f: train을 fit(70%)/cal(30%)로 분할, λ0(quality-first)·정책 팩토리 구성."""
    test_idx = folds[wname][f]
    tr = np.setdiff1d(np.arange(eds.n), test_idx)
    rng = np.random.default_rng(seed + f)
    perm = rng.permutation(tr)
    tr_fit, tr_cal = perm[:int(0.7 * len(perm))], perm[int(0.7 * len(perm)):]
    irt, enc, noise = fit_split(eds, tr_fit)
    b_cal = tier_budgets(eds, tr_cal, CFG)[tier]
    b_te = tier_budgets(eds, test_idx, CFG)[tier]
    fac0 = lambda l: make_pandora(irt, enc, noise, l)
    lam0 = tune_lambda_quality(fac0, eds, tr_cal, b_cal)
    def factory(tau):
        return PacedPandora(lambda l: make_pandora(irt, enc, noise, l, stop_margin=tau),
                            lam0, name=f"paced-tau{tau}")
    return test_idx, tr_cal, b_cal, b_te, factory


class ConservativeNoise:
    """M4 v2 (D12 수정): 예산-중립적 관측 채널 보수화.

    궤적 수준 CRC(정지 마진 τ)는 전역 예산과 충돌(5a 부정적 결과) — 대신 상금 추정
    자체를 보수화한다: 보정셋을 v-분위 구간으로 나눠 구간별 P(y=1)의
    Clopper–Pearson 하한(유한표본·정확 이항)을 취하고 등단조화(isotonic)한다.
    q_lo(v) ≤ 진짜 P(y=1|bin) 가 구간별 1−δ 로 보장 → 엔진이 확신 없는 답에서
    자연히 승급하고, 예산 영향은 페이싱(λ)이 흡수한다.
    """

    def __init__(self, base, delta=0.05, bins=10):
        self.base, self.delta, self.bins = base, delta, bins

    def fit(self, v, y):
        from scipy.stats import beta as beta_dist
        edges = np.quantile(v, np.linspace(0, 1, self.bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        self.edges = edges
        lows = []
        for b in range(self.bins):
            mask = (v > edges[b]) & (v <= edges[b + 1])
            n, k = int(mask.sum()), int(y[mask].sum())
            if n == 0:
                lows.append(0.0)
            elif k == 0:
                lows.append(0.0)
            else:
                lows.append(float(beta_dist.ppf(self.delta, k, n - k + 1)))
        self.q_lo = np.maximum.accumulate(np.array(lows))     # 등단조화 (v↑ ⇒ 신뢰↑)
        return self

    def p_correct(self, v, m=None):
        b = int(np.searchsorted(self.edges, v, side="left")) - 1
        return float(self.q_lo[np.clip(b, 0, self.bins - 1)])

    def lik_ratio(self, v, m=None):
        return self.base.lik_ratio(v)


def build_v2(eds, wname, f, folds, tier, conservative, seed=0):
    """M4 v2용: 잡음모델을 raw/보수화로 선택하는 빌더."""
    test_idx = folds[wname][f]
    tr = np.setdiff1d(np.arange(eds.n), test_idx)
    rng = np.random.default_rng(seed + f)
    perm = rng.permutation(tr)
    tr_fit, tr_cal = perm[:int(0.7 * len(perm))], perm[int(0.7 * len(perm)):]
    irt, enc, noise = fit_split(eds, tr_fit)
    if conservative:
        noise = ConservativeNoise(noise).fit(eds.verifier[tr_cal].ravel(),
                                             eds.quality[tr_cal].ravel())
    b_cal = tier_budgets(eds, tr_cal, CFG)[tier]
    b_te = tier_budgets(eds, test_idx, CFG)[tier]
    fac0 = lambda l: make_pandora(irt, enc, noise, l)
    lam0 = tune_lambda_quality(fac0, eds, tr_cal, b_cal)
    pol = PacedPandora(lambda l: make_pandora(irt, enc, noise, l), lam0,
                       name="cons" if conservative else "raw")
    return test_idx, tr_cal, b_cal, b_te, pol


def stage_5a2():
    """M4 v2 분포 내 검증: 보수화의 비용(품질 delta)과 위험 감소."""
    worlds, ext, folds = ext_worlds()
    print(f"\n[5a2] 보수화 관측 채널 (in-dist, 5-fold 평균)")
    print(f"  {'world':<12}{'tier':<10}{'quality raw→cons':<22}{'risk raw→cons':<20}")
    for wname in worlds:
        eds = ext[wname]
        for tier in ["fast", "balanced"]:
            res = {}
            for cons in [False, True]:
                qs, rs = [], []
                for f in range(CFG["eval"]["k_folds"]):
                    test_idx, _, _, b_te, pol = build_v2(eds, wname, f, folds, tier, cons)
                    a = audit_tier(eds, test_idx, pol, b_te)
                    qs.append(a["quality"]); rs.append(a["risk"])
                res[cons] = (np.mean(qs), np.mean(rs))
            print(f"  {wname:<12}{tier:<10}{res[False][0]:.4f}→{res[True][0]:.4f}"
                  f"          {res[False][1]:.4f}→{res[True][1]:.4f}")


def stage_5a(alpha=0.05):
    worlds, ext, folds = ext_worlds()
    report = {}
    for wname in worlds:
        eds = ext[wname]
        rows = []
        for f in range(CFG["eval"]["k_folds"]):
            for tier in ["fast", "balanced"]:
                test_idx, tr_cal, b_cal, b_te, factory = build(eds, wname, f, folds, tier)
                tau, curve = crc_tau(eds, tr_cal, factory, b_cal, alpha)
                mono = all(curve[i][1] >= curve[i + 1][1] - 0.01
                           for i in range(len(curve) - 1))
                r0 = audit_tier(eds, test_idx, factory(0.0), b_te)
                rc = audit_tier(eds, test_idx, factory(tau), b_te)
                rows.append({"fold": f, "tier": tier, "tau": tau, "mono": mono,
                             "risk0": round(r0["risk"], 4), "riskC": round(rc["risk"], 4),
                             "q0": round(r0["quality"], 4), "qC": round(rc["quality"], 4)})
        report[wname] = rows
    json.dump(report, open(OUT / "phase5a.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    for wname, rows in report.items():
        print(f"\n[5a | {wname}]  (alpha={alpha}, 5-fold)")
        print(f"  {'tier':<10}{'tau(중앙값)':<12}{'단조성':<8}{'test risk: tau=0 → CRC':<26}"
              f"{'quality: tau=0 → CRC':<24}")
        for tier in ["fast", "balanced"]:
            sub = [r for r in rows if r["tier"] == tier]
            taus = sorted(r["tau"] for r in sub)
            mono = all(r["mono"] for r in sub)
            r0 = np.mean([r["risk0"] for r in sub]); rC = np.mean([r["riskC"] for r in sub])
            q0 = np.mean([r["q0"] for r in sub]); qC = np.mean([r["qC"] for r in sub])
            ok = "OK" if rC <= alpha + 0.01 else "VIOL"
            print(f"  {tier:<10}{taus[len(taus)//2]:<12}{'OK' if mono else 'X':<8}"
                  f"{r0:.4f} → {rC:.4f} [{ok}]      {q0:.4f} → {qC:.4f}")


def _shift_test(eds, test_idx, kind, rng):
    """분포 이동 시나리오: 검증기 잡음 ×2 / 도메인 편중 / (LODO는 build 단계에서)"""
    if kind == "noise2x":
        ver = eds.verifier.copy()
        ver[test_idx] = np.clip(eds.quality[test_idx]
                                + rng.normal(0, 0.30, size=(len(test_idx), eds.m)), 0, 1)
        return test_idx, ver
    if kind == "domain-skew":                    # 최고난도 도메인 70% 편중 재표집
        hard_dom = 3
        hard = test_idx[eds.domains[test_idx] == hard_dom]
        rest = test_idx[eds.domains[test_idx] != hard_dom]
        k = min(len(hard), int(0.7 * len(test_idx)))
        pick = np.concatenate([rng.choice(hard, size=k, replace=True),
                               rng.choice(rest, size=len(test_idx) - k, replace=False)])
        return pick, None
    raise ValueError(kind)


def stage_5b():
    """분포 이동 스트레스: raw vs 보수화(M4 v2) 관측 채널."""
    worlds, ext, folds = ext_worlds()
    rng = np.random.default_rng(55)
    print(f"\n[5b] 분포 이동 스트레스 (balanced tier, 5-fold 평균)")
    print(f"  {'world':<12}{'shift':<14}{'quality raw→cons':<22}{'risk raw→cons':<20}")
    for wname in worlds:
        eds = ext[wname]
        for kind in ["noise2x", "domain-skew"]:
            res = {}
            for cons in [False, True]:
                qs, rs = [], []
                for f in range(CFG["eval"]["k_folds"]):
                    test_idx, _, _, b_te, pol = build_v2(eds, wname, f, folds, "balanced", cons)
                    sh_idx, sh_ver = _shift_test(eds, test_idx, kind, rng)
                    b_sh = tier_budgets(eds, sh_idx, CFG)["balanced"]
                    a = audit_tier(eds, sh_idx, pol, b_sh, sh_ver)
                    qs.append(a["quality"]); rs.append(a["risk"])
                res[cons] = (np.mean(qs), np.mean(rs))
            print(f"  {wname:<12}{kind:<14}{res[False][0]:.4f}→{res[True][0]:.4f}"
                  f"          {res[False][1]:.4f}→{res[True][1]:.4f}")


def stage_5c():
    """풀스택 종합 영향: 보수화 on/off의 종합 점수 (in-dist) — 보험료 측정."""
    worlds, ext, folds = ext_worlds()
    W = {t: CFG["tiers"][t]["weight"] for t in CFG["tiers"]}
    print(f"\n[5c] 보수화 채널의 종합 점수 영향 (in-dist, 5-fold)")
    for wname in worlds:
        eds = ext[wname]
        comb = {False: 0.0, True: 0.0}
        for tier in ["fast", "balanced", "premium"]:
            for cons in [False, True]:
                qs = []
                for f in range(CFG["eval"]["k_folds"]):
                    test_idx, _, _, b_te, pol = build_v2(eds, wname, f, folds, tier, cons)
                    qs.append(audit_tier(eds, test_idx, pol, b_te)["quality"])
                comb[cons] += W[tier] * np.mean(qs)
        print(f"  {wname:<12} 종합점수: raw {comb[False]:.4f}  →  cons {comb[True]:.4f}"
              f"  (보험료 {comb[False] - comb[True]:+.4f})")


def stage_5d(conf=0.95):
    """M4 v3 (최종형): conformal '인증서' — 정책 무개입 통계 보증.

    5a(궤적 CRC)·5b(보수화 채널)의 부정적 결과에 따라, M4의 역할을 '제어'에서
    '인증'으로 재정의: 최종 정책(raw)의 보정셋 조기정지 후회율에 Clopper–Pearson
    상한(신뢰 conf)을 부여해 심사에 제시한다. 검증: test 후회율이 인증 상한을
    fold의 ≥conf 비율로 준수하는가 (분포 내 유효성).
    """
    from scipy.stats import beta as beta_dist
    worlds, ext, folds = ext_worlds()
    print(f"\n[5d] Conformal 인증서 (conf={conf}, 5-fold)")
    print(f"  {'world':<12}{'tier':<10}{'인증 상한(평균)':<16}{'test risk(평균)':<16}"
          f"{'준수 fold':<10}")
    for wname in worlds:
        eds = ext[wname]
        for tier in ["fast", "balanced"]:
            uppers, risks, ok = [], [], 0
            for f in range(CFG["eval"]["k_folds"]):
                test_idx, tr_cal, b_cal, b_te, pol = build_v2(eds, wname, f, folds,
                                                              tier, False)
                cal = audit_tier(eds, tr_cal, pol, b_cal)
                n = len(tr_cal)
                k = int(round(cal["risk"] * n))
                upper = float(beta_dist.ppf(conf, k + 1, n - k))    # CP 상한
                te = audit_tier(eds, test_idx, pol, b_te)
                uppers.append(upper); risks.append(te["risk"])
                ok += te["risk"] <= upper
            print(f"  {wname:<12}{tier:<10}{np.mean(uppers):<16.4f}"
                  f"{np.mean(risks):<16.4f}{ok}/{CFG['eval']['k_folds']:<10}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "5a"
    {"5a": stage_5a, "5a2": stage_5a2, "5b": stage_5b, "5c": stage_5c,
     "5d": stage_5d}[stage]()


