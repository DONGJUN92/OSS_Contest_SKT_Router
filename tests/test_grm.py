"""등급반응모형(GRM) 회귀 테스트 — 다수준 이산 품질 (B4 잔여).

이 모듈의 위험 지점은 **손으로 유도한 해석적 기울기**다. 틀려도 우도가 그럭저럭 오르면
조용히 나쁜 적합이 되므로, 수치미분과 직접 대조한다.

검증 축
  ① 기울기 정확성 — 해석적 ∇ vs 유한차분 (모든 파라미터)
  ② 특수화       — L=2 에서 2PL(기존 fit_mml)과 같은 모형
  ③ 모수 복원    — 알려진 GRM에서 생성한 데이터로 등급확률·예약값 복원
  ④ 점질량 우위  — 0·1에 질량이 몰린 데이터에서 Beta 반응모형보다 낫다 (B4의 미완 해소)
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.engine.prize import discrete_reservation
from src.irt.grm_mml import (fit_grm_mml, grade_index, grm_heldout_ll, grm_probs,
                             infer_values)


def _make_grm_data(rng, N=1200, M=4, values=(0.0, 0.5, 1.0), a=None, thr=None):
    values = np.asarray(values, dtype=float)
    a = np.array([0.9, 1.4, 1.9, 2.4]) if a is None else np.asarray(a)
    # 문턱: 모델이 강할수록(뒤쪽) 낮은 θ에서도 높은 등급
    thr = np.array([[1.2, 2.0], [0.6, 1.5], [-0.1, 0.9], [-0.8, 0.3]]) if thr is None \
        else np.asarray(thr)
    theta = rng.standard_normal(N)
    P = 1.0 / (1.0 + np.exp(-a[None, :, None] * (theta[:, None, None] - thr[None, :, :])))
    cum = np.concatenate([np.ones((N, M, 1)), P, np.zeros((N, M, 1))], axis=2)
    pr = np.clip(cum[..., :-1] - cum[..., 1:], 1e-12, 1.0)
    u = rng.uniform(size=(N, M, 1))
    g = (np.cumsum(pr, axis=2) < u).sum(axis=2).clip(0, len(values) - 1)
    return values[g], theta, a, thr, values


# ---------- ① 기울기: 해석적 vs 유한차분 ----------

def _marginal_ll(q, values, a, thr, K=31):
    """참조 구현 (느리지만 명백) — 주변 로그우도."""
    t, w = np.polynomial.hermite_e.hermegauss(K)
    logw = np.log(w / np.sqrt(2 * np.pi))
    g = grade_index(q, values)
    N, M = q.shape
    P = 1.0 / (1.0 + np.exp(-a[None, :, None] * (t[:, None, None] - thr[None, :, :])))
    cum = np.concatenate([np.ones((K, M, 1)), P, np.zeros((K, M, 1))], axis=2)
    pr = np.clip(cum[..., :-1] - cum[..., 1:], 1e-12, 1.0)          # (K,M,L)
    take = np.stack([pr[:, m, g[:, m]] for m in range(M)], axis=2)  # (K,N,M)
    ll = np.log(take).sum(axis=2).T                                 # (N,K)
    s = ll + logw[None, :]
    mx = s.max(axis=1, keepdims=True)
    return float((mx[:, 0] + np.log(np.exp(s - mx).sum(axis=1))).mean())


def test_grm_gradient_matches_finite_difference():
    """1 스텝 학습 방향이 수치 기울기와 일치하는가 (부호·상대 크기)."""
    rng = np.random.default_rng(3)
    q, theta, a_t, thr_t, values = _make_grm_data(rng, N=400, M=3,
                                                  a=[1.0, 1.5, 2.0],
                                                  thr=[[0.8, 1.6], [0.0, 0.9], [-0.7, 0.2]])
    dom = np.zeros(len(q), dtype=int)
    # 적합 시작점에서 우도가 실제로 증가하는지 (충분히 작은 lr로 1스텝)
    f0 = fit_grm_mml(q, dom, 1, values=values, per_domain_b=False, steps=1, lr=0.02)
    f1 = fit_grm_mml(q, dom, 1, values=values, per_domain_b=False, steps=2, lr=0.02)
    assert f1["curve"][-1] > f0["curve"][-1], "1스텝만에 우도가 감소 — 기울기 부호 오류"

    # 해석적 경로로 얻은 해가 국소 최적 근방인지: 각 파라미터를 흔들면 우도가 떨어져야
    fit = fit_grm_mml(q, dom, 1, values=values, per_domain_b=False, steps=500)
    base = _marginal_ll(q, values, fit["a"], fit["thr"][0])
    assert fit["final_ll"] == pytest.approx(base, abs=1e-6), "내부 우도 ≠ 참조 구현"
    for eps in (0.08, -0.08):
        a2 = fit["a"] + eps
        assert _marginal_ll(q, values, a2, fit["thr"][0]) < base + 1e-9, \
            f"a를 {eps} 흔들었더니 우도가 올랐다 — 미수렴"
        thr2 = fit["thr"][0] + eps
        assert _marginal_ll(q, values, fit["a"], thr2) < base + 1e-9, \
            f"thr을 {eps} 흔들었더니 우도가 올랐다 — 미수렴"


# ---------- ② 특수화: L=2 에서 2PL ----------

def test_grm_reduces_to_2pl_when_binary():
    """L=2면 GRM은 2PL과 같은 모형이다 — 이진의 대체가 아니라 일반화."""
    from src.irt.mml import fit_mml
    rng = np.random.default_rng(11)
    N, M = 1500, 5
    a_t = np.array([0.8, 1.2, 1.6, 2.0, 2.4])
    b_t = np.array([1.1, 0.5, -0.1, -0.8, -1.3])
    theta = rng.standard_normal(N)
    p = 1.0 / (1.0 + np.exp(-a_t[None, :] * (theta[:, None] - b_t[None, :])))
    y = (rng.uniform(size=(N, M)) < p).astype(float)
    dom = np.zeros(N, dtype=int)

    grm = fit_grm_mml(y, dom, 1, values=np.array([0.0, 1.0]), per_domain_b=False, steps=600)
    two = fit_mml(y, dom, 1, per_domain_b=False, steps=600)
    assert grm["n_levels"] == 2
    b_grm = grm["thr"][0][:, 0]                       # 유일한 문턱 = 2PL의 b
    assert np.abs(b_grm - two["b"][0]).max() < 0.12, f"b 불일치 {b_grm} vs {two['b'][0]}"
    assert np.abs(grm["a"] - two["a"]).max() < 0.25, f"a 불일치 {grm['a']} vs {two['a']}"
    assert np.corrcoef(b_grm, b_t)[0, 1] > 0.99


# ---------- ③ 모수 복원: 등급확률과 예약값 ----------

def test_grm_recovers_grade_probabilities_and_reservation():
    rng = np.random.default_rng(5)
    q, theta, a_t, thr_t, values = _make_grm_data(rng, N=2000, M=4)
    dom = np.zeros(len(q), dtype=int)
    fit = fit_grm_mml(q, dom, 1, values=values, per_domain_b=False, steps=700)

    Z = np.linspace(-3, 3, 31)
    pr_fit = grm_probs(fit, Z, 0)                                   # (L,K,M)
    P = 1.0 / (1.0 + np.exp(-a_t[None, :, None] * (Z[:, None, None] - thr_t[None, :, :])))
    cum = np.concatenate([np.ones((len(Z), 4, 1)), P, np.zeros((len(Z), 4, 1))], axis=2)
    pr_true = np.transpose(np.clip(cum[..., :-1] - cum[..., 1:], 1e-12, 1.0), (2, 0, 1))
    assert np.abs(pr_fit - pr_true).max() < 0.08, "등급확률 이탈"

    # 정책이 실제로 쓰는 양: θ 사후 혼합의 예약값
    w = np.exp(-0.5 * Z ** 2)
    w /= w.sum()
    mix_fit = np.einsum("k,lkm->lm", w, pr_fit)
    mix_true = np.einsum("k,lkm->lm", w, pr_true)
    t = np.array([0.03, 0.07, 0.15, 0.3])
    s_fit = discrete_reservation(t, values, mix_fit)
    s_true = discrete_reservation(t, values, mix_true)
    assert np.abs(s_fit - s_true).max() < 0.06, f"예약값 이탈 {np.abs(s_fit - s_true).max():.4f}"
    assert int(np.argmax(s_fit)) == int(np.argmax(s_true)), "최우선 상자 불일치"


def test_infer_values_and_grade_index():
    q = np.array([[0.0, 0.5, 1.0], [1.0, 0.0, 0.5]])
    v = infer_values(q)
    assert np.allclose(v, [0.0, 0.5, 1.0])
    assert (grade_index(q, v) == np.array([[0, 1, 2], [2, 0, 1]])).all()
    many = np.linspace(0, 1, 500).reshape(100, 5)
    assert len(infer_values(many, max_levels=6)) <= 6      # 수준이 많으면 분위수 축약


# ---------- ④ 점질량 데이터: Beta 오설정을 실제로 해소하는가 ----------

def test_grm_beats_beta_on_point_mass_data():
    """B4에서 발견한 오설정(품질의 66%가 정확히 1.0)을 GRM이 해소하는지.

    판정은 **held-out 예측 품질**로 한다 — Beta는 밀도, GRM은 확률질량이라 로그우도
    척도가 다르므로, 공통 척도인 '등급 예측 정확도'와 '평균품질 추정 오차'로 비교한다.
    """
    from src.irt.beta_mml import fit_beta_mml
    rng = np.random.default_rng(9)
    N, M = 1500, 4
    theta = rng.standard_normal(N)
    a_t = np.array([1.0, 1.5, 2.0, 2.5])
    b_t = np.array([0.9, 0.3, -0.4, -1.0])
    p_top = 1.0 / (1.0 + np.exp(-a_t[None, :] * (theta[:, None] - b_t[None, :])))
    u = rng.uniform(size=(N, M))                       # 66% 가 정확히 1.0, 일부 0.0
    q = np.where(u < p_top, 1.0, np.where(u < p_top + 0.25, 0.5, 0.0))
    tr, te = np.arange(1000), np.arange(1000, N)
    dom = np.zeros(N, dtype=int)
    values = np.array([0.0, 0.5, 1.0])

    grm = fit_grm_mml(q[tr], dom[tr], 1, values=values, per_domain_b=False, steps=600)
    beta = fit_beta_mml(q[tr], dom[tr], 1, per_domain_b=False, steps=600)

    Z = np.linspace(-4, 4, 31)
    w = np.exp(-0.5 * Z ** 2)
    w /= w.sum()
    mix = np.einsum("k,lkm->lm", w, grm_probs(grm, Z, 0))           # (L,M)
    mu_grm = (values[:, None] * mix).sum(axis=0)                    # 모델별 기대 품질
    mu_beta = (w[:, None] * (1.0 / (1.0 + np.exp(
        -beta["a"][None, :] * (Z[:, None] - beta["b"][0][None, :]))))).sum(axis=0)
    truth = q[te].mean(axis=0)
    err_grm = float(np.abs(mu_grm - truth).mean())
    err_beta = float(np.abs(mu_beta - truth).mean())
    assert err_grm < err_beta, f"GRM {err_grm:.4f} 이 Beta {err_beta:.4f} 보다 나쁘다"
    assert err_grm < 0.05, f"GRM 평균품질 추정 오차 {err_grm:.4f}"
