"""예약값 일반해의 회귀 테스트 (B4).

Weitzman 방정식 E[(Q−σ)⁺] = λc 의 해가 상금 분포 종류와 무관하게 정확한지를 고정한다.
검증 축 4개:
  ① 정의 충족   — 구한 σ를 (★)에 대입하면 잔차 0 (수치적분 대조 포함)
  ② 특수화 일치 — 일반해가 Bernoulli 닫힌형과 정확히 같고, Beta는 κ→0에서 Bernoulli
  ③ 최적성      — 이산 상금 DP 전역 최적해와 엔진 기대가치가 일치
  ④ 결함 회귀   — λc > p̄ (음수 σ) 구간에서 구 닫힌형이 틀렸음을 못 박음
"""
import sys, pathlib
from functools import lru_cache
from itertools import product

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

# numpy 2.0에서 np.trapz → np.trapezoid 로 개명됐다. 두 버전 모두에서 돌아야 한다.
_trapz = getattr(np, "trapezoid", None) or np.trapz

from src.engine.prize import (bernoulli_reservation, beta_mixture_excess,
                              beta_reservation, discrete_excess,
                              discrete_reservation, solve_reservation)
from src.engine.pandora import PandoraPolicy
from src.harness import Session


# ---------- ① 정의 충족: σ를 (★)에 되대입하면 잔차 0 ----------

def test_bernoulli_reservation_satisfies_defining_equation():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.99, size=500)
    t = rng.uniform(0.0, 3.0, size=500)          # λc — p̄를 크게 넘는 구간 포함
    s = bernoulli_reservation(p, t)
    excess = np.where(s >= 0.0, p * (1.0 - s), p - s)      # E[(Q−σ)⁺], Q∈{0,1}
    assert np.allclose(excess, t, atol=1e-12)


def test_beta_excess_matches_survival_integral():
    """닫힌형(정칙 불완전베타 1차 적률)을 **독립 경로**인 생존함수 적분과 대조.

    부분적분 항등식  E[(Q−σ)⁺] = ∫_σ^1 P(Q>q) dq  를 쓴다. 밀도 적분과 달리
    피적분함수가 [0,1]로 유계라 (κ가 작아 α<1 또는 β<1 이어서 밀도가 경계에서
    발산하는 경우에도) 균등격자 사다리꼴이 잘 수렴한다.
    """
    rng = np.random.default_rng(1)
    mu = rng.uniform(0.1, 0.9, size=(1, 6))
    kappa = rng.uniform(0.5, 20.0, size=6)
    w = np.ones(1)
    from scipy.stats import beta as beta_dist
    for s0 in (0.05, 0.3, 0.5, 0.85):
        exc, surv = beta_mixture_excess(np.full(6, s0), mu, kappa, w)
        grid = np.linspace(s0, 1.0, 200_001)
        for m in range(6):
            a, b = kappa[m] * mu[0, m], kappa[m] * (1 - mu[0, m])
            num = _trapz(beta_dist.sf(grid, a, b), grid)
            assert exc[m] == pytest.approx(num, abs=1e-6), f"m={m} σ={s0}"
            assert surv[m] == pytest.approx(beta_dist.sf(s0, a, b), abs=1e-9)


def test_beta_reservation_satisfies_defining_equation():
    rng = np.random.default_rng(2)
    K, M = 7, 9
    mu = rng.uniform(0.05, 0.95, size=(K, M))
    kappa = rng.uniform(0.3, 25.0, size=M)
    w = rng.dirichlet(np.ones(K))
    mean = (w[:, None] * mu).sum(axis=0)
    for scale in (0.05, 0.3, 0.9, 2.0):           # 선형 구간(t>E[Q])까지 훑는다
        t = mean * scale
        s = beta_reservation(t, mu, kappa, w)
        exc, _ = beta_mixture_excess(np.clip(s, 0.0, 1.0), mu, kappa, w)
        exc = np.where(s < 0.0, mean - s, exc)    # σ<0 선형 구간
        assert np.allclose(exc, t, atol=1e-9), f"scale={scale}"


def test_reservation_is_monotone_decreasing_in_cost():
    rng = np.random.default_rng(3)
    mu = rng.uniform(0.05, 0.95, size=(5, 8))
    kappa = rng.uniform(0.5, 15.0, size=8)
    w = rng.dirichlet(np.ones(5))
    prev = None
    for t_scale in np.linspace(0.01, 2.5, 25):
        s = beta_reservation(np.full(8, t_scale), mu, kappa, w)
        if prev is not None:
            assert (s <= prev + 1e-12).all(), "σ는 λc에 대해 비증가여야 한다"
        prev = s


# ---------- ② 특수화: 일반해 ⊃ Bernoulli, Beta(κ→0) → Bernoulli ----------

def test_general_solver_reproduces_bernoulli_closed_form():
    rng = np.random.default_rng(4)
    p = rng.uniform(0.05, 0.95, size=200)
    t = rng.uniform(0.0, 2.0, size=200)
    values = np.array([0.0, 1.0])
    probs = np.vstack([1.0 - p, p])               # (2, M) 이산 지지 = Bernoulli
    assert np.allclose(discrete_reservation(t, values, probs),
                       bernoulli_reservation(p, t), atol=1e-9)


def test_beta_converges_to_bernoulli_as_kappa_to_zero():
    """Beta(κμ, κ(1−μ)) --κ→0--> Bernoulli(μ): 모형이 이진을 '포함'함의 실증."""
    rng = np.random.default_rng(5)
    M = 6
    mu = rng.uniform(0.15, 0.85, size=(1, M))
    w = np.ones(1)
    t = rng.uniform(0.05, 0.6, size=M)
    target_bern = bernoulli_reservation(mu[0], t)
    errs = []
    for kappa_val in (1.0, 0.1, 0.01, 1e-3, 1e-4):
        s = beta_reservation(t, mu, np.full(M, kappa_val), w)
        errs.append(float(np.abs(s - target_bern).max()))
    assert errs[-1] < 1e-3, f"κ→0 수렴 실패: {errs}"
    assert errs == sorted(errs, reverse=True), f"단조 수렴 아님: {errs}"


# ---------- ③ 최적성: 이산 상금 DP 전역 최적 대조 ----------

def _dp_value_discrete(values, probs, c, lam):
    """연속(이산 지지) 상금의 전역 최적 기대 순가치. values는 오름차순, values[0]=0.

    probs·c는 lru_cache 해시를 위해 튜플로 받는다 (probs[l][m]).
    """
    L, M = len(probs), len(probs[0])

    @lru_cache(maxsize=None)
    def V(S, best):
        vals = [values[best]]
        for m in range(M):
            if not (S >> m) & 1:
                S2 = S | (1 << m)
                vals.append(-lam * c[m]
                            + sum(probs[l][m] * V(S2, max(best, l)) for l in range(L)))
        return max(vals)

    return max(-lam * c[m] + sum(probs[l][m] * V(1 << m, l) for l in range(L))
               for m in range(M))


class _DiscreteBelief:
    """이산 지지 상금 신념 + 정확 관측 (고전 Pandora 가정)."""

    def __init__(self, values, probs):
        self.values, self.probs = values, probs

    def reservation(self, target):
        return discrete_reservation(target, self.values, self.probs)

    def update(self, m, v):
        pass

    def prize(self, v, m=None):
        return float(v)


def _engine_value_discrete(values, probs, c, lam):
    """전 상금 조합 열거 × 결정적 리플레이 → 엔진의 기대 순가치."""
    L, M = probs.shape
    ev = 0.0
    for combo in product(range(L), repeat=M):
        prob = float(np.prod([probs[combo[m], m] for m in range(M)]))
        if prob == 0.0:
            continue
        y = np.array([values[combo[m]] for m in range(M)], dtype=float)
        pol = PandoraPolicy(lambda x, d: _DiscreteBelief(values, probs), lam)
        sess = Session(costs=np.array(c), verifier_row=y, remaining_budget=1e9)
        choice = pol.route(sess, None, 0)
        ev += prob * (y[choice] - lam * sess.spent)
    return ev


def test_engine_matches_dp_on_continuous_prizes():
    """★ B4의 핵심 주장: 연속(다수준) 상금에서도 지수 정책이 전역 최적."""
    rng = np.random.default_rng(6)
    values = np.array([0.0, 0.35, 0.7, 1.0])          # 부분점수 4수준
    gaps = []
    for _ in range(120):
        M = int(rng.integers(2, 4))
        probs = rng.dirichlet(np.ones(len(values)), size=M).T      # (L, M)
        c = rng.uniform(0.02, 0.45, size=M)
        lam = float(rng.uniform(0.3, 4.0))                          # 음수 σ 구간 포함
        gaps.append(_dp_value_discrete(values, tuple(map(tuple, probs)),
                                       tuple(c), lam)
                    - _engine_value_discrete(values, probs, c, lam))
    gaps = np.array(gaps)
    assert np.abs(gaps).max() < 1e-9, f"DP 격차 max={np.abs(gaps).max():.3e}"


# ---------- ④ 결함 회귀: 구 닫힌형이 틀렸던 구간을 못 박는다 ----------

def test_old_closed_form_was_wrong_below_zero_and_is_now_fixed():
    """σ<0 구간에서 σ=1−λc/p̄ 는 참값이 아니며 순서를 뒤집는다 (probe 재현)."""
    p = np.array([0.3262, 0.0793])
    t = np.array([4.4958, 1.7975])                    # λc > p̄ — 전 상자 음수 σ
    old = 1.0 - t / p                                 # 구 닫힌형
    new = bernoulli_reservation(p, t)                 # 일반해
    assert np.allclose(new, p - t)                    # 참값 = p̄ − λc
    assert int(np.argmax(old)) != int(np.argmax(new))  # 순서가 실제로 갈린다
    # 참값의 순서가 곧 순가치 순서 (강제 1회 개봉 후 정지 체제)
    assert int(np.argmax(new)) == int(np.argmax(p - t))


# ---------- ⑤ 연속 신념 모형: 모수 복원 (Phase 2의 이진 IRT 검증과 같은 기준) ----------

def test_beta_mml_recovers_generating_model():
    """알려진 Beta 반응모형에서 생성한 데이터로 모형을 되찾는가.

    **판정 대상은 원시 모수가 아니라 정책이 실제로 쓰는 양이다.** 판별도 a는
    b가 θ 분포의 꼬리에 있는 모델에서 약하게만 식별된다 — μ(θ)가 포화해 기울기
    정보가 데이터에 거의 없기 때문이며(이진 IRT에서 극단 난이도 문항의 변별도가
    부정확한 것과 같은 현상), 우도는 완전히 수렴한 상태에서도 축소 추정된다
    (실측 a: 2.4 → 1.62). 그러나 라우팅이 소비하는 것은 a 자체가 아니라
      ① 모델 서열을 정하는 b,  ② 평균함수 μ(θ),  ③ 거기서 유도되는 예약값 σ
    이므로 이 셋으로 판정한다.
    """
    from src.irt.beta_mml import fit_beta_mml
    rng = np.random.default_rng(7)
    N = 1500
    a_true = np.array([0.8, 1.2, 1.6, 2.0, 2.4])
    b_true = np.array([1.1, 0.5, -0.1, -0.8, -1.3])
    k_true = np.array([2.0, 3.0, 4.0, 6.0, 8.0])
    theta = rng.standard_normal(N)
    f_mu = lambda a, b, th: 1.0 / (1.0 + np.exp(-a[None, :] * (th[:, None] - b[None, :])))
    q = rng.beta(k_true[None, :] * f_mu(a_true, b_true, theta),
                 k_true[None, :] * (1.0 - f_mu(a_true, b_true, theta)))
    fit = fit_beta_mml(q, np.zeros(N, dtype=int), 1, per_domain_b=False, steps=1500)

    # ① 서열 파라미터 b — 어느 모델이 어느 난이도에서 강한가
    assert np.corrcoef(fit["b"][0], b_true)[0, 1] > 0.99, f"b 복원 실패: {fit['b'][0]}"
    assert np.corrcoef(fit["kappa"], k_true)[0, 1] > 0.95, f"κ 복원 실패: {fit['kappa']}"
    assert fit["curve"][-1] > fit["curve"][0], "우도가 증가하지 않음"

    # ② 평균함수 — θ 분포의 본체에서 예측 품질이 일치하는가.
    #    잔여 오차는 위 a 축소에서 오며 포화 구간에 몰린다 (실측 max 0.086).
    zs = np.linspace(-2.0, 2.0, 41)
    d_mu = np.abs(f_mu(fit["a"], fit["b"][0], zs) - f_mu(a_true, b_true, zs)).max()
    assert d_mu < 0.10, f"평균함수 이탈 {d_mu:.4f}"

    # ③ 예약값 — 정책이 실제로 비교하는 값 (서열까지 일치해야 한다)
    Z = np.linspace(-4, 4, 41)
    w = np.exp(-0.5 * Z ** 2)
    w /= w.sum()
    t = np.array([0.02, 0.05, 0.1, 0.2, 0.4])
    s_fit = beta_reservation(t, f_mu(fit["a"], fit["b"][0], Z), fit["kappa"], w)
    s_true = beta_reservation(t, f_mu(a_true, b_true, Z), k_true, w)
    assert np.abs(s_fit - s_true).max() < 0.06, f"예약값 이탈 {np.abs(s_fit - s_true).max():.4f}"
    assert int(np.argmax(s_fit)) == int(np.argmax(s_true)), "최우선 상자 불일치"
    # 서열은 **차이가 유의미한 쌍**에서만 물어야 한다. 참 σ 격차가 추정오차 이하인
    # 쌍은 어느 쪽을 열어도 기대가치가 사실상 같으므로 순서가 뒤바뀌어도 무해하다.
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            if abs(s_true[i] - s_true[j]) > 0.06:
                assert np.sign(s_fit[i] - s_fit[j]) == np.sign(s_true[i] - s_true[j]), \
                    f"유의미한 서열 반전: 상자 {i} vs {j}"


def test_beta_belief_reservation_wired_into_engine():
    """GridBelief(kappa=...)가 Beta 예약값 경로를 실제로 타는지 (배선 확인)."""
    from src.engine.pandora import GridBelief, NoiseModel
    rng = np.random.default_rng(8)
    a = np.array([1.0, 1.5, 2.0])
    b = np.array([0.5, 0.0, -0.5])
    noise = NoiseModel().fit(rng.uniform(size=400), (rng.uniform(size=400) > 0.5).astype(float))
    t = np.array([0.05, 0.08, 0.2])
    bern = GridBelief(0.0, 1.0, a, b, noise, kappa=None).reservation(t)
    beta = GridBelief(0.0, 1.0, a, b, noise, kappa=np.full(3, 3.0)).reservation(t)
    assert not np.allclose(bern, beta), "kappa를 줘도 Bernoulli 경로를 타고 있음"
    tiny = GridBelief(0.0, 1.0, a, b, noise, kappa=np.full(3, 1e-4)).reservation(t)
    assert np.allclose(tiny, bern, atol=1e-3), "κ→0 이 Bernoulli로 수렴하지 않음"


def test_solver_handles_degenerate_targets():
    mu = np.full((1, 3), 0.5)
    kappa = np.full(3, 4.0)
    w = np.ones(1)
    s0 = beta_reservation(np.zeros(3), mu, kappa, w)          # 비용 0 → 무조건 개봉
    assert np.allclose(s0, 1.0)
    s_big = beta_reservation(np.full(3, 50.0), mu, kappa, w)  # 비용 폭발 → 깊은 음수
    assert np.allclose(s_big, 0.5 - 50.0)
