"""상금 분포 → 예약값 σ: Weitzman 방정식의 일반해 (B4 — PROJECT_PLAN §9 P1).

Weitzman(1979)의 예약값은 상금 분포 Q와 비용 c에 대해

        E[(Q − σ)⁺] = λ·c                                            (★)

의 유일해다. **지수 정책의 최적성 증명은 Q의 분포 형태에 의존하지 않는다** — 따라서
Q4(품질이 이진인가 연속인가)가 어느 쪽으로 확정돼도 바뀌는 것은 (★)를 푸는 이 모듈
하나이며, 엔진·페이싱·IRT·인증서·하네스는 무변경이다.

(★)의 구조 (Q ∈ [q_lo, q_hi] 유계):
  - E[(Q−σ)⁺]는 σ에 대해 **볼록·비증가**, 도함수는 −P(Q>σ) = −S(σ)
  - σ ≤ q_lo 구간에서는 (Q−σ)⁺ = Q−σ 이므로 **선형**: E = E[Q] − σ
  - σ ≥ q_hi 에서 0
따라서 해는 세 구간으로 나뉘고, 가운데 구간만 수치해가 필요하다:
    t ≤ 0            → σ = q_hi                    (열 이유가 없는 무비용 상자)
    t ≥ E[Q] − q_lo  → σ = E[Q] − t                (선형 구간 — **닫힌형**)
    그 외            → [q_lo, q_hi]에서 안전장치 부착 Newton

★ 이 선형 구간이 기존 닫힌형이 놓친 부분이다. Bernoulli에서 현행 코드는 t > p̄ 일 때도
σ = 1 − t/p̄ 를 쓰는데 참값은 σ = p̄ − t 이고, 둘은 상자 간 **순서를 다르게 매긴다**
(eval/probe_negative_sigma.py: DP 최적해 대비 격차 최대 2.45). 아래 함수들은 세 구간을
모두 정확히 처리하므로 이진·연속 양쪽에서 결함이 소멸한다.
"""
import numpy as np
from scipy.special import betainc

__all__ = ["bernoulli_reservation", "beta_mixture_excess", "discrete_excess",
           "solve_reservation", "beta_reservation", "discrete_reservation",
           "perbox_excess", "perbox_reservation"]


# ---------------- Bernoulli (Q4=이진) — 양 구간 닫힌형 ----------------

def bernoulli_reservation(pbar, target):
    """Q ∈ {0,1}, P(Q=1)=p̄ 의 예약값. 벡터화·분기 없음.

      σ ≥ 0 구간 : E[(Q−σ)⁺] = p̄(1−σ) = t  →  σ = 1 − t/p̄     (t ≤ p̄)
      σ < 0 구간 : E[(Q−σ)⁺] = p̄ − σ   = t  →  σ = p̄ − t       (t > p̄)

    두 식은 t = p̄ 에서 σ = 0 으로 연속이며, 아래 일반해 solve_reservation의
    Bernoulli 특수화와 정확히 일치한다 (tests/test_prize.py에서 고정).
    """
    p = np.maximum(np.asarray(pbar, dtype=float), 1e-12)
    t = np.asarray(target, dtype=float)
    return np.where(t <= p, 1.0 - t / p, p - t)


# ---------------- Beta 혼합 (Q4=연속) ----------------

def _beta_params(mu, kappa):
    """(α, β) = (κμ, κ(1−μ)) — σ에 무관하므로 Newton 반복 바깥에서 1회만 만든다."""
    a = np.maximum(kappa[None, :] * mu, 1e-8)                            # (K,M)
    b = np.maximum(kappa[None, :] * (1.0 - mu), 1e-8)
    return a, b


def _make_excess_fn(mu, kappa, w):
    """σ에 무관한 항을 전부 선계산한 excess 평가기를 만든다.

    닫힌형 근거 — 정칙 불완전베타 I_x(a,b) 의 1차 적률 항등식
        ∫_σ^1 q·f(q;α,β) dq = μ·(1 − I_σ(α+1, β)),   μ = α/(α+β)
    따라서  E[(Q−σ)⁺] = μ(1 − I_σ(α+1,β)) − σ(1 − I_σ(α,β)).
    두 불완전베타의 파라미터를 한 배열로 미리 쌓아두고 **호출당 ufunc 1회**로 끝낸다
    (이 크기의 배열에서는 호출·할당 오버헤드가 원소 수보다 지배적이다).
    """
    K, M = mu.shape
    a, b = _beta_params(mu, kappa)
    A = np.concatenate([a, a + 1.0])                 # (2K, M) — 반복 바깥에서 1회
    B = np.concatenate([b, b])
    wk = w[:, None]

    def excess_surv(sigma):
        sig = np.clip(sigma, 0.0, 1.0)[None, :]      # (1,M) — betainc가 브로드캐스트
        big = betainc(A, B, sig)
        surv = 1.0 - big[:K]
        exc = mu * (1.0 - big[K:]) - sig * surv
        return (wk * exc).sum(axis=0), (wk * surv).sum(axis=0)

    return excess_surv


def beta_mixture_excess(sigma, mu, kappa, w):
    """Q ~ Σ_k w_k Beta(κ·μ_k, κ·(1−μ_k)) 의 (E[(Q−σ)⁺], S(σ)=P(Q>σ)).

    형상: sigma (M,) · mu (K, M) · kappa (M,) · w (K,) → 반환 각 (M,)
    """
    return _make_excess_fn(mu, kappa, w)(np.asarray(sigma, dtype=float))


# ---------------- 이산 지지 (DP 대조·비모수 폴백) ----------------

def discrete_excess(sigma, values, probs):
    """Q가 유한 지지 {v_l} (확률 p_l) 일 때의 (E[(Q−σ)⁺], S(σ)).

    형상: sigma (M,) · values (L,) · probs (L, M) → 각 (M,)
    DP 오라클과 **정확히 같은 분포**를 엔진에 물려 최적성을 검증하는 경로.
    """
    sig = np.asarray(sigma, dtype=float)[None, :]                        # (1,M)
    v = np.asarray(values, dtype=float)[:, None]                         # (L,1)
    gain = np.maximum(v - sig, 0.0)
    return (probs * gain).sum(axis=0), (probs * (v > sig)).sum(axis=0)


# ---------------- 일반해 ----------------

def solve_reservation(excess_surv, target, mean, q_lo=0.0, q_hi=1.0,
                      tol=1e-10, max_iter=40, warm=None):
    """(★) E[(Q−σ)⁺] = target 의 해 σ. 벡터화(M개 상자 동시).

    excess_surv(sigma) -> (excess, survival) 를 받는다 (survival = −d excess/dσ).
    안전장치 부착 Newton: 볼록·감소 구조에서 Newton은 2차 수렴이지만 S(σ)→0 인
    평탄부(σ가 q_hi 근처)에서 스텝이 발산하므로, 매 스텝 브래킷 [lo,hi]에 가두고
    벗어나면 이분으로 강등한다. 브래킷 폭이 tol 이하가 되면 조기 종료 —
    수렴한 상자가 남은 반복을 끌고 가지 않게 한다 (연속 경로 지연의 지배 요인).
    """
    t = np.asarray(target, dtype=float)
    mu = np.asarray(mean, dtype=float)
    M = t.shape[0]

    sigma = np.full(M, q_hi, dtype=float)
    linear = t >= (mu - q_lo)                    # 선형 구간 — 닫힌형 (σ ≤ q_lo)
    sigma[linear] = mu[linear] - t[linear]
    trivial = t <= 0.0                           # 비용 0 이하 → 항상 여는 것이 이득
    sigma[trivial] = q_hi
    todo = ~(linear | trivial)
    if not todo.any():
        return sigma

    lo = np.full(M, q_lo, dtype=float)
    hi = np.full(M, q_hi, dtype=float)
    # 웜스타트: 평균만 맞춘 Bernoulli의 예약값에서 출발한다. 해는 바뀌지 않고
    # (수렴점은 동일) 이분 강등 구간을 건너뛰어 반복 수만 줄인다 — 중간에서
    # 출발하면 S(σ)→0 인 평탄부에서 Newton이 매번 이분으로 떨어져 40회를 다 돈다.
    x = 0.5 * (lo + hi) if warm is None else np.clip(warm, q_lo + 1e-6, q_hi - 1e-6)
    for _ in range(max_iter):
        exc, surv = excess_surv(x)
        f = exc - t                              # f는 감소함수, f(lo)>0>f(hi)
        lo = np.where(f > 0, x, lo)              # f>0 → 해는 오른쪽
        hi = np.where(f > 0, hi, x)
        if np.all((hi - lo)[todo] <= tol) or np.all(np.abs(f[todo]) <= tol):
            break
        step = np.where(surv > 1e-12, f / np.maximum(surv, 1e-12), 0.0)
        x_new = x + step                         # Newton: f/(-f') = f/surv
        bad = ~np.isfinite(x_new) | (x_new <= lo) | (x_new >= hi)
        x = np.where(bad, 0.5 * (lo + hi), x_new)
    sigma[todo] = x[todo]
    return sigma


def beta_reservation(target, mu, kappa, w):
    """Beta 혼합 상금의 예약값 (형상은 beta_mixture_excess와 동일)."""
    mean = (w[:, None] * mu).sum(axis=0)
    return solve_reservation(_make_excess_fn(mu, kappa, w), target, mean,
                             warm=bernoulli_reservation(mean, target))


def discrete_reservation(target, values, probs):
    """이산 지지 상금의 예약값."""
    v = np.asarray(values, dtype=float)
    mean = (probs * v[:, None]).sum(axis=0)
    return solve_reservation(lambda s: discrete_excess(s, values, probs),
                             target, mean, q_lo=float(v.min()), q_hi=float(v.max()))


# ---------------- 상자별 지지가 다른 이산 상금 (관측 상금 체제) ----------------

def perbox_excess(sigma, values, probs):
    """상자마다 **다른 지지점**을 갖는 이산 상금의 (E[(X−σ)⁺], S(σ)).

    형상: sigma (M,) · values (L, M) · probs (L, M) → 각 (M,)

    `discrete_excess` 는 지지 `values` 를 전 상자가 공유한다고 가정한다. 관측 상금
    X_m = E[Q | v] 는 상자마다 검증기 채널이 달라(PerModelNoise) 지지점 자체가 다르므로
    이 일반형이 필요하다.
    """
    sig = np.asarray(sigma, dtype=float)[None, :]                        # (1,M)
    v = np.asarray(values, dtype=float)                                  # (L,M)
    gain = np.maximum(v - sig, 0.0)
    return (probs * gain).sum(axis=0), (probs * (v > sig)).sum(axis=0)


def perbox_reservation(target, values, probs, q_lo=0.0, q_hi=1.0):
    """관측 상금 예약값 — 상자별 지지 (L,M)·확률 (L,M).

    ★ 왜 필요한가 (Phase 17, 레드팀 CRITICAL): Weitzman 의 예약값은 "상자를 열어서
    **실제로 얻는 상금**"의 분포로 풀어야 한다. 그런데 이 라우터가 상자를 열어 얻는 것은
    잠재 품질 Q 가 아니라 검증기를 통해 본 사후평균 **X = E[Q|v]** 이다 — 정지 규칙
    (`pandora.py` 의 `bel.prize(v,m) ≥ max σ`)과 최종 답 선택이 쓰는 바로 그 양이다.
    X 는 Q 의 mean-preserving contraction 이므로 모든 σ 에서 E[(X−σ)⁺] ≤ E[(Q−σ)⁺] 이고,
    따라서 잠재 Q 로 푼 σ 는 **체계적으로 과대**하다. 상자마다 검증기 예리함이 다르면
    (PerModelNoise) 과대 정도가 달라 **상자 순서까지 뒤집힌다.**
    기존 3a DP 검증이 이를 못 본 이유는 `ExactNoise`(관측=진짜 상금)로 돌았기 때문이다.

    q_lo/q_hi: 상금이 보정확률이므로 [0,1] 이 항상 유효한 브래킷이다.
    """
    mean = (np.asarray(probs, dtype=float) * np.asarray(values, dtype=float)).sum(axis=0)
    return solve_reservation(lambda s: perbox_excess(s, values, probs),
                             target, mean, q_lo=q_lo, q_hi=q_hi)
