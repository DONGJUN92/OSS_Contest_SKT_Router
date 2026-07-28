"""연속 품질용 IRT — Beta 반응모형의 주변최대우도 적합 (B4, Q4=연속 분기).

모델
    Q_im | θ_i ~ Beta(κ_m·μ_im, κ_m·(1−μ_im)),   μ_im = sigmoid(a_m(θ_i − b_{d,m}))
    θ ~ N(0,1)

설계 근거
  · **평균 함수가 2PL 그대로**다 — 이진 세계에서 검증된 μ = sigmoid(a(θ−b)) 구조를
    유지하고, 모델당 파라미터는 (a, b_d) 에 정밀도 κ 하나만 추가된다. v5 의사결정
    로그의 "과적합 표면적 극소" 원칙(Optuna 변수 5+→2)과 정합.
  · **Bernoulli를 극한으로 포함**한다: Beta(κμ, κ(1−μ)) 는 κ→0 에서 {0,1} 위의
    Bernoulli(μ)로 수렴한다 (평균 μ 보존, 분산 μ(1−μ)/(1+κ) → μ(1−μ)).
    즉 이 모형은 이진 세계의 일반화이지 대체가 아니다 — tests/test_prize.py가 고정.
  · 예약값은 정칙 불완전베타의 닫힌형으로 풀린다 (engine/prize.py) — 1D 수치적분 불요.

한계 (정직 기록) — 두 가지 모두 실측으로 확인했다

① **경계 점질량**: Beta는 (0,1) 개구간 밀도라 0·1의 점질량을 표현할 수 없고, 경계를
   eps로 압착해 적합한다. 압착이 거칠면 정보가 파괴된다 — 합성 검증에서 표본의 8.7%가
   1−1e-4 위에 몰려 eps=1e-4는 주변LL 5.23, eps=1e-12는 9.94로 **적합이 실제로 나빠졌다**.
   기본값을 1e-9로 낮춘 근거다. 그러나 이는 완화이지 해결이 아니다:
   품질이 0·1에 크게 몰린 채점(전부 맞음/전부 틀림이 흔한 루브릭)에서는 Beta 자체가
   오설정이며, 그때는 **이산 지지 경로**(`engine/prize.py:discrete_reservation`,
   DP 최적성 검증 완료)가 정확해다. 로더 리포트의 `quality_levels`가 어느 쪽인지
   알려준다 — 선택 규칙은 docs/branch_decision_table.md.

② **판별도 a의 약한 식별성**: b가 θ 분포의 꼬리에 있는 모델은 μ(θ)가 포화해 기울기
   정보가 데이터에 거의 없다. 우도가 완전히 수렴한 상태에서도 a가 축소 추정된다
   (합성 검증: 참값 2.4 → 추정 1.87). 이진 IRT에서 극단 난이도 문항의 변별도가
   부정확한 것과 같은 현상이다. 라우팅이 소비하는 것은 a 자체가 아니라 서열
   파라미터 b(복원 corr>0.99)와 예약값 σ이므로 회귀 테스트는 그 둘로 판정한다.
"""
import numpy as np
from scipy.special import digamma, gammaln

from .mml import Adam, _sigmoid


def _softplus(x):
    return np.logaddexp(0.0, x)


def fit_beta_mml(q: np.ndarray, domains: np.ndarray, n_domains: int,
                 per_domain_b: bool = True, K: int = 31, steps: int = 400,
                 lr: float = 0.05, eps: float = 1e-9) -> dict:
    """(a_m, b_{d,m}, κ_m) 적합. q는 (N, M) ∈ [0,1] 연속 품질."""
    N, M = q.shape
    t, w = np.polynomial.hermite_e.hermegauss(K)
    logw = np.log(w / np.sqrt(2 * np.pi))
    D = n_domains if per_domain_b else 1
    dom = domains if per_domain_b else np.zeros(N, dtype=int)

    qc = np.clip(q, eps, 1.0 - eps)                     # Beta는 개구간 밀도 (위 한계 참조)
    logq, log1q = np.log(qc)[:, None, :], np.log1p(-qc)[:, None, :]   # (N,1,M)

    raw_a = np.full(M, 0.55)                            # softplus⁻¹ → a ≈ 1.0
    b = np.zeros((D, M))
    raw_k = np.full(M, 1.85)                            # softplus⁻¹ → κ ≈ 2.0
    opt = Adam([raw_a, b, raw_k], lr)
    curve = []

    for _ in range(steps):
        a, kap = _softplus(raw_a), _softplus(raw_k)
        B = b[dom]                                      # (N,M)
        tb = t[None, :, None] - B[:, None, :]           # (N,K,M)
        z = a[None, None, :] * tb
        mu = _sigmoid(z)
        al = np.maximum(kap[None, None, :] * mu, 1e-8)
        be = np.maximum(kap[None, None, :] * (1.0 - mu), 1e-8)

        logpdf = ((al - 1.0) * logq + (be - 1.0) * log1q
                  - (gammaln(al) + gammaln(be) - gammaln(kap)[None, None, :]))
        s = logpdf.sum(axis=2) + logw[None, :]          # (N,K)
        mx = s.max(axis=1, keepdims=True)
        Ls = np.exp(s - mx)
        curve.append(float((mx[:, 0] + np.log(Ls.sum(axis=1))).mean()))
        r = Ls / Ls.sum(axis=1, keepdims=True)          # θ 사후 (N,K)

        # ∂logf/∂μ = κ[log q − log(1−q) − ψ(α) + ψ(β)]
        dmu = kap[None, None, :] * (logq - log1q - digamma(al) + digamma(be))
        gz = r[:, :, None] * dmu * mu * (1.0 - mu)      # ∂/∂z (사후 가중)
        ga = (gz * tb).sum(axis=(0, 1)) * _sigmoid(raw_a)
        gb_flat = -(gz.sum(axis=1)) * a[None, :]        # (N,M)
        gb = np.zeros_like(b)
        np.add.at(gb, dom, gb_flat)
        # ∂logf/∂κ = μ log q + (1−μ)log(1−q) − μψ(α) − (1−μ)ψ(β) + ψ(κ)
        dk = (mu * logq + (1.0 - mu) * log1q - mu * digamma(al)
              - (1.0 - mu) * digamma(be) + digamma(kap)[None, None, :])
        gk = (r[:, :, None] * dk).sum(axis=(0, 1)) * _sigmoid(raw_k)
        opt.step([-ga / N, -gb / N, -gk / N])           # −logL 최소화

    return {"a": _softplus(raw_a), "b": b, "kappa": _softplus(raw_k),
            "per_domain_b": per_domain_b, "curve": curve, "final_ll": curve[-1]}


def beta_heldout_ll(params: dict, q: np.ndarray, domains: np.ndarray,
                    K: int = 31, eps: float = 1e-4) -> float:
    """적합된 (a,b,κ)로 held-out 품질행렬의 주변 로그우도 (쿼리당 평균)."""
    a, b, kap = params["a"], params["b"], params["kappa"]
    dom = domains if params["per_domain_b"] else np.zeros(len(q), dtype=int)
    t, w = np.polynomial.hermite_e.hermegauss(K)
    logw = np.log(w / np.sqrt(2 * np.pi))
    qc = np.clip(q, eps, 1.0 - eps)
    mu = _sigmoid(a[None, None, :] * (t[None, :, None] - b[dom][:, None, :]))
    al = np.maximum(kap[None, None, :] * mu, 1e-8)
    be = np.maximum(kap[None, None, :] * (1.0 - mu), 1e-8)
    logpdf = ((al - 1.0) * np.log(qc)[:, None, :] + (be - 1.0) * np.log1p(-qc)[:, None, :]
              - (gammaln(al) + gammaln(be) - gammaln(kap)[None, None, :]))
    s = logpdf.sum(axis=2) + logw[None, :]
    mx = s.max(axis=1, keepdims=True)
    return float((mx[:, 0] + np.log(np.exp(s - mx).sum(axis=1))).mean())
