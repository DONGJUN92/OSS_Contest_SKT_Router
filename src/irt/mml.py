"""2PL IRT의 주변최대우도(MML) 적합 — 가우스-에르미트 구적 + Adam (numpy 전용).

모델: P(y_im = 1 | θ_i) = sigmoid(a_m (θ_i − b_{d_i, m})),  θ ~ N(0,1)
  - per_domain_b=False: b는 (1, M) — 표준 1D 2PL (World A 가정)
  - per_domain_b=True : b는 (D, M) — 다집단(multigroup) 2PL. 도메인 특화 모델
    (World B의 specialist 구조)을 모델당 D개 파라미터로 흡수. 파라미터 수는
    여전히 M(1+D)개 수준 — v5의 '과적합 표면적 극소' 원칙 유지.

식별성: θ ~ N(0,1) 고정으로 위치·척도 고정, a>0은 softplus로 강제.
"""
import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def _log_sigmoid(z):
    return -np.logaddexp(0.0, -z)


class Adam:
    def __init__(self, params: list[np.ndarray], lr: float):
        self.p = params
        self.lr = lr
        self.m = [np.zeros_like(x) for x in params]
        self.v = [np.zeros_like(x) for x in params]
        self.t = 0

    def step(self, grads: list[np.ndarray]):
        self.t += 1
        for i, g in enumerate(grads):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            mh = self.m[i] / (1 - 0.9 ** self.t)
            vh = self.v[i] / (1 - 0.999 ** self.t)
            self.p[i] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def fit_mml(y: np.ndarray, domains: np.ndarray, n_domains: int,
            per_domain_b: bool = False, K: int = 31, steps: int = 400,
            lr: float = 0.05) -> dict:
    """(a_m, b_{d,m}) 적합. 반환: params + 학습곡선 + θ 사후평균."""
    N, M = y.shape
    t, w = np.polynomial.hermite_e.hermegauss(K)      # ∫f(x)e^{-x²/2}dx 노드
    logw = np.log(w / np.sqrt(2 * np.pi))             # N(0,1) 가중치로 정규화
    D = n_domains if per_domain_b else 1
    dom = domains if per_domain_b else np.zeros(N, dtype=int)

    raw_a = np.full(M, 0.55)                          # softplus^-1 초기값 (a≈1.0)
    b = np.zeros((D, M))
    opt = Adam([raw_a, b], lr)
    curve = []

    for _ in range(steps):
        a = np.logaddexp(0.0, raw_a)                  # softplus
        B = b[dom]                                    # (N, M)
        z = a[None, None, :] * (t[None, :, None] - B[:, None, :])       # (N,K,M)
        ll = y[:, None, :] * _log_sigmoid(z) + (1 - y[:, None, :]) * _log_sigmoid(-z)
        s = ll.sum(axis=2) + logw[None, :]            # (N, K)
        mx = s.max(axis=1, keepdims=True)
        Ls = np.exp(s - mx)
        logL = float((mx[:, 0] + np.log(Ls.sum(axis=1))).mean())
        curve.append(logL)

        r = Ls / Ls.sum(axis=1, keepdims=True)        # 노드별 사후 (N, K)
        p = _sigmoid(z)
        g = r[:, :, None] * (y[:, None, :] - p)       # dlogL/dz 누적항 (N,K,M)
        ga = (g * (t[None, :, None] - B[:, None, :])).sum(axis=(0, 1)) * _sigmoid(raw_a)
        gb_flat = -(g.sum(axis=1)) * a[None, :]       # (N, M)
        gb = np.zeros_like(b)
        np.add.at(gb, dom, gb_flat)
        opt.step([-ga / N, -gb / N])                  # −logL 최소화

    a = np.logaddexp(0.0, raw_a)
    theta_hat = (r * t[None, :]).sum(axis=1)          # 마지막 스텝의 사후평균 θ
    return {"a": a, "b": b, "per_domain_b": per_domain_b, "curve": curve,
            "theta_hat": theta_hat, "final_ll": curve[-1]}


def heldout_marginal_ll(params: dict, y: np.ndarray, domains: np.ndarray,
                        K: int = 31) -> float:
    """적합된 (a,b)로 held-out 응답행렬의 주변 로그우도 (쿼리당 평균)."""
    a, b = params["a"], params["b"]
    dom = domains if params["per_domain_b"] else np.zeros(len(y), dtype=int)
    t, w = np.polynomial.hermite_e.hermegauss(K)
    logw = np.log(w / np.sqrt(2 * np.pi))
    B = b[dom]
    z = a[None, None, :] * (t[None, :, None] - B[:, None, :])
    ll = y[:, None, :] * _log_sigmoid(z) + (1 - y[:, None, :]) * _log_sigmoid(-z)
    s = ll.sum(axis=2) + logw[None, :]
    mx = s.max(axis=1, keepdims=True)
    return float((mx[:, 0] + np.log(np.exp(s - mx).sum(axis=1))).mean())
