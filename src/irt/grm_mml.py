"""등급반응모형(Samejima GRM)의 주변최대우도 적합 — 다수준 이산 품질 (B4 잔여).

모델 (L개 등급 v_0 < v_1 < … < v_{L−1})
    P(Q ≥ v_l | θ) = sigmoid(a_m(θ − b_{m,l})),   l = 1..L−1,   b_{m,1} ≤ … ≤ b_{m,L−1}
    P(Q = v_l | θ) = P(Q ≥ v_l) − P(Q ≥ v_{l+1})                  (경계 항은 1, 0)

왜 이 모형인가 (B4의 실측 근거)
  Beta 반응모형은 (0,1) **개구간** 밀도라 0·1의 점질량을 표현하지 못한다. 그런데 부분점수
  채점은 점질량을 크게 만든다 — World F-c 실측에서 **정확히 1.0이 65.8%**, 0.0이 1.2%였다.
  그 체제에서 Beta는 오설정이고 경계 압착 상수 eps가 결과를 좌우한다
  (docs/reflections/phase10.md). GRM은 등급 확률을 직접 모델링하므로 점질량이 자연스럽다.

설계 정합성
  · **L=2에서 2PL과 정확히 일치**한다: P(Q ≥ v_1|θ) = sigmoid(a(θ−b_1)) 이 곧 정답확률.
    즉 이 모형도 이진의 대체가 아니라 **일반화**다 (tests/test_grm.py가 고정).
  · 소비처 `engine/prize.py:discrete_reservation` 은 **이미 DP 최적성 검증을 마쳤다**
    (4수준 부분점수, 120 인스턴스, 격차 <1e-9). 남아 있던 것이 이 적합부 하나였다.
  · 특수함수를 쓰지 않아 Beta 경로(호출당 정칙 불완전베타 1,230회)의 성능 문제도 없다.

식별성: θ ~ N(0,1) 고정, a > 0 은 softplus, 문턱 단조성 b_{m,1} ≤ … 는 누적합으로 강제
(b_{m,1} 과 양수 증분 δ_{m,l} = softplus(raw) 의 누적).
"""
import numpy as np

from .mml import Adam, _sigmoid


def _softplus(x):
    return np.logaddexp(0.0, x)


def _thresholds(b1, raw_d):
    """(M,) 첫 문턱 + (M, L−2) 증분 → (M, L−1) 단조 문턱."""
    if raw_d.shape[1] == 0:
        return b1[:, None]
    return np.concatenate([b1[:, None], b1[:, None] + np.cumsum(_softplus(raw_d), axis=1)],
                          axis=1)


def grade_index(q: np.ndarray, values: np.ndarray) -> np.ndarray:
    """연속/이산 품질을 가장 가까운 등급 인덱스로 사상 (L개 지지)."""
    return np.abs(np.asarray(q)[..., None] - np.asarray(values)[None, None, :]).argmin(-1) \
        if q.ndim == 2 else np.abs(q[..., None] - values[None, :]).argmin(-1)


def infer_values(q: np.ndarray, max_levels: int = 8) -> np.ndarray:
    """관측된 품질에서 등급 지지를 추정. 수준이 많으면 분위수로 축약한다."""
    uniq = np.unique(q)
    if len(uniq) <= max_levels:
        return uniq
    return np.unique(np.quantile(q, np.linspace(0, 1, max_levels)))


def fit_grm_mml(q: np.ndarray, domains: np.ndarray, n_domains: int, values=None,
                per_domain_b: bool = True, K: int = 31, steps: int = 400,
                lr: float = 0.05) -> dict:
    """(a_m, b_{d,m,l}) 적합. q는 (N, M) 품질 행렬, values는 등급 지지(오름차순).

    반환의 `probs(theta_nodes)` 는 (L, K, M) 등급확률 텐서를 주며,
    `engine/prize.py:discrete_reservation` 이 그대로 소비한다.
    """
    N, M = q.shape
    values = np.asarray(infer_values(q) if values is None else values, dtype=float)
    L = len(values)
    if L < 2:
        raise ValueError(f"등급이 {L}개 — GRM은 2개 이상 필요")
    g = grade_index(q, values)                          # (N, M) 등급 인덱스
    t, w = np.polynomial.hermite_e.hermegauss(K)
    logw = np.log(w / np.sqrt(2 * np.pi))
    D = n_domains if per_domain_b else 1
    dom = domains if per_domain_b else np.zeros(N, dtype=int)

    raw_a = np.full(M, 0.55)                            # softplus⁻¹ → a ≈ 1.0
    b1 = np.zeros((D, M))                               # 첫 문턱
    raw_d = np.full((D, M, L - 2), -0.5) if L > 2 else np.zeros((D, M, 0))
    opt = Adam([raw_a, b1, raw_d], lr)
    curve = []
    onehot = np.eye(L, dtype=bool)[g]                   # (N, M, L)

    for _ in range(steps):
        a = _softplus(raw_a)
        # 누적확률 P(Q ≥ v_l | θ) — l = 1..L-1
        P = np.empty((D, K, M, L - 1))
        for d in range(D):
            thr = _thresholds(b1[d], raw_d[d])          # (M, L-1)
            P[d] = _sigmoid(a[None, :, None] *
                            (t[:, None, None] - thr[None, :, :]))
        # 등급확률 = 인접 누적확률의 차 (경계는 1, 0)
        ones = np.ones((D, K, M, 1))
        zeros = np.zeros((D, K, M, 1))
        cum = np.concatenate([ones, P, zeros], axis=3)  # (D,K,M,L+1)
        pr = np.clip(cum[..., :-1] - cum[..., 1:], 1e-12, 1.0)   # (D,K,M,L)

        pr_n = pr[dom]                                  # (N,K,M,L)
        ll = np.log(pr_n[onehot[:, None, :, :].repeat(K, axis=1)].reshape(N, K, M))
        s = ll.sum(axis=2) + logw[None, :]
        mx = s.max(axis=1, keepdims=True)
        Ls = np.exp(s - mx)
        curve.append(float((mx[:, 0] + np.log(Ls.sum(axis=1))).mean()))
        r = Ls / Ls.sum(axis=1, keepdims=True)          # θ 사후 (N,K)

        # ∂logL/∂(누적확률 항): 관측 등급 l 의 확률 pr = C_l − C_{l+1}
        #   ∂log pr/∂C_l = 1/pr,  ∂log pr/∂C_{l+1} = −1/pr
        inv = np.zeros((N, K, M, L))
        obs_pr = pr_n[onehot[:, None, :, :].repeat(K, axis=1)].reshape(N, K, M)
        inv[onehot[:, None, :, :].repeat(K, axis=1)] = (1.0 / obs_pr).ravel()
        dC = np.zeros((N, K, M, L + 1))
        dC[..., :-1] += inv
        dC[..., 1:] -= inv
        dC = dC[..., 1:-1]                              # 내부 문턱만 (L-1)

        ga = np.zeros(M)
        gb1 = np.zeros((D, M))
        gd = np.zeros((D, M, max(L - 2, 0)))
        for d in range(D):
            sel = dom == d
            if not sel.any():
                continue
            thr = _thresholds(b1[d], raw_d[d])          # (M,L-1)
            Pd = P[d]                                   # (K,M,L-1)
            dP = Pd * (1.0 - Pd)                        # sigmoid'
            wgt = (r[sel][:, :, None, None] * dC[sel]).sum(axis=0)   # (K,M,L-1)
            z_arg = t[:, None, None] - thr[None, :, :]
            ga += (wgt * dP * z_arg).sum(axis=(0, 2)) * _sigmoid(raw_a)
            gthr = -(wgt * dP).sum(axis=0) * a[:, None]              # (M,L-1)
            gb1[d] = gthr.sum(axis=1)                   # b1은 모든 문턱에 공통
            if L > 2:                                   # δ_l 은 문턱 l..L-1 에 기여
                tail = np.cumsum(gthr[:, ::-1], axis=1)[:, ::-1][:, 1:]
                gd[d] = tail * _sigmoid(raw_d[d])
        opt.step([-ga / N, -gb1 / N, -(gd / N).reshape(raw_d.shape)])

    a = _softplus(raw_a)
    thr = np.stack([_thresholds(b1[d], raw_d[d]) for d in range(D)])  # (D,M,L-1)
    return {"a": a, "thr": thr, "values": values, "per_domain_b": per_domain_b,
            "curve": curve, "final_ll": curve[-1], "n_levels": L}


def grm_probs(params: dict, theta_nodes: np.ndarray, domain: int) -> np.ndarray:
    """(L, K, M) 등급확률 — discrete_reservation 소비 형식.

    반환[l, k, m] = P(Q_m = values[l] | θ = theta_nodes[k]).
    """
    a = params["a"]
    thr = params["thr"][domain if params["per_domain_b"] else 0]      # (M,L-1)
    P = _sigmoid(a[None, :, None] * (theta_nodes[:, None, None] - thr[None, :, :]))
    K, M, _ = P.shape
    cum = np.concatenate([np.ones((K, M, 1)), P, np.zeros((K, M, 1))], axis=2)
    pr = np.clip(cum[..., :-1] - cum[..., 1:], 1e-12, 1.0)            # (K,M,L)
    return np.transpose(pr, (2, 0, 1))


def grm_heldout_ll(params: dict, q: np.ndarray, domains: np.ndarray, K: int = 31) -> float:
    """적합된 파라미터로 held-out 품질행렬의 주변 로그우도 (쿼리당 평균)."""
    values = params["values"]
    g = grade_index(q, values)
    t, w = np.polynomial.hermite_e.hermegauss(K)
    logw = np.log(w / np.sqrt(2 * np.pi))
    N, M = q.shape
    ll = np.empty((N, K))
    doms = domains if params["per_domain_b"] else np.zeros(N, dtype=int)
    for d in np.unique(doms):
        sel = doms == d
        pr = grm_probs(params, t, int(d))                # (L,K,M)
        take = pr[g[sel]]                                # (n,M,K,M) 아님 — 인덱싱 주의
        take = np.stack([pr[g[sel][:, m], :, m] for m in range(M)], axis=2)  # (n,K,M)
        ll[sel] = np.log(np.clip(take, 1e-12, 1.0)).sum(axis=2)
    s = ll + logw[None, :]
    mx = s.max(axis=1, keepdims=True)
    return float((mx[:, 0] + np.log(np.exp(s - mx).sum(axis=1))).mean())
