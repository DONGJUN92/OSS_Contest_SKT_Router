"""신념 예측기 — Amortized IRT 인코더와 판별식 대조군 (Phase 17: 패키징 이관).

**왜 여기 있나 (Phase 17 레드팀 지적 #8)**: 두 클래스는 원래 루트의 실험 스크립트
`phase2_stages.py`에 있었고, `src/router.py`가 `from phase2_stages import IRTEncoder` 로
당겨 썼다. 그런데 `pyproject.toml`은 `src*`·`baselines*` 만 패키징하므로 **설치된
패키지에서 `LPBRouter.fit()` 이 ImportError** 로 죽었다 — 게다가 `phase2_stages` 는
모듈 로드 시점에 `run_phase2`(→ `config.yaml` 파일 읽기)를 끌어온다. 배포 경로가 실험
하네스에 의존하면 안 되므로 라이브러리 본체로 옮겼다.

`phase2_stages` 는 이 모듈을 재수출하는 하위호환 shim 으로 남아 기존 실행기·프로브가
무변경 동작한다 (`tests/test_packaging.py` 가 두 경로가 같은 객체임을 고정).

  IRTEncoder : features → (μ, σ) → probit 근사 p̄_m.  IRT 파라미터 (a, b)는 MML 적합값
               고정, 선형 헤드 2개만 학습 → 과적합 표면적 최소 (v5 설계 원칙).
  DiscLR     : 모델별 독립 로지스틱 회귀 (v4 프로파일러 = 판별식 신념 대조군).
               `baselines.LearnedRouter`(RouteLLM류 단일호출)의 예측기이기도 하다.
"""
import numpy as np

from .irt.mml import Adam, _sigmoid

__all__ = ["C_PROBIT", "IRTEncoder", "DiscLR"]

C_PROBIT = np.pi / 8.0


class IRTEncoder:
    """features → (μ, σ) → p̄_m = sigmoid(a_m(μ − b_{d,m}) / sqrt(1 + c a² σ²) + r_m(x)).

    선형 헤드 2개(μ, σ)만 학습 — IRT 파라미터 (a, b)는 MML 적합값으로 고정.

    ── Phase 19 용량 확장 (기본 off, 기존 경로 비트 보존) ──────────────────────────
    `hidden`  : μ·σ 헤드 앞에 tanh 은닉층을 둔다. 난이도가 특성의 비선형 함수일 때 필요.
    `residual`: **모델별 잔차 헤드** `r_m(x)` 를 로짓에 더한다.
        왜: 단일 잠재 θ 는 "모델 X 는 코드에 강하고 수학에 약하다"를 **구조적으로 표현할 수
        없다** — 모든 모델이 같은 θ 축 위의 한 점으로 정렬되기 때문이다. Phase 16 이 실물
        RouterBench 에서 본 mmlu/arc 약점이 정확히 이 한계였고, 그때는 도메인 라벨로
        (다집단 2PL) 우회했지만 SKT 런타임엔 라벨이 없다. 잔차 헤드는 라벨 없이 같은
        자유도를 준다: IRT 백본이 공통 난이도를, r_m 이 모델별 편차를 담는다.
        r_m ≡ 0 이면 기존 모형과 동일하므로 **일반화이지 대체가 아니다.**
    두 확장 모두 채택은 측정으로 판단한다 (`eval/probe_predictor.py`).
    """

    def __init__(self, a, b, l2=1e-4, hidden: int = 0, residual: bool = False,
                 l2_res: float = 1e-3):
        self.a, self.b, self.l2 = a, b, l2
        self.hidden, self.residual, self.l2_res = int(hidden), bool(residual), l2_res

    def _trunk(self, X1):
        if not self.hidden:
            return X1, None
        H = np.tanh(X1 @ self.W1)
        return H, H

    def _forward(self, X1, dom):
        T, H = self._trunk(X1)
        mu = T @ self.Wmu
        u = T @ self.Ws
        s = np.logaddexp(0.0, u)                       # softplus
        B = self.b[dom] if self.b.shape[0] > 1 else np.tile(self.b, (len(X1), 1))
        d = np.sqrt(1 + C_PROBIT * self.a[None, :] ** 2 * s[:, None] ** 2)
        z0 = self.a[None, :] * (mu[:, None] - B) / d
        z = z0 + (X1 @ self.Wr) if self.residual else z0
        return mu, u, s, B, d, z0, _sigmoid(z), H

    def fit(self, X, dom, y, steps=1500, lr=0.05, seed: int = 0):
        X1 = np.hstack([X, np.ones((len(X), 1))])
        din, M = X1.shape[1], y.shape[1]
        rng = np.random.default_rng(seed)
        params = []
        if self.hidden:
            # 작은 무작위 초기화 — 0 이면 tanh 은닉층의 대칭이 깨지지 않는다
            self.W1 = rng.normal(0, 1.0 / np.sqrt(din), size=(din, self.hidden))
            self.Wmu = np.zeros(self.hidden)
            self.Ws = np.zeros(self.hidden)
            params += [self.W1]
        else:
            self.Wmu = np.zeros(din)
            self.Ws = np.zeros(din)
        params += [self.Wmu, self.Ws]
        if self.residual:
            self.Wr = np.zeros((din, M))
            params.append(self.Wr)
        opt = Adam(params, lr)
        for _ in range(steps):
            mu, u, s, B, d, z0, p, H = self._forward(X1, dom)
            e = (p - y) / y.size                       # 평균 BCE의 dL/dz
            gmu = (e * self.a[None, :] / d).sum(axis=1)
            gs = (e * (-z0) * C_PROBIT * self.a[None, :] ** 2 * s[:, None] / d ** 2).sum(axis=1)
            gu = gs * _sigmoid(u)
            grads = []
            if self.hidden:
                dH = np.outer(gmu, self.Wmu) + np.outer(gu, self.Ws)
                grads.append(X1.T @ (dH * (1.0 - H ** 2)) + self.l2 * self.W1)
                gWmu, gWs = H.T @ gmu + self.l2 * self.Wmu, H.T @ gu + self.l2 * self.Ws
            else:
                gWmu, gWs = X1.T @ gmu + self.l2 * self.Wmu, X1.T @ gu + self.l2 * self.Ws
            grads += [gWmu, gWs]
            if self.residual:
                grads.append(X1.T @ e + self.l2_res * self.Wr)
            opt.step(grads)
        return self

    def predict(self, X, dom):
        X1 = np.hstack([X, np.ones((len(X), 1))])
        return self._forward(X1, dom)[-2]

    def predict_row(self, x, d):
        return self.predict(x[None, :], np.array([d]))[0]

    def belief_row(self, x):
        """Phase 3 M2용: 쿼리의 θ 신념 (μ, σ) 반환.

        ⚠ 잔차 헤드는 θ 가 아니라 **로짓에 직접** 얹히므로 (μ,σ) 만으로는 전달되지 않는다.
        Weitzman 엔진은 (μ,σ)+ICC 로 신념을 만들므로, residual=True 를 쓸 때는
        `residual_row()` 를 함께 소비해야 예측과 신념이 일치한다 (`router.py` 참조).
        """
        X1 = np.hstack([x, 1.0])
        T = np.tanh(X1[None, :] @ self.W1)[0] if self.hidden else X1
        mu = float(T @ self.Wmu)
        s = float(np.logaddexp(0.0, float(T @ self.Ws)))
        return mu, s

    def residual_row(self, x):
        """모델별 로짓 잔차 r_m(x) — residual=False 면 0 벡터."""
        if not self.residual:
            return None
        X1 = np.hstack([x, 1.0])
        return X1 @ self.Wr


class DiscLR:
    """판별식 베이스라인: 모델별 독립 로지스틱 회귀 (v4 프로파일러에 해당)."""

    def __init__(self, l2=1e-4):
        self.l2 = l2

    def fit(self, X, dom, y, steps=1500, lr=0.05):
        X1 = np.hstack([X, np.ones((len(X), 1))])
        self.W = np.zeros((X1.shape[1], y.shape[1]))
        opt = Adam([self.W], lr)
        for _ in range(steps):
            p = _sigmoid(X1 @ self.W)
            opt.step([X1.T @ ((p - y) / y.size) + self.l2 * self.W])
        return self

    def predict(self, X, dom):
        X1 = np.hstack([X, np.ones((len(X), 1))])
        return _sigmoid(X1 @ self.W)

    def predict_row(self, x, d):
        return self.predict(x[None, :], None)[0]
