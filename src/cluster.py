"""무감독 프롬프트 군집 → pseudo-domain (Phase 17, 레드팀 MAJOR "최고 EV 미구현 기법").

**문제**: 다집단 2PL(도메인별 난이도 `b_{d,m}`)은 "모델 X가 이 종류의 문제에 강하다"를
잡아 준다. 실물 RouterBench 에서 다집단은 mmlu +0.07 · chinese +0.18, 종합 0.865→0.871 로
**측정된 이득**이 있었다. 그런데 SKT 런타임은 task/benchmark 이름을 주지 않으므로
(대회 상세) 배포 구성은 단일집단으로 퇴화해 그 이득을 통째로 버린다.

**해법**: 도메인 라벨 대신 **프롬프트 자체에서** 집단을 만든다. 프롬프트 특성은 런타임에
항상 있으므로(인코더 출력) 규칙 위반이 없고, 라벨도 필요 없다. 공개 데이터로 군집 중심을
적합해 두고, 추론 시 가장 가까운 중심의 인덱스를 pseudo-domain 으로 쓴다.

Phase 13 은 "프롬프트→도메인 **분류기**는 실데이터 전에 정당화되지 않는다"며 보류했는데,
그 판단은 **지도 분류기**(도메인 라벨을 맞추는 문제)를 전제한 것이었다. 무감독 군집은
라벨을 맞추려 하지 않고 난이도 이질성만 흡수하므로 전제가 다르고, k 하나만 늘어난다
(v5 의 "탐색 표면적 축소" 원칙과 충돌하지 않는 수준). 실제 값은 측정으로 판정한다:
`eval/probe_cluster_domain.py`.

의존성 0·결정론 (numpy k-means, k-means++ 초기화, 시드 고정).
"""
import numpy as np


class PromptClusterer:
    """프롬프트 특성 → 군집 인덱스. 결정론적 k-means (k-means++ 초기화)."""

    def __init__(self, k: int = 8, seed: int = 0, iters: int = 60, tol: float = 1e-7):
        self.k, self.seed, self.iters, self.tol = k, seed, iters, tol

    def _init_centers(self, X, rng):
        n = len(X)
        centers = [X[rng.integers(n)]]
        for _ in range(self.k - 1):
            d2 = np.min(((X[:, None, :] - np.stack(centers)[None, :, :]) ** 2).sum(-1),
                        axis=1)
            tot = d2.sum()
            if tot <= 0:                                  # 전부 동일점 → 임의 보충
                centers.append(X[rng.integers(n)])
                continue
            centers.append(X[rng.choice(n, p=d2 / tot)])
        return np.stack(centers)

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        # 특성 스케일 정규화 — 해싱 인코딩과 수치 특성의 스케일이 크게 달라 그대로 두면
        # 거리 계산이 몇몇 차원에 지배된다.
        self.mu_ = X.mean(axis=0)
        self.sd_ = np.maximum(X.std(axis=0), 1e-6)
        Z = (X - self.mu_) / self.sd_
        rng = np.random.default_rng(self.seed)
        self.k = int(min(self.k, max(1, len(np.unique(Z, axis=0)))))
        C = self._init_centers(Z, rng)
        for _ in range(self.iters):
            lab = self._assign(Z, C)
            newC = C.copy()
            for j in range(len(C)):
                mask = lab == j
                if mask.any():
                    newC[j] = Z[mask].mean(axis=0)
            shift = float(np.abs(newC - C).max())
            C = newC
            if shift < self.tol:
                break
        self.C_ = C
        return self

    @staticmethod
    def _assign(Z, C):
        # (n,k) 거리 — ‖z‖² 는 argmin 에 무관하므로 생략
        return np.argmin(-2.0 * Z @ C.T + (C ** 2).sum(axis=1)[None, :], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, dtype=float) - self.mu_) / self.sd_
        return self._assign(Z, self.C_).astype(np.int64)

    def predict_row(self, x: np.ndarray) -> int:
        return int(self.predict(np.asarray(x, dtype=float)[None, :])[0])

    @property
    def n_clusters(self) -> int:
        return len(self.C_)
