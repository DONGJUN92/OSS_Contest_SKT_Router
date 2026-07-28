"""사전 비용 추정 — 호출 전에는 출력 토큰 수를 모른다 (Phase 17, 레드팀 MAJOR).

**결함**: `cost_mirror.cost_matrix` 는 아직 호출하지 않은 모델의 **실제** `out_tokens[i,m]`
으로 비용을 만들고, 정책은 그것을 `Session.costs` 로 그대로 받았다. 즉 예약지수
σ = 1 − λc/p̄ 와 예산 가능성 판정이 **오라클 비용**을 쓰고 있었다.

그런데 대회 상세가 명시한 런타임 정보는
    호출 **전**: 프롬프트 · 모델 프로파일 + 공식 **비용 정책**(단가) · 남은 예산·호출 수
    호출 **후**: 반환 출력 · 실제 토큰 수 · 실제 비용
이다. prefill 은 프롬프트 길이로 사전에 알 수 있지만 **decode 길이는 호출해 봐야 안다** —
특히 `ax-k1-think` 같은 reasoning 모델은 분산이 크다. `config.yaml` 에 `avg_out_tokens` 가
있었지만 합성 생성기(`synth.py`)만 쓰고 라우터는 쓰지 않았다.

이 모듈은 공개 데이터로 **프롬프트 → 모델별 출력 길이**를 학습해 정책에게 추정 비용을
공급하고, 하네스는 실제 비용으로 과금한다(`harness.Session.charge_costs`). 추정이 실제보다
낮으면 호스트가 호출을 거부할 수 있으므로 `cost_margin` 안전마진을 둔다.

계층 3종 (전부 로컬·의존성 0)
  · "oracle"  : 실제 out_tokens (Phase 16 까지의 암묵 가정 — ablation 상한 참조용)
  · "mean"    : 모델별 train 평균 길이 (특성 없이도 즉시 가능한 최소 정직 기준)
  · "ridge"   : 프롬프트 특성 → 모델별 길이 ridge 회귀 (기본)
"""
import numpy as np


class DecodeLengthEstimator:
    """프롬프트 특성 → 모델별 출력 토큰 수. numpy 닫힌형 ridge (의존성 0·결정론)."""

    def __init__(self, mode: str = "ridge", l2: float = 1.0):
        self.mode, self.l2 = mode, l2

    def fit(self, features: np.ndarray, out_tokens: np.ndarray):
        y = np.asarray(out_tokens, dtype=float)                    # (N, M)
        self.mean_ = y.mean(axis=0)                                # (M,)
        self.floor_ = np.maximum(np.percentile(y, 5, axis=0), 1.0)
        if self.mode == "ridge":
            X = np.hstack([np.asarray(features, dtype=float),
                           np.ones((len(y), 1))])                  # (N, d+1)
            A = X.T @ X + self.l2 * np.eye(X.shape[1])
            A[-1, -1] -= self.l2                                   # 절편은 벌하지 않는다
            self.W_ = np.linalg.solve(A, X.T @ y)                  # (d+1, M)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        n = len(features)
        if self.mode == "mean":
            return np.tile(self.mean_, (n, 1))
        X = np.hstack([np.asarray(features, dtype=float), np.ones((n, 1))])
        return np.maximum(X @ self.W_, self.floor_[None, :])


def estimated_cost_matrix(ds, estimator=None, mode: str = "ridge",
                          train_idx: np.ndarray | None = None) -> np.ndarray:
    """(N, M) **정책이 호출 전에 볼 수 있는** 비용 행렬.

    prefill 은 실제 in_tokens (프롬프트 길이는 사전에 안다), decode 는 추정 길이.
    estimator 를 주지 않으면 train_idx 로 즉석 적합한다 (없으면 전체 — ablation 전용).
    mode="oracle" 이면 실제 비용 행렬과 같아진다.
    """
    from .cost_mirror import cost_matrix
    if mode == "oracle":
        return cost_matrix(ds)
    if estimator is None:
        idx = np.arange(ds.n) if train_idx is None else np.asarray(train_idx)
        estimator = DecodeLengthEstimator(mode=mode).fit(ds.features[idx],
                                                        ds.out_tokens[idx])
    prefill = np.array([ds.models[mid].prefill_price for mid in ds.model_ids])
    decode = np.array([ds.models[mid].decode_price for mid in ds.model_ids])
    out_hat = estimator.predict(ds.features)
    return (ds.in_tokens[:, None] * prefill[None, :] / 1000.0
            + out_hat * decode[None, :] / 1000.0)


def estimation_report(ds, est_cmat: np.ndarray) -> dict:
    """추정 비용 대 실제 비용의 오차 요약 — 마진 선택 근거를 수치로 남긴다."""
    from .cost_mirror import cost_matrix
    true = cost_matrix(ds)
    rel = (est_cmat - true) / np.maximum(true, 1e-12)
    return {"mean_rel_err": round(float(rel.mean()), 4),
            "mae_rel": round(float(np.abs(rel).mean()), 4),
            "p90_underestimate": round(float(-np.percentile(rel, 10)), 4),
            "frac_underestimated": round(float((rel < 0).mean()), 4)}
