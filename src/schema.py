"""정규화 데이터 스키마 (PROJECT_PLAN.md Phase 1).

실제 챌린지 공개 데이터가 오면 이 스키마로 변환하는 로더만 추가하면 된다.
모든 다운스트림(비용, 하네스, 정책)은 이 스키마에만 의존한다.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ModelMeta:
    model_id: str
    prefill_price: float   # per 1k input tokens
    decode_price: float    # per 1k output tokens


@dataclass
class Dataset:
    """(쿼리 × 모델) 사전 계산 행렬 형태 — 챌린지의 '사전 계산 결과' 채점 구조와 동형.

    quality[i, m]    : 모델 m의 쿼리 i 출력 품질 (이진 0/1; Q4 답변에 따라 연속형 확장)
    out_tokens[i, m] : 모델 m이 쿼리 i에 실제 생성한 토큰 수
    verifier[i, m]   : 검증기가 관측하게 될 잡음 점수 (실데이터에선 검증기 모델이 생성)
    features[i]      : 쿼리 표현 벡터 (실데이터에선 인코더 출력)
    """
    model_ids: list[str]
    models: dict[str, ModelMeta]
    domains: np.ndarray        # (N,) int
    features: np.ndarray       # (N, d) float
    in_tokens: np.ndarray      # (N,) int
    out_tokens: np.ndarray     # (N, M) int
    quality: np.ndarray        # (N, M) float in [0,1]
    verifier: np.ndarray       # (N, M) float in [0,1]
    world: str = "unknown"
    extras: dict = field(default_factory=dict)
    # ★ D36: `features` 를 만든 텍스트 인코더. 배포 어댑터는 런타임에 **텍스트 프롬프트**를
    # 받으므로 적합 때와 **같은** 인코더로 인코딩해야 한다 (다른 인코더 = 다른 특성공간 =
    # 조용히 다른 정책). 초판은 이 연결이 없어 `SubmissionRouter` 가 텍스트 입력에서 죽었다.
    # 합성 세계처럼 특성이 직접 주어지는 경로에서는 None 이어도 무방하다.
    text_encoder: object = None

    @property
    def n(self) -> int:
        return len(self.domains)

    @property
    def m(self) -> int:
        return len(self.model_ids)

    def stratified_folds(self, k: int, seed: int) -> list[np.ndarray]:
        """도메인 층화 k-fold — fold별 테스트 인덱스 배열 반환."""
        rng = np.random.default_rng(seed)
        folds: list[list[int]] = [[] for _ in range(k)]
        for d in np.unique(self.domains):
            idx = np.where(self.domains == d)[0]
            rng.shuffle(idx)
            for j, i in enumerate(idx):
                folds[j % k].append(int(i))
        return [np.sort(np.array(f)) for f in folds]
