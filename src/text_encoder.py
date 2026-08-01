"""TextEncoder — 프롬프트 텍스트 → 특성 벡터 (Phase 8, 심사 권고 2 / 백로그 B3).

파이프라인 접속 계약: Dataset.features[i]를 생성 — IRTEncoder(선형 헤드)가 무변경 소비.
비지도 변환이므로 fold 누수 없음 (라벨 미사용).

백엔드 2종:
  hashing  : 문자 3-gram signed feature hashing + 수치 특성. 다운로드·의존성 0,
             결정론, 오프라인 규칙 완전 준수. 기본값.
  st       : sentence-transformers (all-MiniLM-L6-v2, 최초 1회 로컬 캐시 다운로드).
             실배포에서는 다국어 모델(bge-m3 등)로 교체 — 어댑터 동일.
"""
import hashlib
import re

import numpy as np


class HashingEncoder:
    def __init__(self, dim: int = 96):
        self.dim = dim

    def _hash_vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim)
        t = text.lower()
        for i in range(len(t) - 2):
            g = t[i:i + 3]
            h = int(hashlib.blake2b(g.encode(), digest_size=8).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def _numeric(self, text: str) -> np.ndarray:
        nums = re.findall(r"\d+", text)
        return np.array([
            min(len(text) / 300.0, 3.0),               # 길이
            min(len(nums) / 12.0, 3.0),                # 숫자 개수 (정렬 리스트 길이 등)
            min(max((len(x) for x in nums), default=0) / 6.0, 3.0),  # 최대 자릿수
            min(len(text.split()) / 60.0, 3.0),        # 단어 수
        ])

    def encode(self, prompts: list[str]) -> np.ndarray:
        return np.stack([np.concatenate([self._hash_vec(p), self._numeric(p)])
                         for p in prompts])


class STEncoder:
    """sentence-transformers 백엔드 (설치·캐시 존재 시).

    ★ D65 (외부 레드팀 2026-08-01): 초판은 모델명이 **하드코딩**돼 있어
    `get_encoder("st")` 가 항상 `all-MiniLM-L6-v2` — **영어 전용 모델**을 만들었다.
    SKT 지정과제이고 후보가 A.X 계열이면 비공개셋 프롬프트는 한국어일 공산이 크다.
    docstring 은 "실배포에서는 다국어 모델(bge-m3 등)로 교체"라고 적어 두고 **교체할
    인자를 노출하지 않은 상태**였다 — D36(`text_encoder` 미대입)·D63(비용 규약)과 같은
    "문서는 인정했는데 코드 경로가 없다" 계열이다.
    """

    #: 다국어 우선 후보 (설치·캐시된 것 중 첫 번째를 쓴다). 한국어를 포함한다.
    MULTILINGUAL = ("BAAI/bge-m3",
                    "intfloat/multilingual-e5-small",
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name or self.DEFAULT
        self.model = SentenceTransformer(self.model_name)

    def encode(self, prompts: list[str]) -> np.ndarray:
        emb = self.model.encode(prompts, batch_size=128, show_progress_bar=False,
                                normalize_embeddings=True)
        return np.asarray(emb, dtype=float)


def get_encoder(backend: str = "hashing", model_name: str | None = None,
                dim: int | None = None):
    """인코더 팩토리. `backend` 는 `"이름"` 또는 `"이름:인자"` 형태를 받는다.

        hashing            문자 3-gram 해싱 (기본, 의존성 0)
        hashing:256        해싱 차원 지정 — 96 은 충돌이 많다 (D65)
        st                 sentence-transformers 기본 모델 (영어)
        st:multilingual    설치된 다국어 모델 중 첫 번째 (한국어 대응)
        st:<model_name>    임의 모델 지정

    ST 백엔드는 **네트워크 다운로드가 필요할 수 있다.** 챌린지 규칙은 *라우터 추론 시*
    외부 호출을 금지하며 인코더 가중치는 오프라인 캐시에서 로드되지만, 심사 환경에서
    캐시가 없으면 실패하므로 **기본값은 여전히 `hashing`** 이다. 어느 쪽이 실제로 나은지는
    `src/encoder_scan.py` 가 데이터로 판정한다.
    """
    if ":" in backend:
        backend, arg = backend.split(":", 1)
        if backend == "hashing":
            dim = int(arg)
        else:
            model_name = arg
    if backend == "st":
        names = list(STEncoder.MULTILINGUAL) if model_name == "multilingual" \
            else [model_name]
        errs = []
        for nm in names:
            try:
                return STEncoder(nm)
            except Exception as e:
                errs.append(f"{nm}: {e!r:.60}")
        print(f"[text_encoder] ST 백엔드 불가 → hashing 폴백. 시도: {'; '.join(errs)}")
    return HashingEncoder(dim=int(dim) if dim else 96)
