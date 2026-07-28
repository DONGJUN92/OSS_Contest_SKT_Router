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
    """sentence-transformers 백엔드 (설치·캐시 존재 시)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, prompts: list[str]) -> np.ndarray:
        emb = self.model.encode(prompts, batch_size=128, show_progress_bar=False,
                                normalize_embeddings=True)
        return np.asarray(emb, dtype=float)


def get_encoder(backend: str = "hashing"):
    if backend == "st":
        try:
            return STEncoder()
        except Exception as e:                          # 미설치/미캐시 → 해싱 폴백
            print(f"[text_encoder] ST 백엔드 불가({e!r:.80}) → hashing 폴백")
    return HashingEncoder()
