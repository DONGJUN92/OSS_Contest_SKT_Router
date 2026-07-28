"""프롬프트 특성 — 난이도 예측용 풍부한 표현 (Phase 19).

**왜 필요한가 (측정된 배분 오류)**: `eval/probe_predictor_ceiling.py` 기준 이 라우터가
가져가지 못한 달성폭은 46% 인데, 결정 구조 차이는 달성폭의 1.4~3.5% 에 불과하다. 남은 몫의
대부분은 "프롬프트만 보고 어느 모델이 맞힐지" 예측하는 능력에 있다. 그런데 기존
`text_encoder.HashingEncoder` 는 **문자 3-gram 96차 signed hashing + 수치 4개**뿐이었다 —
단어 정보도, idf 가중도, 문제 유형 신호도 없다.

이 모듈은 의존성 0·결정론을 유지하면서 세 블록을 쌓는다.
  ① 문자 n-gram (3,4)  signed hashing + idf   — 표기·형태 신호
  ② 단어 n-gram (1,2)  signed hashing + idf   — 어휘·주제 신호
  ③ 구조 특성 (20개)                          — 길이·수치·보기·코드·연산자·한글비율 등
블록마다 L2 정규화해 스케일이 큰 블록이 거리 계산을 지배하지 않게 한다.

idf 는 **라벨을 쓰지 않는 비지도 통계**이므로 fold 누수가 없다 (`text_encoder` 와 같은 논리).
챌린지의 오프라인 규칙도 그대로 만족한다 — 다운로드·네트워크 없음.
"""
import hashlib
import re

import numpy as np

_WORD_RE = re.compile(r"[0-9a-zA-Z가-힣_]+")
_OPT_RE = re.compile(r"(?:^|\n)\s*\(?[A-Ea-e1-5][).:]\s")
_CODE_RE = re.compile(r"\b(def|class|return|import|function|SELECT|for|while)\b")
_MATH_RE = re.compile(r"[+\-*/=<>^%]|\b(sum|product|derivative|integral|mod)\b")
_Q_RE = re.compile(r"(which|what|how many|how much|why|다음 중|무엇|얼마|몇)", re.IGNORECASE)


def _h(token: str, dim: int) -> tuple[int, float]:
    """signed feature hashing — 버킷과 부호."""
    d = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    v = int.from_bytes(d, "big")
    return v % dim, (1.0 if (v >> 63) & 1 else -1.0)


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class RichEncoder:
    """프롬프트 → 특성 벡터. `fit(prompts)` 로 idf 를 잡고 `encode(prompts)` 로 변환한다.

    dim_char / dim_word 를 0 으로 두면 해당 블록을 끈다 (ablation 용).
    """

    def __init__(self, dim_char: int = 160, dim_word: int = 160,
                 char_ns=(3, 4), word_ns=(1, 2), use_idf: bool = True):
        self.dim_char, self.dim_word = dim_char, dim_word
        self.char_ns, self.word_ns = tuple(char_ns), tuple(word_ns)
        self.use_idf = use_idf
        self.idf_c = None
        self.idf_w = None

    # ---------------- 토큰 ----------------

    @staticmethod
    def _char_grams(text: str, ns):
        t = text.lower()
        for n in ns:
            for i in range(max(len(t) - n + 1, 0)):
                yield t[i:i + n]

    @staticmethod
    def _word_grams(text: str, ns):
        ws = _WORD_RE.findall(text.lower())
        for n in ns:
            for i in range(max(len(ws) - n + 1, 0)):
                yield " ".join(ws[i:i + n])

    # ---------------- idf ----------------

    def fit(self, prompts):
        n = max(len(prompts), 1)
        if self.use_idf and self.dim_char:
            df = np.zeros(self.dim_char)
            for p in prompts:
                seen = {_h(g, self.dim_char)[0] for g in self._char_grams(p, self.char_ns)}
                for b in seen:
                    df[b] += 1.0
            self.idf_c = np.log((1.0 + n) / (1.0 + df)) + 1.0
        if self.use_idf and self.dim_word:
            df = np.zeros(self.dim_word)
            for p in prompts:
                seen = {_h(g, self.dim_word)[0] for g in self._word_grams(p, self.word_ns)}
                for b in seen:
                    df[b] += 1.0
            self.idf_w = np.log((1.0 + n) / (1.0 + df)) + 1.0
        return self

    # ---------------- 구조 특성 ----------------

    @staticmethod
    def _struct(text: str) -> np.ndarray:
        t = text
        low = t.lower()
        words = _WORD_RE.findall(low)
        nums = re.findall(r"\d+", t)
        digits = sum(c.isdigit() for c in t)
        alpha = sum(c.isalpha() for c in t)
        hangul = sum("가" <= c <= "힣" for c in t)
        nw = max(len(words), 1)
        return np.array([
            min(len(t) / 300.0, 4.0),                                   # 길이
            min(nw / 60.0, 4.0),                                        # 단어 수
            min(len(nums) / 12.0, 4.0),                                 # 수 개수
            min(max((len(x) for x in nums), default=0) / 6.0, 4.0),      # 최대 자릿수
            digits / max(len(t), 1),                                    # 숫자 비율
            hangul / max(len(t), 1),                                    # 한글 비율
            1.0 - (alpha + digits) / max(len(t), 1),                    # 비영숫자 비율
            min(sum(len(w) for w in words) / nw / 8.0, 3.0),            # 평균 단어 길이
            len(set(words)) / nw,                                       # 어휘 다양성
            min(t.count("\n") / 8.0, 3.0),                              # 줄 수
            float(bool(_OPT_RE.search(t))),                             # 보기 목록 존재
            min(len(_OPT_RE.findall(t)) / 5.0, 2.0),                    # 보기 개수
            float(bool(_CODE_RE.search(t))),                            # 코드 신호
            min(len(_MATH_RE.findall(t)) / 6.0, 3.0),                   # 연산자 밀도
            float("?" in t),                                            # 물음표
            float(bool(_Q_RE.search(t))),                               # 질문 유형어
            min(t.count(",") / 10.0, 3.0),                              # 콤마 (목록성)
            min(t.count(":") / 4.0, 2.0),
            float(bool(re.search(r"\bstep by step|차례로|단계", low))),   # 다단계 요구
            min(len(re.findall(r"[.!?]", t)) / 6.0, 3.0),               # 문장 수
        ])

    STRUCT_DIM = 20

    def encode(self, prompts) -> np.ndarray:
        rows = []
        for p in prompts:
            blocks = []
            if self.dim_char:
                v = np.zeros(self.dim_char)
                for g in self._char_grams(p, self.char_ns):
                    b, s = _h(g, self.dim_char)
                    v[b] += s
                if self.idf_c is not None:
                    v = v * self.idf_c
                blocks.append(_l2(v))
            if self.dim_word:
                v = np.zeros(self.dim_word)
                for g in self._word_grams(p, self.word_ns):
                    b, s = _h(g, self.dim_word)
                    v[b] += s
                if self.idf_w is not None:
                    v = v * self.idf_w
                blocks.append(_l2(v))
            blocks.append(self._struct(p))
            rows.append(np.concatenate(blocks))
        return np.stack(rows)

    @property
    def dim(self) -> int:
        return self.dim_char + self.dim_word + self.STRUCT_DIM


def get_encoder(backend: str = "rich", prompts=None, **kw):
    """인코더 팩토리 — "rich"(기본, Phase 19) · "hashing"(구 기본) · "st"(임베딩 백엔드).

    "rich" 는 idf 를 위해 코퍼스가 필요하므로 prompts 를 받으면 즉시 fit 한다.
    """
    if backend == "rich":
        enc = RichEncoder(**kw)
        return enc.fit(prompts) if prompts is not None else enc
    from .text_encoder import get_encoder as legacy
    return legacy(backend)
