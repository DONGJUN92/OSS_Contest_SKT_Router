"""Cost Mirror — 채점 비용 산식의 정확한 복제 (PROJECT_PLAN.md Phase 1).

현재는 파라미터화된 표준 산식(prefill/decode 분리 과금)을 구현.
Phase 0의 Q3 답변 수령 시 이 파일 하나만 주최측 산식으로 교체하고
tests/test_cost_mirror.py 로 오차 0을 재검증한다.
"""
import numpy as np
from .schema import Dataset


def vote_prefill_mult(cfg: dict, n: int) -> int:
    """[모델×N표] 상자의 프롬프트 프리필 배수 (Phase 13 — 논문 레버 검토 산물).

    기본은 N — 프롬프트를 N회 프리필한다(보수적, 독립 호출 N회 = 프리필 N회 가정).
    Q3(주최측 비용 산식) 확인 시 KV-cache 프리픽스 공유가 사실이면 `cost.prefix_sharing`
    을 켜서 프롬프트를 **1회만** 과금한다 (디코드는 항상 N배 그대로 — 샘플이 다르므로).

    ⚠ 기본 OFF로 두는 이유(적대적 검증 결론): 위험이 **비대칭**이다. 과대비용(현행)은
    투표 상자를 약간 덜 쓰게 할 뿐이나, 공유를 가정했다가 host가 N-prefill로 청구하면
    비용이 20~40% 낮게 잡혀 투표 상자를 과다 개봉 → 비공개셋 실채점에서 예산 초과(특히
    가중치 최고 fast tier). 이득은 조건부(P(host 공유)×이득, N-prefill이면 정확히 0)다.
    상세: docs/reflections/phase13.md, eval/probe_prefix_sharing.py.
    """
    return 1 if (cfg or {}).get("cost", {}).get("prefix_sharing", False) else int(n)


def call_cost(ds: Dataset, i: int, m_idx: int) -> float:
    """쿼리 i에 대해 모델 m_idx를 1회 호출하는 비용."""
    meta = ds.models[ds.model_ids[m_idx]]
    return (
        meta.prefill_price * ds.in_tokens[i] / 1000.0
        + meta.decode_price * ds.out_tokens[i, m_idx] / 1000.0
    )


def cost_matrix(ds: Dataset) -> np.ndarray:
    """(N, M) 전체 비용 행렬 — 정책의 사전 비용 추정과 하네스 과금이 공유하는 단일 원천."""
    prefill = np.array([ds.models[mid].prefill_price for mid in ds.model_ids])
    decode = np.array([ds.models[mid].decode_price for mid in ds.model_ids])
    return (
        ds.in_tokens[:, None] * prefill[None, :] / 1000.0
        + ds.out_tokens * decode[None, :] / 1000.0
    )
