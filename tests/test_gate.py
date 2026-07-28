"""배포 게이트 회귀 (Phase 18).

게이트가 고정해야 하는 성질
  ① 끝까지 돌아 **배포 라우터 객체**를 만들고, 그 라우터가 제출 어댑터로 배포 가능하다
  ② 결정은 train 안에서만 이뤄진다 (테스트 인덱스 미접촉)
  ③ 대안 채택은 `max(절대 margin, se_k × 쌍대 표준오차)` 를 넘어야 한다 —
     첫 구현이 margin 0.01 만 봐서 SE 0.031 짜리 잡음으로 정책을 바꿨다 (D21 재발)
  ④ 예측층은 결정권이 없다 (AUC 임계값을 어떻게 흔들어도 chosen 이 바뀌지 않는다)
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from tests.test_packaging import _tiny_dataset, MIN_CFG


def _ds():
    ds = _tiny_dataset(n=420, seed=3)
    return ds, np.arange(120, 420), np.arange(0, 120)


def test_gate_runs_and_returns_deployable_routers():
    from src.gate import run_gate
    from src.submission import SubmissionRouter
    from src.cost_mirror import cost_matrix
    from src.harness import tier_budgets
    ds, tr, te = _ds()
    dec, routers = run_gate(ds, MIN_CFG, tr, tiers=["fast"], meta=None,
                            lpb_kwargs={"use_domain": False})
    assert set(routers) == {"fast"}
    assert dec.chosen["fast"] in ("lpb", "learned", "cascade_routing")
    assert dec.deploy in ("lpb", "selective")
    # 배포 라우터가 push 규약으로 유효 액션 열을 내는가 (규칙 불변식 포함)
    cmat = cost_matrix(ds)
    budget = tier_budgets(ds, te, MIN_CFG)["fast"]
    sub = SubmissionRouter({"fast": routers["fast"]}, ds.model_ids,
                           observe=lambda p, mid, rec: ds.verifier[rec["_row"], rec["_m"]])
    sub.begin_tier("fast", budget=budget, n_queries=len(te))
    for i in te[:25]:
        hist, called, spent = [], [], 0.0
        for _ in range(ds.m + 2):
            act = sub.step(ds.features[i], "fast", hist, cmat[i], budget - spent, domain=0)
            if act.kind in ("answer", "abstain"):
                if act.kind == "answer":
                    assert act.model_id in called, "규칙 위반: 미호출 모델을 답으로 지명"
                break
            m = sub.index[act.model_id]
            spent += cmat[i, m]
            called.append(act.model_id)
            hist.append({"model_id": act.model_id, "output": None, "_row": i, "_m": m})
        else:
            raise AssertionError("어댑터가 종료하지 않음")
        sub.end_query(spent, "fast")


def test_gate_records_required_margin_from_paired_se():
    """SE 문턱이 실제로 기록되고, 절대 margin 보다 클 수 있어야 한다."""
    from src.gate import run_gate
    ds, tr, te = _ds()
    dec, _ = run_gate(ds, MIN_CFG, tr, tiers=["fast"], meta=None,
                      lpb_kwargs={"use_domain": False})
    s = dec.per_tier_scores["fast"]
    assert s["_paired_se"] >= 0.0
    assert s["_required_margin"] >= min(0.01, s["_paired_se"])
    # 두 값이 각각 소수 4자리로 반올림돼 기록되므로 허용오차는 2×5e-5 여야 한다
    # (초판은 1e-9 를 썼고 반올림 차이 1e-4 로 실패했다 — 테스트 자체의 버그였다)
    assert abs(s["_observed_gap"] - (s[s["_best_alt"]] - s["lpb"])) <= 2e-4
    # 요구 문턱을 넘지 못했으면 반드시 lpb 여야 한다 (잡음 선택 금지)
    if s["_observed_gap"] <= s["_required_margin"]:
        assert dec.chosen["fast"] == "lpb"


def test_prediction_layer_has_no_authority():
    """임계값을 극단으로 흔들어도 결정층 선택은 불변이어야 한다."""
    from src.gate import run_gate
    ds, tr, te = _ds()
    picks = []
    for thr in (0.0, 1.0):
        dec, _ = run_gate(ds, MIN_CFG, tr, tiers=["fast"], meta=None, threshold=thr,
                          lpb_kwargs={"use_domain": False})
        picks.append((dec.predicted, dec.chosen["fast"]))
    assert picks[0][0] != picks[1][0], "임계값이 예측층을 바꿔야 한다 (테스트 전제)"
    assert picks[0][1] == picks[1][1], "예측층이 결정을 바꿨다 — 결정권이 새고 있다"


def test_infinite_se_k_pins_lpb():
    """se_k=∞ 면 어떤 격차도 문턱을 못 넘으므로 항상 LPB (기본 정책 고정 확인)."""
    from src.gate import run_gate
    ds, tr, te = _ds()
    dec, routers = run_gate(ds, MIN_CFG, tr, tiers=["fast"], meta=None, se_k=1e9,
                            lpb_kwargs={"use_domain": False})
    assert dec.chosen["fast"] == "lpb" and dec.deploy == "lpb"
    assert type(routers["fast"]).__name__ == "LPBRouter"
