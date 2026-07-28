"""Cost Mirror 단위 테스트 — 완료 기준: 산식 오차 0 (PROJECT_PLAN.md Phase 1)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import yaml
from src.synth import make_world
from src.cost_mirror import call_cost, cost_matrix

CFG = yaml.safe_load(open(pathlib.Path(__file__).resolve().parents[1] / "config.yaml", encoding="utf-8"))


def test_scalar_matches_hand_computation():
    ds = make_world(CFG, "irt", seed=7)
    i, m = 3, 2
    meta = ds.models[ds.model_ids[m]]
    expected = meta.prefill_price * ds.in_tokens[i] / 1000 + meta.decode_price * ds.out_tokens[i, m] / 1000
    assert abs(call_cost(ds, i, m) - expected) == 0.0


def test_matrix_matches_scalar_loop_exactly():
    ds = make_world(CFG, "specialist", seed=7)
    cmat = cost_matrix(ds)
    rng = np.random.default_rng(0)
    for _ in range(200):
        i, m = int(rng.integers(ds.n)), int(rng.integers(ds.m))
        assert cmat[i, m] == call_cost(ds, i, m), f"mismatch at ({i},{m})"


def test_costs_positive_and_ordered_by_model_size():
    ds = make_world(CFG, "irt", seed=7)
    cmat = cost_matrix(ds)
    assert (cmat > 0).all()
    # 평균 비용이 모델 크기(가격) 순서와 일치해야 함
    mean_costs = cmat.mean(axis=0)
    assert (np.diff(mean_costs) > 0).all(), "config의 모델 순서는 비용 오름차순이어야 함"


def test_vote_prefill_mult_helper():
    """Q3 대비 prefix-sharing 토글 (Phase 13). 기본 OFF = N배, ON = 1배."""
    from src.cost_mirror import vote_prefill_mult
    assert vote_prefill_mult({}, 5) == 5                       # 키 없음 → 보수적 기본(N)
    assert vote_prefill_mult({"cost": {"prefix_sharing": False}}, 5) == 5
    assert vote_prefill_mult({"cost": {"prefix_sharing": True}}, 5) == 1    # 공유 → 1
    assert vote_prefill_mult({"cost": {"prefix_sharing": True}}, 1) == 1


def test_prefix_sharing_toggles_vote_box_prefill():
    """토글이 [모델×N표] 상자의 프리필만 바꾸고 디코드는 불변임을 고정 (하위호환 포함)."""
    from src.synth_ext import extend_with_votes
    base = make_world(CFG, "irt", seed=7)
    off = extend_with_votes(base, CFG)                         # 기본 OFF (현행)
    on = extend_with_votes(base, dict(CFG, cost={"prefix_sharing": True}))
    for mid in base.model_ids:
        base_pf = base.models[mid].prefill_price
        for N in (1, 3, 5):
            vid = mid if N == 1 else f"{mid}#v{N}"
            assert off.models[vid].prefill_price == base_pf * N   # OFF: 프리필 N배(보수적)
            assert on.models[vid].prefill_price == base_pf        # ON: 프리필 1배(공유)
            assert off.models[vid].decode_price == on.models[vid].decode_price \
                == base.models[mid].decode_price                  # 디코드는 토글 무관


if __name__ == "__main__":
    test_scalar_matches_hand_computation()
    test_matrix_matches_scalar_loop_exactly()
    test_costs_positive_and_ordered_by_model_size()
    test_vote_prefill_mult_helper()
    test_prefix_sharing_toggles_vote_box_prefill()
    print("cost_mirror: 5/5 tests passed (오차 0 + prefix-sharing 토글)")
