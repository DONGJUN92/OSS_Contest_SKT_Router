"""Phase 21 회귀 — 외부 레드팀(2026-08-01) 지적 D64~D67.

공통 성질(그리고 이 저장소가 반복해 배운 것): **결함이 있어도 숫자가 이상해지지 않는다.**
  · 게이트가 아무 진행도 안 알려도 결과는 나온다 (끝까지 기다리면)
  · 인코더가 96차원이어도 점수는 나온다 (조금 낮을 뿐)
  · 채택 문턱이 과보수적이어도 유효한 정책이 나온다 (조금 나쁠 뿐)
그래서 전부 "돌려 보면 안다"가 아니라 **불변식으로 고정**한다.
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from tests.test_packaging import _tiny_dataset, MIN_CFG


# ─────────────── D64: 게이트 운영성 (진행 표시 · 부분 저장 · quick) ───────────────

def test_gate_reports_progress_for_every_fold():
    """수령 당일 운영자가 **끝날지 여부를 알 수 있어야** 한다.

    900초 무출력이 문제였다. 진행 콜백이 fold 마다 정확히 한 번 발동해야 한다
    (D32 규칙: 기구의 발동을 증명 가능하게).
    """
    from src.gate import run_gate
    ds = _tiny_dataset(n=300, seed=11)
    seen = []
    run_gate(ds, MIN_CFG, np.arange(60, 300), tiers=["fast"], meta=None,
             n_splits=2, n_repeats=2, progress=lambda s, i: seen.append((s, i)))
    folds = [i for s, i in seen if s == "fold"]
    assert len(folds) == 4, f"2-fold × 2반복 = 4회여야 한다 (got {len(folds)})"
    assert any(s == "tier_done" for s, _ in seen)
    assert any(s == "verifier" for s, _ in seen)


def test_gate_writes_a_partial_artifact_before_it_finishes(tmp_path):
    """중간에 끊겨도 이미 끝난 tier 의 비교 결과는 살아 있어야 한다."""
    import json
    from src.gate import run_gate
    ds = _tiny_dataset(n=300, seed=11)
    out = tmp_path / "gate.json"
    partials = []

    def spy(stage, info):
        if stage == "tier_done" and out.exists():
            partials.append(json.load(open(out, encoding="utf-8")))

    run_gate(ds, MIN_CFG, np.arange(60, 300), tiers=["fast", "balanced"], meta=None,
             n_splits=2, n_repeats=1, out_path=str(out), progress=spy)
    # 첫 tier 가 끝난 직후에는 아직 아무 파일도 없다(그 tier 의 dump 가 콜백 뒤에 온다) —
    # 그러나 두 번째 tier 가 끝날 때는 첫 tier 판정이 디스크에 있어야 한다.
    assert partials, "tier 경계에서 부분 아티팩트가 있어야 한다"
    assert partials[-1]["status"] == "partial"
    assert "fast" in partials[-1]["chosen_so_far"]
    final = json.load(open(out, encoding="utf-8"))
    assert final.get("deploy") in ("lpb", "selective"), "마지막에는 최종 판정으로 덮인다"


def test_gate_decision_is_unchanged_by_the_progress_callback():
    """진행 표시는 **보고**다 — 판정에 영향을 주면 안 된다."""
    from src.gate import run_gate
    ds = _tiny_dataset(n=300, seed=11)
    kw = dict(tiers=["fast"], meta=None, n_splits=2, n_repeats=1)
    a, _ = run_gate(ds, MIN_CFG, np.arange(60, 300), **kw)
    b, _ = run_gate(ds, MIN_CFG, np.arange(60, 300), progress=lambda s, i: None, **kw)
    assert a.chosen == b.chosen and a.per_tier_scores == b.per_tier_scores


# ─────────────── D65: 인코더 선택층 (예측기 축을 데이터가 고른다) ───────────────

def test_encoder_scan_picks_by_paired_threshold_and_reports_a_table():
    from src.encoder_scan import scan_encoders
    from src.text_encoder import get_encoder
    ds = _tiny_dataset(n=300, seed=4)
    prompts = [f"질문 {i}: 값 {i * 7 % 53} 을 계산하시오" for i in range(ds.n)]
    cands = {"hashing96": get_encoder("hashing:96"), "hashing256": get_encoder("hashing:256")}
    rep = scan_encoders(prompts, ds, np.arange(60, ds.n), cands, n_splits=2)
    assert rep.baseline == "hashing96" and rep.best in cands
    assert len(rep.rows) == 2 and all(np.isfinite(r["logloss"]) for r in rep.rows)
    assert "인코더 선택층" in rep.table()
    alt = [r for r in rep.rows if r["name"] != rep.baseline][0]
    if rep.changed:                       # 바꿨다면 반드시 문턱을 넘었어야 한다
        assert alt["delta"] > alt["required"]


def test_encoder_scan_drops_duplicate_arms_instead_of_reporting_zero_effect():
    """★ D55 의 교훈을 이 축에도 — 두 팔의 특성이 같으면 '효과 없음'이 아니라 '실험 없음'."""
    from src.encoder_scan import scan_encoders
    from src.text_encoder import get_encoder
    ds = _tiny_dataset(n=240, seed=4)
    prompts = [f"q{i}" for i in range(ds.n)]
    cands = {"a": get_encoder("hashing:96"), "b": get_encoder("hashing:96")}   # 동일 팔
    rep = scan_encoders(prompts, ds, np.arange(60, ds.n), cands, n_splits=2)
    assert len(rep.rows) == 1, "동일 팔은 제외돼야 한다"
    assert any("동일" in n for n in rep.notes)


def test_get_encoder_accepts_dim_and_model_name():
    from src.text_encoder import get_encoder, HashingEncoder
    e = get_encoder("hashing:256")
    assert isinstance(e, HashingEncoder) and e.encode(["가나다 abc"]).shape[1] == 260
    assert get_encoder("st:definitely-not-a-real-model") is not None   # 폴백은 조용히


def test_apply_encoder_carries_the_encoder_to_the_deployment_path():
    """D36 계약 유지 — 선택된 인코더가 어댑터까지 흘러야 한다."""
    from src.encoder_scan import apply_encoder
    from src.text_encoder import get_encoder
    ds = _tiny_dataset(n=120, seed=4)
    apply_encoder(ds, [f"p{i}" for i in range(ds.n)], get_encoder("hashing:128"))
    assert ds.features.shape[1] == 132 and ds.text_encoder is not None


# ─────────────── D66: 채택 규칙이 표준오차를 본다 ───────────────

def test_selective_router_records_paired_se_not_just_means():
    """초판은 평균만 비교했다 — 게이트(D26)가 이미 배운 것을 여기서는 안 쓰고 있었다."""
    from baselines.policies import SelectiveRouter
    ds = _tiny_dataset(n=360, seed=6)
    sr = SelectiveRouter(MIN_CFG, 1, seed=0, use_domain=False).fit(ds, np.arange(60, 360),
                                                                  "fast")
    d = sr.sel_detail
    assert {"best_alt", "gap", "paired_se", "required", "margin_mode"} <= set(d)
    assert d["margin_mode"] == "absolute" and d["required"] >= 0.0
    assert sr.chosen in ("lpb", "learned", "cascade_routing")


def test_paired_se_mode_never_requires_more_than_absolute_mode():
    """사전 등록된 성질: `paired_se` 는 절대 바닥을 없앤 것이므로 요구치가 더 낮거나 같다."""
    from baselines.policies import SelectiveRouter
    ds = _tiny_dataset(n=360, seed=6)
    a = SelectiveRouter(MIN_CFG, 1, seed=0, use_domain=False,
                        margin_mode="absolute").fit(ds, np.arange(60, 360), "fast")
    b = SelectiveRouter(MIN_CFG, 1, seed=0, use_domain=False,
                        margin_mode="paired_se").fit(ds, np.arange(60, 360), "fast")
    assert b.sel_detail["required"] <= a.sel_detail["required"] + 1e-12
    assert a.sel_detail["gap"] == pytest.approx(b.sel_detail["gap"])   # 같은 분할·같은 후보


def test_unknown_margin_mode_fails_loudly():
    from baselines.policies import SelectiveRouter
    with pytest.raises(ValueError, match="margin_mode"):
        SelectiveRouter(MIN_CFG, 1, margin_mode="whatever")


def test_selective_router_is_still_deployable_through_the_adapter():
    """규칙을 바꿔도 제출 경로는 유지돼야 한다 (Phase 17 이 세운 불변식)."""
    from baselines.policies import SelectiveRouter
    from src.submission import SubmissionRouter
    from src.cost_mirror import cost_matrix
    ds = _tiny_dataset(n=360, seed=6)
    sr = SelectiveRouter(MIN_CFG, 1, seed=0, use_domain=False,
                         margin_mode="paired_se").fit(ds, np.arange(60, 360), "fast")
    sub = SubmissionRouter({"fast": sr}, ds.model_ids, observe=lambda p, m, r: 0.5)
    act = sub.step(ds.features[0], "fast", [], cost_matrix(ds)[0], 1e6)
    assert act.kind in ("call", "answer", "abstain")


# ─────────── D70: 인코더가 배포 경로에 실려 있는가 (소비 지점 계약) ───────────

def test_demo_dataset_carries_its_encoder_into_the_deployment_path():
    """★ D70 — `demo.py` 는 인코더를 익명으로 쓰고 버려 텍스트 프롬프트에서 죽었다.

    9 Phase 동안 무증상이었던 이유: **오프라인 경로는 특성 행렬만 쓴다.** 인코더 객체는
    배포에서만 필요하고, 없어도 아무 점수도 이상해지지 않는다 (Phase 20 §0 의 성질).
    그래서 "점수가 맞는가"가 아니라 **"텍스트가 관통하는가"** 를 직접 고정한다.

    저장소 전체 21개 소비 지점을 같은 형태로 고쳤으므로, 이 테스트는 그 계약의 대표 표본이다.
    """
    import subprocess, sys as _sys, pathlib as _pl
    root = _pl.Path(__file__).resolve().parents[1]
    src = (root / "demo.py").read_text(encoding="utf-8")
    assert "ds.text_encoder = _enc" in src, (
        "demo.py 가 인코더를 데이터셋에 실어야 한다 — 없으면 시연영상 촬영 대상이 "
        "챌린지 실입력(텍스트 프롬프트)에서 죽는다")


def test_text_prompt_survives_the_full_deploy_path_with_price_metadata():
    """텍스트 프롬프트 + 단가 메타데이터가 **동시에** 관통하는가 (D63 × D70).

    두 결함이 같은 경로에 있었으므로 함께 고정한다 — 하나만 고치면 다른 하나가 남는다.
    """
    from src.router import LPBRouter
    from src.submission import SubmissionRouter
    from src.text_encoder import get_encoder
    enc = get_encoder("hashing:128")
    ds = _tiny_dataset(n=360, seed=9)
    ds.features = enc.encode([f"문항 {i}: 값을 구하시오" for i in range(ds.n)])
    ds.text_encoder = enc                       # D70 계약
    r = LPBRouter(MIN_CFG, 1, seed=0, use_domain=False,
                  cost_mode="ridge").fit(ds, np.arange(120, ds.n), "fast")
    sub = SubmissionRouter({"fast": r}, ds.model_ids, observe=lambda p, m, rec: 0.6)
    price = {mid: {"prefill_price": ds.models[mid].prefill_price,
                   "decode_price": ds.models[mid].decode_price} for mid in ds.model_ids}
    hist, called = [], []
    for _ in range(ds.m + 1):
        act = sub.step("한국어 텍스트 프롬프트입니다", "fast", hist, price, 1e6,
                       remaining_calls=ds.m - len(hist))
        if act.kind != "call":
            break
        called.append(act.model_id)
        hist.append({"model_id": act.model_id, "output": "..."})
    assert sub.cost_path == "price_policy", "단가 경로가 발동해야 한다"
    assert act.kind in ("answer", "abstain")
    if act.kind == "answer":
        assert act.model_id in called          # 챌린지 규칙 불변식
