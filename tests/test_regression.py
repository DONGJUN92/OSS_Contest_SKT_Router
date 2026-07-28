"""회귀 테스트 스위트 (B8) — 핵심 불변량을 pytest로 고정.

실데이터 적용 중 코드 변경이 이론 보증·규칙 준수를 깨뜨리지 않도록 방어한다.
전체 < 60초 목표 (빠른 축소판 검증).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))


# ---------- 1. Weitzman 엔진 = DP 전역 최적 (Phase 3a 축소판) ----------

def test_engine_matches_dp_optimum():
    from run_phase3 import dp_value, engine_value
    rng = np.random.default_rng(1)
    for _ in range(60):
        M = int(rng.integers(2, 5))
        p = rng.uniform(0.2, 0.9, size=M)
        c = rng.uniform(0.01, 0.3, size=M)
        lam = float(rng.uniform(0.5, 2.0))
        gap = dp_value(tuple(p), tuple(c), lam) - engine_value(p, c, lam)
        assert abs(gap) < 1e-9, f"DP 격차 {gap}"


# ---------- 2. 하네스 규칙 강제 (챌린지 규칙 동형성) ----------

def test_session_refuses_unaffordable_and_charges_once():
    from src.harness import Session
    s = Session(costs=np.array([1.0, 5.0]), verifier_row=np.array([0.3, 0.9]),
                remaining_budget=1.5)
    assert s.call(0) is not None and s.spent == 1.0
    assert s.call(1) is None                       # 예산 부족 → 거부
    assert s.call(0) is not None and s.spent == 1.0  # 재열람 무과금


def test_final_answer_must_be_called_and_budget_never_exceeded():
    from src.synth import make_world
    from src.harness import run_tier, tier_budgets
    from baselines.policies import AllModel
    ds = make_world(CFG, "irt", seed=7)
    idx = np.arange(300)
    budget = tier_budgets(ds, idx, CFG)["fast"]
    r = run_tier(ds, idx, AllModel(4, "all-largest"), "fast", budget)
    assert r.total_cost <= budget + 1e-9           # zero_quality 의미론: 초과 불가


# ---------- 3. λ 단조성 (지수 정책의 기본 성질) ----------

def test_lambda_monotonic_calls():
    from src.synth import make_world
    from src.harness import run_tier, tier_budgets
    from src.irt.mml import fit_mml
    from src.engine.pandora import NoiseModel
    from phase2_stages import IRTEncoder
    from run_phase3 import make_pandora
    ds = make_world(CFG, "irt", seed=7)
    idx = np.arange(300)
    tr = np.arange(300, 900)
    irt = fit_mml(ds.quality[tr], ds.domains[tr], 4, per_domain_b=True, steps=150)
    enc = IRTEncoder(irt["a"], irt["b"]).fit(ds.features[tr], ds.domains[tr],
                                             ds.quality[tr], steps=400)
    noise = NoiseModel().fit(ds.verifier[tr].ravel(), ds.quality[tr].ravel(), steps=500)
    budget = tier_budgets(ds, idx, CFG)["balanced"] * 1e6   # 무제약 자연 지출
    calls = [run_tier(ds, idx, make_pandora(irt, enc, noise, lam), "x", budget)
             .calls_per_query for lam in [0.05, 0.4, 3.2]]
    assert calls[0] >= calls[1] >= calls[2]


# ---------- 4. 페이싱 v3 생존 모드 (파산 차단 구조) ----------

def test_survival_mode_forces_absolute_lambda():
    from src.engine.pacing import PacedPandora

    class _Dummy:
        def __init__(self, lam):
            self.lam = lam
    pol = PacedPandora(lambda l: _Dummy(l), lam0=0.01)
    pol.reset("t", budget=10.0, n_queries=100)
    pol.min_costs = [0.5] * 20                     # 쿼리당 최저가 0.5 관측
    pol.t, pol.cum = 20, 9.0                       # 잔여 1.0 < 80 × 0.5 → 생존 모드
    pol.observe_spend(0.0)
    assert pol.inner.lam >= 1.2 / 0.5              # 절대 생존 λ (예약지수 전면 비양수)


# ---------- 5. 재현성 (시드 고정 결정론) ----------

def test_world_generation_deterministic():
    from src.synth import make_world
    from src.textworld import build_textworld
    a1, a2 = make_world(CFG, "irt", 42), make_world(CFG, "irt", 42)
    assert (a1.quality == a2.quality).all() and (a1.verifier == a2.verifier).all()
    t1, t2 = build_textworld(CFG, seed=5), build_textworld(CFG, seed=5)
    assert t1.prompts == t2.prompts
    assert t1.samples[0][0] == t2.samples[0][0]


# ---------- 6. ScoringVerifier 접속 계약 ----------

def test_verifier_contract_shapes_and_no_label_use():
    from src.textworld import build_textworld, to_dataset
    from src.verifier import feature_matrix, fold_verifier_matrix
    cfg = dict(CFG, synth=dict(CFG["synth"], n_queries=120))
    tw = build_textworld(cfg, seed=3)
    ds, meta = to_dataset(tw, cfg)
    F = feature_matrix(meta)
    assert F.shape[:2] == ds.quality.shape
    tr = np.arange(80)
    V, _ = fold_verifier_matrix(F, ds.quality, tr)
    assert V.shape == ds.quality.shape and (V >= 0).all() and (V <= 1).all()
    # 라벨 치환 시 test 행 점수가 변하면 누수 — train 라벨만 바꿔 test 불변성 확인
    q2 = ds.quality.copy()
    q2[100:] = 1 - q2[100:]                        # test 라벨 뒤집기
    V2, _ = fold_verifier_matrix(F, q2, tr)
    assert np.allclose(V, V2)                      # test 라벨은 점수에 영향 없어야 함


# ---------- 7. 검증기 태스크 일반화 (하위호환 + 미지 도메인 폴백) ----------

def test_verifier_backward_compat_known_tasks():
    """textworld 4태스크는 특화 검사 그대로 — 특성 순서·구조 검사값 비트 불변."""
    from src.verifier import extract_features, feature_matrix, FEATURE_NAMES
    from src.textworld import build_textworld, to_dataset
    assert FEATURE_NAMES == ["has_marker", "parse_ok", "struct", "vote_share", "out_len",
                             "repetition", "hedge", "echo", "ans_len",
                             "dom_arith", "dom_sort", "dom_strcount", "dom_jsonfmt"]
    # 알려진 태스크는 특화 구조 검사 경로 (struct = index 2)
    assert extract_features("Sort ...", "Final answer: 1, 2, 3", 1.0, "sort")[2] == 1.0
    assert extract_features("Sort ...", "Final answer: 3, 1, 2", 1.0, "sort")[2] == 0.0
    assert extract_features("Compute ...", "Final answer: 42", 1.0, "arith")[2] == 1.0
    assert extract_features("Compute ...", "Final answer: forty-two", 1.0, "arith")[2] == 0.0
    # feature_matrix: 모든 task가 TASKS에 속하면 도메인 어휘=TASKS → 13특성 유지
    cfg = dict(CFG, synth=dict(CFG["synth"], n_queries=60))
    _, meta = to_dataset(build_textworld(cfg, seed=3), cfg)
    assert feature_matrix(meta).shape[2] == 13


def test_verifier_generic_fallback_unknown_domain():
    """미지 태스크·도메인에서도 크래시 없이 타입-일반 신호 + 데이터 주도 원-핫.

    ※ Phase 18: 예전에는 미지 태스크 예시로 'code' 를 썼는데, 그 이름이 실도메인 검사기로
    **등록**되면서 더 이상 미지가 아니게 됐다. 폴백 경로를 검증하려면 실제로 등록되지 않은
    이름을 써야 한다 → 'summarize'.
    """
    from src.verifier import extract_features, feature_matrix, feature_names, STRUCT_CHECKS
    assert "summarize" not in STRUCT_CHECKS
    f_json = extract_features("Write code.", "Final answer: {\"a\": 1}", 1.0, "summarize")
    f_prose = extract_features("Write code.", "Final answer: it depends on many things",
                               0.3, "summarize")
    assert f_json.shape == (13,) and f_json[2] == 1.0        # 파싱되는 JSON → 1.0
    assert f_prose[2] == 0.3                                 # 장황한 산문 → 0.3
    assert list(f_json[9:]) == [0, 0, 0, 0]                  # 미지 태스크 → dom 원-핫 전부 0
    # 등록된 실도메인은 폴백이 아니라 특화 검사를 쓴다 (Phase 18)
    from src.verifier import _tail_features, _TAIL_FEATURES
    j = _TAIL_FEATURES.index("tail_struct_task")
    assert _tail_features("Q\n(A) x\n(B) y", "... so the answer is\nB", "mmlu")[j] == 1.0
    assert _tail_features("Q\n(A) x\n(B) y", "I think it is complicated", "mmlu")[j] == 0.0
    assert _tail_features("compute", "Final answer:\n1,234", "gsm8k")[j] == 1.0
    assert _tail_features("compute", "maybe around forty two", "gsm8k")[j] == 0.0
    assert _tail_features("write fn", "def f(x):\n    return x + 1", "mbpp")[j] == 1.0
    # feature_matrix가 실도메인 어휘를 데이터에서 도출 (base 9 + 실도메인 2 = 11)
    meta = {"prompts": ["p", "q"], "tasks": ["math", "code"],
            "rep_text": [["Final answer: 42", "x"], ["Final answer: 7", "y"]],
            "vote_share": np.ones((2, 2))}
    F = feature_matrix(meta)
    assert F.shape == (2, 2, len(feature_names(["code", "math"])))   # 11
    assert F.shape[2] == 11


# ---------- 8. 호출 수 상한 (call-cap, 대회 '남은 호출 수' 신호) ----------

def test_session_respects_call_cap():
    from src.harness import Session
    s = Session(costs=np.array([1.0, 1.0, 1.0]), verifier_row=np.array([0.3, 0.5, 0.9]),
                remaining_budget=100.0, max_calls=2)
    assert s.call(0) is not None
    assert s.call(1) is not None
    assert s.call(2) is None            # 상한 도달 → 새 호출 거부
    assert s.call(0) is not None        # 재열람은 상한 무관 (무카운트)
    assert len(s.called) == 2 and s.spent == 2.0


def test_pandora_feasible_under_call_cap():
    """호출 상한 하에서 LPB가 상한을 넘지 않고 무응답 0으로 답한다 (Weitzman이 상자를
    많이 열려는 저 λ + 넉넉한 예산 = 상한만이 유일한 제약인 조건에서 검증)."""
    from src.synth import make_world
    from src.synth_ext import extend_with_votes
    from src.harness import run_tier, tier_budgets
    from src.irt.mml import fit_mml
    from src.engine.pandora import NoiseModel
    from phase2_stages import IRTEncoder
    from run_phase3 import make_pandora
    ds = extend_with_votes(make_world(CFG, "irt", seed=7), CFG)
    idx, tr = np.arange(300), np.arange(300, 1200)
    irt = fit_mml(ds.quality[tr], ds.domains[tr], 4, per_domain_b=True, steps=150)
    enc = IRTEncoder(irt["a"], irt["b"]).fit(ds.features[tr], ds.domains[tr],
                                             ds.quality[tr], steps=400)
    noise = NoiseModel().fit(ds.verifier[tr].ravel(), ds.quality[tr].ravel(), steps=500)
    budget = tier_budgets(ds, idx, CFG)["premium"]     # 넉넉한 예산 → 호출 상한만이 제약
    for cap in (1, 2, 3):
        r = run_tier(ds, idx, make_pandora(irt, enc, noise, 0.05), "x", budget, max_calls=cap)
        assert r.calls_per_query <= cap + 1e-9, f"cap {cap} 초과: {r.calls_per_query}"
        assert r.unanswered == 0, f"cap {cap}에서 무응답 발생"


# ---------- 9. SelectiveRouter (하이브리드 메타 라우터, Phase 15) ----------

def test_selective_router_fits_selects_and_respects_budget():
    """LPB vs learned를 train 내 held-out으로 선택 → 유효 정책·예산 준수 (구조 고정)."""
    from src.synth import make_world
    from src.synth_ext import extend_with_votes
    from src.harness import run_tier, tier_budgets
    from baselines.policies import SelectiveRouter
    ds = extend_with_votes(make_world(CFG, "irt", seed=7), CFG)
    tr, te = np.arange(400, 1400), np.arange(0, 300)
    sr = SelectiveRouter(CFG, CFG["synth"]["n_domains"], seed=0,
                         use_domain=False).fit(ds, tr, "fast")
    assert sr.chosen in ("lpb", "learned")          # 둘 중 하나 선택
    b_te = tier_budgets(ds, te, CFG)["fast"]
    r = run_tier(ds, te, sr.policy(), "fast", b_te)
    assert r.total_cost <= b_te + 1e-9              # 예산 준수 (규칙 강제)
    assert 0.0 <= r.mean_quality <= 1.0
    # margin=0(learned 우선)과 margin=∞(항상 LPB)의 극단이 선택을 실제로 바꾸는지
    sr_lpb = SelectiveRouter(CFG, CFG["synth"]["n_domains"], seed=0, margin=1e9,
                             use_domain=False).fit(ds, tr, "fast")
    assert sr_lpb.chosen == "lpb"                    # 큰 margin → 항상 LPB (기본 정책)
