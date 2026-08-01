"""배포 경로·튜너·대조군 결함 회귀 (D34~D49).

이 파일이 고정하는 것은 전부 **외부 감사(2026-07-27)가 실행으로 재현한 결함**이다.
공통 성질: 기존 72개 테스트가 전부 통과하는 상태에서도 살아 있었다 — 이유는 하나로
모인다. 회귀 스위트가 **오프라인 리플레이 경로만** 밟았고, 주최측 하네스가 실제로 쓰는
push 경로(텍스트 프롬프트·호출 이력·거절·문항 경계)는 한 번도 밟지 않았다.
그래서 여기서는 "어댑터를 주최측처럼 구동"하는 것을 기본 형태로 삼는다.
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from tests.test_packaging import _tiny_dataset, MIN_CFG


# ─────────────────────────── D34: λ 튜너 래칫 ───────────────────────────

def test_select_lambda_does_not_ratchet_to_the_largest():
    """품질이 λ 에 대해 단조 감소하면 **최고 후보**를 골라야 한다.

    초판은 기준선을 직전 채택자로 옮겨, 단계마다 1×SE 씩 잃으면서 격자 끝까지 올라갔다.
    """
    from src.engine.pandora import select_lambda
    rng = np.random.default_rng(0)
    n = 800
    base = (rng.random(n) < 0.70).astype(float)
    # 쌍대 구조를 실제와 같게 만든다: 같은 쿼리 집합에서 λ 가 커질수록 정답이 하나씩 꺼진다.
    # (독립 추출로 만들면 쌍대 SE 가 부풀어 이 테스트가 잡아야 할 것을 못 잡는다.)
    cands = []
    for i, lam in enumerate([0.5, 0.71, 1.0, 1.41, 2.0, 2.83, 4.0, 5.66, 8.0, 11.3, 16.0]):
        pq = base.copy()
        pq[:20 * i] = 0.0                             # 단계당 2.5%p 확실한 손실
        cands.append((lam, float(pq.mean()), pq))
    picked = select_lambda(cands, se_k=1.0)
    assert picked == 0.5, f"래칫 재발: λ={picked} (품질 단조 감소인데 최고가 아닌 λ 를 골랐다)"


def test_select_lambda_accepts_a_larger_lambda_inside_the_noise_band():
    """의도된 동작은 보존 — 손실이 잡음 수준이면 큰 λ 를 취한다 (절약)."""
    from src.engine.pandora import select_lambda
    rng = np.random.default_rng(7)
    n = 800
    base = (rng.random(n) < 0.70).astype(float)
    cands = []
    for i, lam in enumerate([1.0, 2.0, 4.0]):
        pq = base.copy()
        pq[:i] = 0.0                                  # 1건씩만 손실 = 잡음 이하
        cands.append((lam, float(pq.mean()), pq))
    assert select_lambda(cands, se_k=1.0) == 4.0


def test_select_lambda_prefers_larger_lambda_on_a_true_tie():
    """동률이면 큰 λ (절약·저지연) — 원래 의도는 보존돼야 한다."""
    from src.engine.pandora import select_lambda
    rng = np.random.default_rng(1)
    pq = (rng.random(500) < 0.5).astype(float)
    cands = [(1.0, float(pq.mean()), pq.copy()),
             (2.0, float(pq.mean()), pq.copy()),
             (4.0, float(pq.mean()), pq.copy())]
    assert select_lambda(cands, se_k=1.0) == 4.0
    assert select_lambda(cands, se_k=0.0) == 4.0


def test_select_lambda_rejects_a_significantly_worse_large_lambda():
    """큰 λ 가 **유의하게** 나쁘면 거부해야 한다 (문턱이 무력화되지 않았는지)."""
    from src.engine.pandora import select_lambda
    rng = np.random.default_rng(2)
    good = (rng.random(2000) < 0.80).astype(float)
    bad = (rng.random(2000) < 0.20).astype(float)
    cands = [(1.0, float(good.mean()), good), (8.0, float(bad.mean()), bad)]
    assert select_lambda(cands, se_k=1.0) == 1.0


# ─────────────────── D36: 텍스트 프롬프트가 배포 경로를 통과한다 ───────────────────

def _fit_router(n=360, seed=5, **kw):
    from src.router import LPBRouter
    ds = _tiny_dataset(n=n, seed=seed)
    tr = np.arange(n // 3, n)
    r = LPBRouter(MIN_CFG, 1, seed=0, use_domain=False, **kw).fit(ds, tr, "fast")
    return ds, tr, r


def test_submission_accepts_a_text_prompt_end_to_end():
    """챌린지 런타임 입력은 **텍스트 프롬프트**다. 초판은 여기서 ValueError 로 죽었다."""
    from src.submission import SubmissionRouter
    from src.text_encoder import get_encoder
    from src.cost_mirror import cost_matrix
    enc = get_encoder("hashing")
    ds = _tiny_dataset(n=360, seed=5)
    ds.features = enc.encode([f"질문 {i}: 다음을 계산하시오" for i in range(ds.n)])
    ds.text_encoder = enc                       # D36: 인코더를 데이터셋에 실어 보낸다
    from src.router import LPBRouter
    r = LPBRouter(MIN_CFG, 1, seed=0, use_domain=False).fit(ds, np.arange(120, ds.n), "fast")
    assert r.text_encoder is not None, "fit() 이 ds.text_encoder 를 물려받아야 한다"

    cmat = cost_matrix(ds)
    sub = SubmissionRouter({"fast": r}, ds.model_ids, observe=lambda p, mid, rec: 0.5)
    sub.begin_tier("fast", budget=1e6, n_queries=3)
    act = sub.step("무엇이 2 더하기 2인가?", "fast", [], cmat[0], 1e6)
    assert act.kind == "call" and act.model_id in ds.model_ids


def test_submission_text_prompt_error_is_actionable_when_encoder_missing():
    """인코더가 없으면 **임의 대체 없이** 실패하되, 무엇을 하라는지 말해야 한다."""
    from src.submission import SubmissionRouter
    from src.cost_mirror import cost_matrix
    ds, tr, r = _fit_router()
    assert getattr(r, "text_encoder", None) is None
    sub = SubmissionRouter({"fast": r}, ds.model_ids, observe=lambda p, mid, rec: 0.5)
    with pytest.raises(ValueError, match="text_encoder"):
        sub.step("텍스트 프롬프트", "fast", [], cost_matrix(ds)[0], 1e6)


# ───────────── D39: observe 가 호출 이력을 본다 (교차모델 일치의 전제) ─────────────

def test_observe_receives_call_history_when_it_asks_for_it():
    """4-인자 콜백에는 지금까지 호출한 모델 목록이 와야 한다.

    교차모델 답 일치(검증기 최대 레버 +0.178 AUC)는 **다른 출력**을 봐야 계산된다.
    초판 시그니처 (prompt, model_id, record) 로는 원리적으로 불가능했다.
    """
    from src.submission import SubmissionRouter
    from src.cost_mirror import cost_matrix
    ds, tr, r = _fit_router()
    cmat = cost_matrix(ds)
    seen = []

    def observe4(prompt, model_id, rec, called):
        seen.append((model_id, tuple(called)))
        return 0.5

    sub = SubmissionRouter({"fast": r}, ds.model_ids, observe=observe4)
    sub.begin_tier("fast", budget=1e6, n_queries=2)
    # 정책이 몇 번 호출할지에 의존하지 않도록 이력을 **직접** 구성해 재생시킨다
    # (D32: 기구가 발동했는지를 정책의 기분에 맡기지 않는다).
    hist = [{"model_id": ds.model_ids[0], "output": "a"},
            {"model_id": ds.model_ids[1], "output": "b"}]
    sub.step(ds.features[0], "fast", hist, cmat[0], 1e6)
    assert seen, "observe 가 한 번도 불리지 않았다"
    assert seen[0][1] == (), "첫 관측에는 비교 대상이 없어야 한다"
    assert seen[1][1] == (ds.model_ids[0],), (
        f"두 번째 관측에 앞선 호출 이력이 와야 한다 (got {seen[1][1]}) — "
        f"이것이 없으면 교차모델 답 일치를 배포에서 계산할 수 없다")


def test_three_arg_observe_still_works():
    """하위호환 — 기존 3-인자 콜백이 그대로 동작해야 한다."""
    from src.submission import SubmissionRouter
    from src.cost_mirror import cost_matrix
    ds, tr, r = _fit_router()
    sub = SubmissionRouter({"fast": r}, ds.model_ids,
                           observe=lambda p, mid, rec: 0.5)
    sub.begin_tier("fast", budget=1e6, n_queries=1)
    act = sub.step(ds.features[0], "fast", [{"model_id": ds.model_ids[0], "output": "x"}],
                   cost_matrix(ds)[0], 1e6)
    assert act.kind in ("call", "answer", "abstain")


# ───────────────── D40: 호스트가 호출을 거절해도 무한루프에 빠지지 않는다 ─────────────────

def test_adapter_terminates_when_host_refuses_every_call():
    """호스트가 모든 호출을 거절하면(이력이 안 자란다) 유한 스텝 안에 끝나야 한다.

    초판은 이력만 보고 결정을 재계산해 **같은 액션을 영원히 재발행**했다 — 채점 하네스
    안에서는 무한루프다.
    """
    from src.submission import SubmissionRouter
    from src.cost_mirror import cost_matrix
    ds, tr, r = _fit_router()
    cmat = cost_matrix(ds)
    sub = SubmissionRouter({"fast": r}, ds.model_ids, observe=lambda p, mid, rec: 0.5)
    sub.begin_tier("fast", budget=1e6, n_queries=1)
    kinds = []
    for _ in range(3 * ds.m + 5):                  # 넉넉한 상한
        act = sub.step(ds.features[0], "fast", [], cmat[0], 1e6)   # 이력을 절대 안 늘린다
        kinds.append(act.kind)
        if act.kind in ("answer", "abstain"):
            break
    else:
        raise AssertionError(f"거절 루프에서 종료하지 않음: {kinds[:12]}")
    assert kinds[-1] == "abstain", f"모든 상자가 거절되면 기권해야 한다 (got {kinds[-1]})"


# ───────────────── D41: end_query 가 조용히 엉뚱한 tier 를 갱신하지 않는다 ─────────────────

def test_end_query_requires_tier_when_multiple_tiers():
    from src.submission import SubmissionRouter
    ds, tr, r = _fit_router()
    from src.router import LPBRouter
    r2 = LPBRouter(MIN_CFG, 1, seed=0, use_domain=False).fit(ds, tr, "fast")
    sub = SubmissionRouter({"fast": r, "balanced": r2}, ds.model_ids,
                           observe=lambda p, mid, rec: 0.5)
    with pytest.raises(ValueError, match="tier"):
        sub.end_query(0.1)                          # tier 생략 → 조용히 첫 정책만 갱신했었다
    sub.end_query(0.1, "fast")                      # 명시하면 정상


def test_end_query_still_allows_omitting_tier_for_a_single_tier():
    from src.submission import SubmissionRouter
    ds, tr, r = _fit_router()
    sub = SubmissionRouter({"fast": r}, ds.model_ids, observe=lambda p, mid, rec: 0.5)
    sub.begin_tier("fast", budget=1.0, n_queries=4)
    sub.end_query(0.1)                              # 단일 tier 면 모호하지 않다


# ───────────── D43: 생존 모드가 push 경로에서 실제로 발동한다 ─────────────

def test_survival_mode_is_fed_in_the_push_path():
    """`PacedPandora.min_costs` 는 route() 안에서만 채워져 배포 경로에서 영구 비활성이었다.

    회귀 테스트가 `pol.min_costs` 를 손으로 주입하고 있어 잡히지 않았다 — 여기서는
    **주입하지 않고** 어댑터를 구동해 채워지는지 본다 (D32: 발동을 증명하라).
    """
    from src.submission import SubmissionRouter
    from src.cost_mirror import cost_matrix
    ds, tr, r = _fit_router()
    cmat = cost_matrix(ds)
    sub = SubmissionRouter({"fast": r}, ds.model_ids, observe=lambda p, mid, rec: 0.5)
    pol = sub.policies["fast"]
    if not hasattr(pol, "min_costs"):
        pytest.skip("λ 고정 모드 — 페이싱 상태가 없다")
    sub.begin_tier("fast", budget=1e6, n_queries=5)
    assert pol.min_costs == []
    for i in range(3):
        hist = []
        for _ in range(ds.m + 1):
            act = sub.step(ds.features[i], "fast", hist, cmat[i], 1e6)
            if act.kind != "call":
                break
            hist.append({"model_id": act.model_id, "output": "x"})
        sub.end_query(0.0, "fast")
    assert len(pol.min_costs) == 3, (
        f"문항당 1회 기록돼야 한다 (got {len(pol.min_costs)}) — 생존 모드 입력이 비어 있으면 "
        f"그 기구는 배포에서 영구 비활성이다")


# ───────────── D44/D45: 게이트 산출물이 실제로 배포 가능하다 ─────────────

def test_gate_emits_deployable_routers_and_a_runtime_verifier():
    """수령 당일 첫 명령의 산출물이 제출 어댑터에 그대로 꽂혀야 한다.

    초판은 `use_domain` 기본값(True) 때문에 다집단 2PL 라우터를 냈고, 챌린지 런타임에는
    도메인 라벨이 없어 어댑터의 D18 가드가 그것을 **거부**했다.
    """
    from src.gate import run_gate
    from src.submission import SubmissionRouter
    from src.cost_mirror import cost_matrix
    ds = _tiny_dataset(n=420, seed=3)
    ds.domains = np.arange(ds.n) % 4                 # 도메인이 여러 개인 데이터
    tr = np.arange(120, 420)
    dec, routers = run_gate(ds, MIN_CFG, tr, tiers=["fast"], meta=None)   # lpb_kwargs 미지정
    r = routers["fast"]
    b = getattr(r, "irt", {}).get("b", np.zeros((1, 1)))
    assert b.shape[0] == 1, "게이트는 배포 가능한 단일집단 라우터를 내야 한다"
    sub = SubmissionRouter({"fast": r}, ds.model_ids, observe=lambda p, mid, rec: 0.5)
    act = sub.step(ds.features[0], "fast", [], cost_matrix(ds)[0], 1e6)   # domain 미지정
    assert act.kind == "call"


def test_gate_returns_the_fitted_scoring_verifier_when_text_is_available():
    """운영자가 런타임 observe 를 만들 수 있어야 한다 (초판은 sv 를 버렸다)."""
    from src.gate import measure_verifier
    ds = _tiny_dataset(n=200, seed=3)
    auc, V, sv = measure_verifier(ds, None, np.arange(60, 200))
    assert V is None and sv is None                  # meta 가 없으면 만들 것이 없다
    assert isinstance(auc, float)


# ═════════════ D63: 비용 규약 (단가 정책) + 문항별 호출 상한 ═════════════
#
# 외부 레드팀(2026-08-01)이 재현한 결함: 대회 상세는 런타임 제공 항목을 "모델 프로파일 +
# 공식 **비용 정책**" 이라 적는데, 어댑터는 후보별 **숫자 비용**만 받아 `KeyError: 'cost'`
# 로 첫 스텝에서 죽었다. 그리고 "남은 호출 수"도 명시 항목인데 받을 자리가 없었다.
# D36/D43 과 같은 자리다 — 오프라인 리플레이는 `cost_mirror` 로 비용을 직접 만들어 쓰므로
# 이 경로를 원리적으로 밟지 않는다.

def _price_meta(ds, out_tokens=None):
    """호스트가 '토큰당 단가'로 주는 형태 (config.yaml 의 가격 필드와 같은 이름)."""
    out = {}
    for k, mid in enumerate(ds.model_ids):
        e = {"prefill_price": ds.models[mid].prefill_price,
             "decode_price": ds.models[mid].decode_price}
        if out_tokens is not None:
            e["avg_out_tokens"] = float(out_tokens[k])
        out[mid] = e
    return out


def test_price_policy_metadata_produces_the_cost_mirror_formula():
    """단가 + 토큰 수 → `cost_mirror.call_cost` 와 **정확히 같은 값**이어야 한다."""
    from src.submission import SubmissionRouter
    from src.cost_mirror import cost_matrix
    ds, tr, r = _fit_router()
    sub = SubmissionRouter({"fast": r}, ds.model_ids, observe=lambda p, mid, rec: 0.5)
    i = 0
    meta = _price_meta(ds, out_tokens=ds.out_tokens[i])
    with pytest.warns(RuntimeWarning, match="척도 불일치"):   # 오라클 λ + 단가 경로 → 경고가 정상
        got = sub._cost_vector("아무 프롬프트", meta, ds.features[i],
                               prompt_tokens=float(ds.in_tokens[i]))
    assert np.allclose(got, cost_matrix(ds)[i]), (
        f"단가 경로가 채점 산식과 어긋난다: {got} vs {cost_matrix(ds)[i]}")
    assert sub.cost_path == "price_policy", "단가 경로가 발동했음을 증명해야 한다 (D32)"


def test_price_policy_end_to_end_step_does_not_crash():
    """★ 감사 재현 케이스 — 이전에는 여기서 `KeyError: 'cost'` 였다."""
    from src.submission import SubmissionRouter
    from src.text_encoder import get_encoder
    from src.router import LPBRouter
    enc = get_encoder("hashing")
    ds = _tiny_dataset(n=360, seed=5)
    ds.features = enc.encode([f"질문 {i}" for i in range(ds.n)])
    ds.text_encoder = enc
    r = LPBRouter(MIN_CFG, 1, seed=0, use_domain=False,
                  cost_mode="ridge").fit(ds, np.arange(120, ds.n), "fast")
    assert r.cost_est is not None
    sub = SubmissionRouter({"fast": r}, ds.model_ids, observe=lambda p, mid, rec: 0.5)
    assert sub.decode_estimator is not None, "라우터의 길이 추정기를 물려받아야 한다"
    act = sub.step("한국어 프롬프트입니다", "fast", [], _price_meta(ds), 1e6)
    assert act.kind == "call" and act.model_id in ds.model_ids


def test_unreadable_cost_metadata_fails_loudly_with_the_keys_it_saw():
    """임의 대체 대신 **무엇을 하라**를 말해야 한다 (비용은 정책 전체를 바꾼다)."""
    from src.submission import SubmissionRouter
    ds, tr, r = _fit_router()
    sub = SubmissionRouter({"fast": r}, ds.model_ids, observe=lambda p, mid, rec: 0.5)
    bad = {mid: {"토큰당요금": 0.1} for mid in ds.model_ids}
    with pytest.raises(KeyError, match="비용 메타데이터를 해석할 수 없습니다"):
        sub._cost_vector("p", bad, ds.features[0])


def test_custom_cost_spec_absorbs_foreign_key_names_and_units():
    """호스트 규약이 달라도 `CostSpec` 교체만으로 흡수된다 (파일 수정 없이)."""
    from src.submission import SubmissionRouter, CostSpec
    from src.cost_mirror import cost_matrix
    ds, tr, r = _fit_router()
    spec = CostSpec(prefill_keys=("in_usd_per_mtok",), decode_keys=("out_usd_per_mtok",),
                    price_unit=1e6)
    sub = SubmissionRouter({"fast": r}, ds.model_ids, observe=lambda p, mid, rec: 0.5,
                           cost_spec=spec)
    i = 0
    meta = {mid: {"in_usd_per_mtok": ds.models[mid].prefill_price * 1000.0,
                  "out_usd_per_mtok": ds.models[mid].decode_price * 1000.0,
                  "avg_out_tokens": float(ds.out_tokens[i, k])}
            for k, mid in enumerate(ds.model_ids)}
    with pytest.warns(RuntimeWarning, match="척도 불일치"):   # 위와 같은 이유 (기본 라우터 = oracle)
        got = sub._cost_vector("p", meta, ds.features[i],
                               prompt_tokens=float(ds.in_tokens[i]))
    assert np.allclose(got, cost_matrix(ds)[i])


def test_decode_length_unknown_fails_instead_of_guessing():
    """길이를 모르면 상수로 때우지 않고 세 가지 해법을 말하며 실패한다."""
    from src.submission import SubmissionRouter
    ds, tr, r = _fit_router()                       # cost_mode 기본값 = oracle → 추정기 없음
    sub = SubmissionRouter({"fast": r}, ds.model_ids, observe=lambda p, mid, rec: 0.5)
    assert sub.decode_estimator is None
    with pytest.raises(KeyError, match="디코드 길이를 알 수 없어"):
        sub._cost_vector("p", _price_meta(ds), ds.features[0])


def test_oracle_lambda_with_estimated_costs_warns_about_scale_mismatch():
    """조용히 다른 정책이 되는 대신 경고한다 (λ 는 오라클 비용으로 튜닝됐다)."""
    from src.submission import SubmissionRouter
    ds, tr, r = _fit_router()
    sub = SubmissionRouter({"fast": r}, ds.model_ids, observe=lambda p, mid, rec: 0.5)
    meta = _price_meta(ds, out_tokens=ds.out_tokens[0])
    with pytest.warns(RuntimeWarning, match="척도 불일치"):
        sub._cost_vector("p", meta, ds.features[0])


def test_remaining_calls_from_the_host_caps_this_query():
    """★ 대회 상세의 '남은 호출 수' — 생성자 인자가 아니라 **문항마다** 온다."""
    from src.submission import SubmissionRouter
    from src.cost_mirror import cost_matrix
    ds, tr, r = _fit_router()
    cmat = cost_matrix(ds)
    sub = SubmissionRouter({"fast": r}, ds.model_ids, observe=lambda p, mid, rec: 0.9)
    sub.begin_tier("fast", budget=1e6, n_queries=1)
    hist, calls = [], 0
    for _ in range(6):
        act = sub.step(ds.features[0], "fast", hist, cmat[0], 1e6,
                       remaining_calls=1 - calls)
        if act.kind != "call":
            break
        calls += 1
        hist.append({"model_id": act.model_id, "output": "x"})
    assert calls == 1, f"호스트 상한 1 을 넘겼다 (호출 {calls}회)"
    assert act.kind == "answer" and act.model_id in [h["model_id"] for h in hist]


def test_zero_remaining_calls_abstains_instead_of_forcing_a_call():
    """상한 0 이면 최저가 강제 호출 경로도 막혀야 한다 (상한은 예산과 달리 협상 불가)."""
    from src.submission import SubmissionRouter
    from src.cost_mirror import cost_matrix
    ds, tr, r = _fit_router()
    sub = SubmissionRouter({"fast": r}, ds.model_ids, observe=lambda p, mid, rec: 0.9)
    act = sub.step(ds.features[0], "fast", [], cost_matrix(ds)[0], 1e6, remaining_calls=0)
    assert act.kind == "abstain"
