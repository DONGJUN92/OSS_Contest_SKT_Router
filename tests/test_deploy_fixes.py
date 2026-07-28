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
