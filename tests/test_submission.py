"""제출 어댑터 등가성 테스트 (B10).

어댑터의 유일한 정당성은 **내부 정책과 같은 결정을 내린다**는 것이다. 어댑터는
제어 흐름이 반대(주최측이 스텝마다 호출)이고 신념을 이력에서 복원하므로, 재생 순서나
정지 조건이 조금만 어긋나도 조용히 다른 정책이 된다 — 그 사고를 여기서 못 박는다.

검증 축
  ① 액션 열 일치   : route()가 실제로 호출한 모델 순서 == step()이 지시한 호출 순서
  ② 최종 답 일치   : 두 경로가 고른 최종 모델이 동일
  ③ 규칙 준수      : step()은 예산을 넘는 호출을 지시하지 않는다
  ④ 페이싱 동일    : 같은 지출을 알려주면 λ 궤적이 같다
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import yaml

from src.cost_mirror import cost_matrix
from src.harness import Session, tier_budgets
from src.router import LPBRouter
from src.submission import Action, SubmissionRouter
from src.synth import make_world

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))


class _RecordingSession(Session):
    """route()가 실제로 호출한 순서를 기록하는 Session."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.order: list[int] = []

    def call(self, m_idx: int):
        before = len(self.called)
        v = super().call(m_idx)
        if v is not None and len(self.called) > before:
            self.order.append(m_idx)
        return v


def _fit(world="irt", tier="fast", seed=7):
    ds = make_world(CFG, world, seed=seed)
    tr = np.arange(1000, 2000)
    te = np.arange(0, 200)
    router = LPBRouter(CFG, CFG["synth"]["n_domains"], seed=0).fit(ds, tr, tier)
    return ds, tr, te, router


def _drive_adapter(sub, ds, cmat, i, tier, remaining, pass_domain=True):
    """어댑터를 주최측처럼 구동 → (호출 순서, 최종 답, 지출).

    pass_domain=False: 챌린지 런타임처럼 **도메인 라벨 없이** 구동 (단일집단 배포 경로).
    """
    hist, spent = [], 0.0
    order = []
    for _ in range(ds.m + 2):                       # 무한루프 방지 상한
        kw = {"domain": int(ds.domains[i])} if pass_domain else {}
        act = sub.step(ds.features[i], tier, hist, cmat[i], remaining - spent, **kw)
        m = sub.index[act.model_id]
        if act.kind == "answer":
            return order, m, spent
        assert cmat[i, m] <= remaining - spent + 1e-9, "예산 초과 호출 지시"
        order.append(m)
        spent += cmat[i, m]
        hist.append({"model_id": act.model_id, "output": None, "_row": i, "_m": m})
    raise AssertionError("어댑터가 종료하지 않음")


@pytest.mark.parametrize("world,tier", [("irt", "fast"), ("irt", "balanced"),
                                        ("specialist", "fast")])
def test_adapter_matches_internal_policy(world, tier):
    ds, tr, te, router = _fit(world, tier)
    cmat = cost_matrix(ds)
    budget = tier_budgets(ds, te, CFG)[tier]

    pol = router.policy()                            # 내부 경로
    pol.reset(tier=tier, budget=budget, n_queries=len(te))
    sub = SubmissionRouter({tier: router}, ds.model_ids,
                           observe=lambda p, mid, rec: ds.verifier[rec["_row"], rec["_m"]])
    sub.begin_tier(tier, budget=budget, n_queries=len(te))

    spent_total_a = spent_total_b = 0.0
    for i in te:
        # 내부 경로
        sess = _RecordingSession(costs=cmat[i], verifier_row=ds.verifier[i],
                                 remaining_budget=budget - spent_total_a)
        choice_a = pol.route(sess, ds.features[i], int(ds.domains[i]))
        if choice_a not in sess.called and sess.called:
            choice_a = max(sess.called, key=lambda m: sess.verifier_row[m])
        spent_total_a += sess.spent
        pol.observe_spend(sess.spent)

        # 어댑터 경로
        order_b, choice_b, spent_b = _drive_adapter(
            sub, ds, cmat, i, tier, budget - spent_total_b)
        spent_total_b += spent_b
        sub.end_query(spent_b, tier)

        assert sess.order == order_b, f"쿼리 {i}: 호출 열 불일치 {sess.order} vs {order_b}"
        if sess.called:
            assert choice_a == choice_b, f"쿼리 {i}: 최종 답 불일치"
        assert spent_b == pytest.approx(sess.spent), f"쿼리 {i}: 지출 불일치"

    assert spent_total_b == pytest.approx(spent_total_a)
    assert spent_total_b <= budget + 1e-9
    assert pol.inner.lam == pytest.approx(sub.policies[tier].inner.lam), "λ 궤적 불일치"


def test_adapter_refuses_to_guess_domain():
    """다집단 적합인데 도메인을 안 주면 조용히 틀리지 말고 즉시 실패해야 한다."""
    ds, tr, te, router = _fit()
    cmat = cost_matrix(ds)
    sub = SubmissionRouter({"fast": router}, ds.model_ids, observe=lambda *a: 0.5)
    sub.begin_tier("fast", budget=1e6, n_queries=10)
    with pytest.raises(ValueError, match="도메인"):
        sub.step(ds.features[0], "fast", [], cmat[0], remaining_budget=1e6)
    act = sub.step(ds.features[0], "fast", [], cmat[0], remaining_budget=1e6, domain=2)
    assert act.kind == "call"


def test_adapter_domain_free_when_single_group():
    """런타임에 도메인이 없다 (대회 상세: task 이름 미제공). use_domain=False 라우터는
    도메인 인자 없이 동작하고 내부 route()와 등가여야 한다 — 단일집단 배포 경로.

    도메인이 가장 중요한 specialist 세계로 검증한다: 단일집단이라도 어댑터 재생이
    route()와 정확히 같은 호출 열·최종 답·지출·λ 궤적을 내야 한다.
    """
    world, tier = "specialist", "fast"
    ds = make_world(CFG, world, seed=7)
    tr, te = np.arange(1000, 2000), np.arange(0, 200)
    router = LPBRouter(CFG, CFG["synth"]["n_domains"], seed=0,
                       use_domain=False).fit(ds, tr, tier)
    cmat = cost_matrix(ds)
    budget = tier_budgets(ds, te, CFG)[tier]

    pol = router.policy()
    pol.reset(tier=tier, budget=budget, n_queries=len(te))
    sub = SubmissionRouter({tier: router}, ds.model_ids,
                           observe=lambda p, mid, rec: ds.verifier[rec["_row"], rec["_m"]])
    assert sub.n_domains == 1, "단일집단이면 도메인 요구가 사라져야 한다 (D18 가드 미발동)"
    sub.begin_tier(tier, budget=budget, n_queries=len(te))

    spent_a = spent_b = 0.0
    for i in te:
        sess = _RecordingSession(costs=cmat[i], verifier_row=ds.verifier[i],
                                 remaining_budget=budget - spent_a)
        choice_a = pol.route(sess, ds.features[i], int(ds.domains[i]))
        if choice_a not in sess.called and sess.called:
            choice_a = max(sess.called, key=lambda m: sess.verifier_row[m])
        spent_a += sess.spent
        pol.observe_spend(sess.spent)

        order_b, choice_b, spent_bi = _drive_adapter(       # 도메인 없이 구동
            sub, ds, cmat, i, tier, budget - spent_b, pass_domain=False)
        spent_b += spent_bi
        sub.end_query(spent_bi, tier)

        assert sess.order == order_b, f"쿼리 {i}: 도메인 없이도 호출 열 일치해야"
        if sess.called:
            assert choice_a == choice_b, f"쿼리 {i}: 최종 답 불일치"
        assert spent_bi == pytest.approx(sess.spent)
    assert spent_b == pytest.approx(spent_a)
    assert pol.inner.lam == pytest.approx(sub.policies[tier].inner.lam), "λ 궤적 불일치"


def test_adapter_cost_vector_formats():
    ds, tr, te, router = _fit()
    sub = SubmissionRouter({"fast": router}, ds.model_ids, observe=lambda *a: 0.5)
    ref = cost_matrix(ds)[0]
    as_dict = {mid: float(ref[i]) for i, mid in enumerate(ds.model_ids)}
    as_meta = {mid: {"cost": float(ref[i])} for i, mid in enumerate(ds.model_ids)}
    for form in (ref, list(ref), as_dict, as_meta):
        assert np.allclose(sub._cost_vector(None, form), ref)


def test_adapter_abstains_when_budget_exhausted():
    """예산이 최저가 1회조차 못 되면 **기권**해야 한다 (Phase 17 규칙 위반 수정).

    이전 동작은 `Action("answer", 최저가모델)` 이었고, 그 모델은 호출된 적이 없다.
    챌린지 규칙은 "최종 답은 호출한 후보 출력 중 하나"이므로 그 액션은 규칙 위반이다.
    (이 테스트는 예전에 그 위반 동작을 고정하고 있었다 — 테스트가 결함을 지켜준 사례.)
    """
    ds, tr, te, router = _fit()
    cmat = cost_matrix(ds)
    sub = SubmissionRouter({"fast": router}, ds.model_ids, observe=lambda *a: 0.5)
    sub.begin_tier("fast", budget=1e-9, n_queries=10)
    act = sub.step(ds.features[0], "fast", [], cmat[0], remaining_budget=1e-9, domain=0)
    assert isinstance(act, Action)
    assert act.kind == "abstain", "예산이 없으면 호출도 지시하지 않고, 답도 지명하지 않는다"


def test_adapter_never_answers_an_uncalled_model():
    """어떤 예산에서도 answer 는 호출 이력 안의 모델만 지명한다 (규칙 불변식)."""
    ds, tr, te, router = _fit()
    cmat = cost_matrix(ds)
    budget = tier_budgets(ds, te, CFG)["fast"]
    sub = SubmissionRouter({"fast": router}, ds.model_ids,
                           observe=lambda p, mid, rec: ds.verifier[rec["_row"], rec["_m"]])
    sub.begin_tier("fast", budget=budget, n_queries=len(te))
    for i in te[:60]:
        for rb in (1e-9, budget * 1e-4, budget):
            hist, called = [], []
            for _ in range(ds.m + 2):
                act = sub.step(ds.features[i], "fast", hist, cmat[i], rb, domain=0)
                if act.kind == "abstain":
                    assert act.model_id == ""
                    break
                if act.kind == "answer":
                    assert act.model_id in called, "호출하지 않은 모델을 답으로 지명했다"
                    break
                m = sub.index[act.model_id]
                called.append(act.model_id)
                hist.append({"model_id": act.model_id, "output": None, "_row": i, "_m": m})


def test_adapter_deploys_selective_router():
    """`SelectiveRouter` 가 제출 규약으로 실제 배포되는가 (Phase 17).

    Phase 15는 이 정책을 "3모델·실텍스트 한계의 극복책"으로 발표했지만, 어댑터가
    `r.irt` 와 `policy().inner` 를 무조건 요구해 **배포 자체가 불가능**했다.
    두 분기(lpb 선택 / learned 선택) 모두 유효한 액션 열을 내는지 고정한다.
    """
    from baselines.policies import SelectiveRouter
    ds = make_world(CFG, "irt", seed=7)
    tr, te = np.arange(400, 1400), np.arange(0, 40)
    cmat = cost_matrix(ds)
    budget = tier_budgets(ds, te, CFG)["fast"]
    # margin=+∞ → 항상 LPB. margin=−∞ → train-cal 최고 대안 (learned 또는 cascade_routing).
    # 세 어댑터 종류(pandora/lookahead/single)를 모두 밟기 위해 남은 하나는 명시 구성한다.
    variants = []
    for margin in (1e9, -1e9):
        sr = SelectiveRouter(CFG, CFG["synth"]["n_domains"], seed=0, margin=margin,
                             use_domain=False).fit(ds, tr, "fast")
        variants.append(sr)
    assert variants[0].chosen == "lpb"
    assert variants[1].chosen in ("learned", "cascade_routing")
    if variants[1].chosen != "learned":                 # 단일호출 어댑터도 반드시 덮는다
        forced = SelectiveRouter(CFG, CFG["synth"]["n_domains"], seed=0, margin=1e9,
                                 use_domain=False).fit(ds, tr, "fast")
        forced.chosen = "learned"
        forced._pol = forced._fit_learned(ds, tr, "fast")
        variants.append(forced)
    kinds = set()
    for sr in variants:
        sub = SubmissionRouter({"fast": sr}, ds.model_ids,
                              observe=lambda p, mid, rec: ds.verifier[rec["_row"], rec["_m"]])
        kinds.add(sub._kind["fast"])
        sub.begin_tier("fast", budget=budget, n_queries=len(te))
        for i in te:
            hist, called, spent = [], [], 0.0
            for _ in range(ds.m + 2):
                act = sub.step(ds.features[i], "fast", hist, cmat[i], budget - spent, domain=0)
                if act.kind in ("answer", "abstain"):
                    if act.kind == "answer":
                        assert act.model_id in called
                    break
                m = sub.index[act.model_id]
                assert cmat[i, m] <= budget - spent + 1e-9
                spent += cmat[i, m]
                called.append(act.model_id)
                hist.append({"model_id": act.model_id, "output": None, "_row": i, "_m": m})
            else:
                raise AssertionError("어댑터가 종료하지 않음")
            sub.end_query(spent, "fast")
    assert {"pandora", "single"} <= kinds, f"어댑터 종류 미커버: {kinds}"
