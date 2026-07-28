"""제출 인터페이스 어댑터 (B10 — PROJECT_PLAN §9 P3).

챌린지 규격(challenge_description.md §4)
    입력: 프롬프트 · budget tier · 호출 이력 · 후보 모델 메타데이터
    출력: 특정 후보 모델 **호출**, 또는 기존 출력 중 최종 **답 선택**

**본질은 제어 흐름의 반전이다.** 내부 정책 `PandoraPolicy.route()` 는 스스로 루프를
돌며 `Session.call()` 로 모델을 *당긴다*(pull). 주최측 하네스는 반대로 매 스텝 우리를
호출해 액션 하나를 *받아간다*(push). 어댑터는 **호출 이력으로부터 신념 상태를 복원**해
같은 결정을 내린다 — Weitzman 정책은 (신념, 비용, λ)만의 함수이고 신념은 관측의
순차 갱신이므로, 이력을 순서대로 재생하면 내부 루프와 정확히 같은 상태에 도달한다.

이 등가성은 추측이 아니라 **회귀 테스트로 고정**돼 있다
(`tests/test_submission.py`: 6세계 전 쿼리에서 route()와 step()의 액션 열이 일치).

주최측 실제 API 확인 시 교체 지점 (전부 이 파일 안):
  ① `Action` 의 표현 — ("call", id) / ("answer", id) 를 주최측 열거형으로
  ② `_cost_vector` — 후보별 비용을 메타데이터에서 읽는 방식 (Q3)
  ③ `_observe`     — 호출된 출력에서 검증기 점수를 얻는 방식
  ④ `begin_tier` / `end_query` 훅의 호출 시점 (예산·페이싱 상태 갱신)
"""
from dataclasses import dataclass
import inspect

import numpy as np


@dataclass(frozen=True)
class Action:
    """주최측에 돌려줄 액션.

    kind = "call"    모델 호출
         | "answer"  최종 답 선택 — **반드시 이미 호출한 모델**이어야 한다 (챌린지 규칙)
         | "abstain" 예산이 최저가 1회조차 못 되어 유효한 답이 없음 (Phase 17 신설).
                     예전에는 이 경우 호출하지 않은 모델을 answer 로 지명해 규칙을
                     위반했다. 하네스의 무응답=0점 의미론과 대응된다.
    """
    kind: str
    model_id: str

    def __repr__(self) -> str:                       # 로그 가독성
        return f"{self.kind}:{self.model_id}"


class SubmissionRouter:
    """tier별로 적합된 LPBRouter를 챌린지 호출 규약으로 감싼다.

    사용 흐름
        sub = SubmissionRouter({t: fitted[t] for t in tiers}, model_ids, observe=...)
        sub.begin_tier("fast", budget=B, n_queries=N)      # tier 시작 1회
        for prompt in prompts:                             # 문항마다
            hist = []
            while True:
                act = sub.step(prompt, "fast", hist, meta, remaining_budget=rb)
                if act.kind == "answer":
                    break
                out = harness_call(act.model_id)           # 주최측이 실제 호출
                hist.append({"model_id": act.model_id, "output": out})
            sub.end_query(spent_on_this_query)             # 페이싱 λ 갱신

    observe: (prompt, model_id, record) -> v ∈ [0,1]
        호출된 출력에서 검증기 점수를 얻는 함수. 실배포에서는
        `ScoringVerifier`(Phase 8)가 출력 텍스트로부터 계산한다. 사전 계산된 점수를
        주최측이 직접 준다면 그 값을 그대로 돌려주면 된다.
    """

    def __init__(self, routers: dict, model_ids: list[str], observe,
                 domain_of=None, max_calls: int | None = None,
                 remaining_is_net: bool = False, cost_margin: float | None = None):
        self.routers = routers
        self.model_ids = list(model_ids)
        self.index = {m: i for i, m in enumerate(self.model_ids)}
        self.observe = observe
        self.max_calls = max_calls          # 호출 수 상한 (하네스 Session.max_calls와 동일 의미론)
        self.policies = {t: r.policy() for t, r in routers.items()}
        self._encoded: dict[str, np.ndarray] = {}
        # ── D37: `remaining_budget` 의 의미론 (주최측 규약 미확인 → 명시 스위치) ──
        # False (기본): 호스트가 **이번 문항 시작 시점의** 잔여 예산을 준다 → 이번 문항에서
        #               이미 쓴 금액을 우리가 빼야 한다.
        # True         : 호스트가 **이미 이번 문항 지출을 반영한** 잔여 예산을 준다 → 빼면
        #               이중 차감이다. 초판은 이 경우를 표현할 수 없었고, 저장소 자체 드라이버가
        #               쓰는 규약(net)에서 balanced 품질 −0.0250 손실이 측정됐다.
        self.remaining_is_net = bool(remaining_is_net)
        # ── D38: 계획 단계 비용 보수화. 라우터가 `cost_margin` 으로 적합됐으면 어댑터도
        # 같은 마진을 써야 route() 와 같은 결정을 낸다 (branch_decision_table 이 권장하는
        # 배포 구성 `cost_margin=0.05` 가 초판 어댑터에는 아예 반영되지 않았다).
        if cost_margin is None:
            margins = [float(getattr(r, "cost_margin", 0.0) or 0.0) for r in routers.values()]
            cost_margin = max(margins) if margins else 0.0
        self.cost_margin = float(cost_margin)
        # ── D39: observe 규약 확장. 교차모델 답 일치(+0.178 AUC)는 **이미 호출한 다른 출력**을
        # 봐야 계산되는데 초판 시그니처 (prompt, model_id, record) 로는 원리적으로 불가능했다.
        # 4번째 인자로 지금까지의 호출 이력을 넘긴다. 기존 3-인자 콜백도 그대로 동작한다.
        try:
            self._observe_arity = len(inspect.signature(observe).parameters)
        except (TypeError, ValueError):                 # 내장/C 콜러블 등
            self._observe_arity = 3
        # ── D40: 문항 단위 상태. 호스트가 호출을 거절하면 이력이 자라지 않아 같은 액션을
        # 영원히 재발행한다(채점 하네스 무한루프). 거절로 판단된 상자를 이 문항에서 제외한다.
        self._refused: set[int] = set()
        self._last_req: tuple | None = None
        self._prev_hist_len: int = -1       # -1 = 아직 어떤 문항도 시작되지 않음
        # ── Phase 17: 어댑터를 **정책 종류에 무관**하게 만든다 (레드팀 MAJOR) ──
        # 이전 버전은 `r.irt["b"]` 와 `policy().inner` 를 무조건 요구했다. 그래서
        #   · `baselines.SelectiveRouter` (Phase 15가 "한계 극복"이라 발표한 정책)
        #   · 학습형 단일호출로 전환된 경우
        # 는 **배포 자체가 불가능**했다 — 선언된 극복책이 제출 경로에 없었던 셈이다.
        # 이제 tier 마다 정책 형태를 판별해 두 어댑터 중 하나를 붙인다.
        self._kind = {t: self._classify(r) for t, r in routers.items()}
        dims = [int(r.irt["b"].shape[0]) for r in routers.values() if hasattr(r, "irt")]
        self.n_domains = max(dims) if dims else 1
        # 다집단 2PL은 도메인마다 b_{d,m}이 다르다. 도메인을 모르면 **조용히 다른 정책이
        # 된다** — 기본값 0으로 넘어가지 않고 즉시 실패시킨다 (개발 중 발견한 함정).
        # 단 cluster 모드(pseudo-domain)는 프롬프트에서 집단을 스스로 정하므로 예외 대상이 아니다.
        self._self_grouping = all(getattr(r, "clusterer", None) is not None
                                  for r in routers.values()) and bool(routers)
        if domain_of is None and self.n_domains > 1 and not self._self_grouping:
            def _missing(prompt, meta):
                raise ValueError(
                    f"이 라우터는 도메인 {self.n_domains}개로 적합됐습니다. "
                    f"step(..., domain=...)로 넘기거나 domain_of를 지정하세요 — "
                    f"도메인을 틀리면 예약값이 달라져 다른 정책이 됩니다")
            domain_of = _missing
        self.domain_of = domain_of or (lambda prompt, meta: 0)

    @staticmethod
    def _classify(router) -> str:
        """정책 형태 판별.

        "pandora"   — Weitzman 예약지수로 열고 멈춘다 (LPBRouter)
        "lookahead" — 1스텝 lookahead 결합 결정 (CascadeRouting; SelectiveRouter 가 고를 수 있다)
        "single"    — 1회 호출 후 확정 (LearnedRouter)
        """
        pol = router.policy() if hasattr(router, "policy") else router
        if type(pol).__name__ == "CascadeRouting":
            return "lookahead"
        inner = getattr(pol, "inner", pol)
        return "pandora" if hasattr(inner, "make_belief") else "single"

    # ---------------- tier / 쿼리 경계 훅 ----------------

    def begin_tier(self, tier: str, budget: float, n_queries: int):
        """tier 평가 시작. 페이싱 컨트롤러의 예산·진도 상태를 초기화한다.

        Phase 17: 훅이 없는 정책(λ 고정·단일호출)도 조용히 통과시킨다 — 주최측 하네스가
        이 훅을 불러 줄지 알 수 없으므로 `PacedPandora` 도 reset 없이 동작하게 고쳤다.
        """
        pol = self.policies[tier]
        if hasattr(pol, "reset"):
            pol.reset(tier=tier, budget=budget, n_queries=n_queries)

    def end_query(self, spent: float, tier: str = None):
        """문항 종료. 실제 지출을 알려 shadow price λ를 갱신한다 (M3).

        ★ D41: 초판은 `tier=None` 이면 **딕셔너리의 첫 정책**만 갱신하고 break 했다. 그런데
        클래스 docstring 의 사용 예가 정확히 `sub.end_query(spent)` 였다. 다중 tier 로 구성하면
        나머지 두 tier 는 영원히 페이싱되지 않고, 첫 tier 의 λ 는 다른 tier 의 지출로 오염된다.
        조용히 틀리는 대신 즉시 실패시킨다 (D18 도메인 가드와 같은 철학).
        """
        if tier is None:
            if len(self.policies) != 1:
                raise ValueError(
                    f"tier 를 지정하세요 — 이 라우터는 tier {sorted(self.policies)} 로 "
                    f"구성돼 있고, tier 를 생략하면 어느 페이싱 컨트롤러를 갱신할지 알 수 없습니다. "
                    f"end_query(spent, tier='fast') 처럼 부르십시오")
            tier = next(iter(self.policies))
        pol = self.policies[tier]
        if hasattr(pol, "observe_spend"):
            pol.observe_spend(spent)
        # 다음 step() 이 확실히 '새 문항'으로 판정되게 한다 (호스트가 이 훅을 부르는 경우).
        self._prev_hist_len = -1
        self._last_req = None

    # ---------------- 핵심: 한 스텝 ----------------

    def step(self, prompt, tier: str, call_history: list, model_metadata,
             remaining_budget: float, domain=None) -> Action:
        """이력을 보고 다음 액션 하나를 결정한다 (내부 route()와 등가).

        call_history: [{"model_id": str, "output": ...}, ...]  호출 순서대로
        model_metadata: 후보별 비용 정보 (형식은 `_cost_vector` 참조)
        domain: 문항의 도메인/태스크 (챌린지 공개 데이터가 제공). 생략 시 domain_of 사용
        """
        pol = self.policies[tier]
        feats = self._features(prompt, tier)
        dom = int(self.domain_of(prompt, model_metadata) if domain is None else domain)
        costs = self._cost_vector(prompt, model_metadata)
        self._begin_query_if_new(pol, tier, call_history, costs)
        if self._kind[tier] == "single":
            return self._step_single(pol, tier, feats, dom, costs, call_history,
                                     remaining_budget)
        if self._kind[tier] == "lookahead":
            return self._step_lookahead(pol, tier, prompt, feats, dom, costs, call_history,
                                        remaining_budget)

        # Phase 17: `policy()` 는 페이싱 래퍼(PacedPandora) 또는 맨 PandoraPolicy 를 준다
        # (per_query_budget=True 이면 후자). `pol.inner` 를 무조건 참조하던 코드는 λ 고정
        # 모드에서 AttributeError 로 죽었다 — 두 형태를 모두 받는다.
        inner = getattr(pol, "inner", pol)
        lam = float(inner.lam)

        # 1) 이력을 순서대로 재생해 신념 상태를 복원 (내부 루프와 동일한 갱신 순서)
        bel = inner.make_belief(feats, dom)
        obs = self._replay(prompt, call_history, bel)

        # 2) 예산이 닿고 호출 상한에 걸리지 않는 미개봉 상자
        afford = self._afford(obs, costs, remaining_budget)

        # 3) 개봉 규칙 — route()와 같은 식 (open_order 존중, Phase 15)
        if afford:
            res = bel.reservation(lam * costs)
            max_sigma = max(float(res[m]) for m in afford)      # 잔여 최대 예약값 (정지 기준)
            if getattr(inner, "open_order", "sigma") == "value":
                pbar = bel.pbar()
                m_star = max(afford, key=lambda m: float(pbar[m]) - lam * costs[m])
            else:
                m_star = max(afford, key=lambda m: float(res[m]))
            if not obs:
                # D42: 전략적 기권 — route() 의 '첫 개봉 전 순가치 판정' 을 어댑터에도 포팅.
                # 없으면 allow_abstain=True 라우터에서 두 경로가 다른 정책이 된다.
                if getattr(inner, "allow_abstain", False):
                    pbar = bel.pbar()
                    if max(float(pbar[m]) - lam * float(costs[m]) for m in afford) <= 0.0:
                        return Action("abstain", "")
                return self._request(tier, call_history, m_star)
            best_prize = max(bel.prize(v, m) for m, v in obs.items())
            if best_prize < max_sigma + inner.stop_margin:
                return self._request(tier, call_history, m_star)
        elif not obs:
            # 아무것도 못 여는데 이력도 없다 → 최저가 1회 시도 (route()의 강제 최소 응답)
            act = self._cheapest_fallback(tier, call_history, costs, remaining_budget)
            if act is not None:
                return act
            # ★ Phase 17 규칙 위반 수정: 예전에는 여기서 `Action("answer", 최저가모델)` 을
            # 돌려줬다 — **호출하지 않은 모델을 최종 답으로 지명**하는 것이고 챌린지 규칙
            # ("최종 답은 호출한 후보 출력 중 하나")을 정면으로 위반한다. 부를 수 있는 상자가
            # 없으면 유효한 답이 없으므로 기권을 명시한다 (하네스의 무응답=0점 의미론과 일치).
            return Action("abstain", "")

        # 4) 정지: 연 상자 중 상금 추정 최대
        best = max(obs, key=lambda m: bel.prize(obs[m], m))
        return Action("answer", self.model_ids[best])

    # ---------------- 공용 헬퍼 (D37~D42) ----------------

    def _idx(self, model_id: str) -> int:
        """model_id → 열 인덱스. 미지의 id 는 bare KeyError 대신 원인을 말한다."""
        try:
            return self.index[model_id]
        except KeyError:
            raise KeyError(
                f"후보 모델 {model_id!r} 는 이 라우터가 적합된 모델 목록에 없습니다 "
                f"({self.model_ids}). 호스트 후보 풀이 바뀌었다면 라우터를 재적합해야 합니다"
            ) from None

    def _obs_value(self, prompt, rec, called: list) -> float:
        """검증기 관측 1건. D39: 4-인자 콜백이면 **호출 이력**을 함께 넘긴다.

        교차모델 답 일치(`verifier.DynamicAgreementVerifier`)는 이미 열린 다른 출력을 봐야
        계산되므로, 이 인자가 없으면 검증기의 최대 레버(+0.178 AUC)가 배포에서 원리적으로
        사용 불가능하다. 3-인자 콜백은 그대로 동작한다 (하위호환).
        """
        if self._observe_arity >= 4:
            return float(self.observe(prompt, rec["model_id"], rec, called))
        return float(self.observe(prompt, rec["model_id"], rec))

    def _replay(self, prompt, call_history: list, bel) -> dict:
        """호출 이력을 순서대로 재생해 신념을 복원하고 관측 dict 을 만든다."""
        obs: dict[int, float] = {}
        called: list[str] = []
        for rec in call_history:
            m = self._idx(rec["model_id"])
            if m in obs:
                continue                             # 중복 열람은 재갱신하지 않음
            v = self._obs_value(prompt, rec, list(called))
            called.append(rec["model_id"])
            obs[m] = v
            bel.update(m, v)
        return obs

    def _plan_cost(self, costs, m: int) -> float:
        """계획 단계 비용 — D38: 라우터와 같은 안전마진을 적용한다."""
        return float(costs[m]) * (1.0 + self.cost_margin)

    def _left(self, obs: dict, costs, remaining_budget: float) -> float:
        """이번 문항에서 아직 쓸 수 있는 금액 (D37: 잔여예산 의미론 스위치)."""
        if self.remaining_is_net:
            return float(remaining_budget)
        return float(remaining_budget) - sum(float(costs[m]) for m in obs)

    def _afford(self, obs: dict, costs, remaining_budget: float) -> list[int]:
        """개봉 가능한 미개봉 상자 — 예산·호출상한·거절이력을 모두 반영."""
        if self.max_calls is not None and len(obs) >= self.max_calls:
            return []
        left = self._left(obs, costs, remaining_budget)
        return [m for m in range(len(self.model_ids))
                if m not in obs and m not in self._refused
                and self._plan_cost(costs, m) <= left + 1e-12]

    def _request(self, tier: str, call_history: list, m_star: int) -> Action:
        """호출 액션 발행 + 다음 스텝의 거절 판정을 위한 기록 (D40)."""
        self._last_req = (tier, len(call_history), m_star)
        return Action("call", self.model_ids[m_star])

    def _cheapest_fallback(self, tier, call_history, costs, remaining_budget):
        """최저가 1회 강제 시도 (route() 의 강제 최소 응답). 불가하면 None."""
        order = sorted((m for m in range(len(self.model_ids)) if m not in self._refused),
                       key=lambda m: float(costs[m]))
        for m in order:
            if self._plan_cost(costs, m) <= float(remaining_budget) + 1e-12:
                return self._request(tier, call_history, m)
        return None

    def _begin_query_if_new(self, pol, tier: str, call_history: list, costs) -> None:
        """문항 경계 처리 + 거절 감지 (D40 / D43).

        **거절 감지**: 호스트가 호출을 거부하면 `call_history` 가 자라지 않는다. 직전에 요청한
        (tier, 이력 길이) 로 다시 불렸다면 그 요청은 수락되지 않은 것이므로 해당 상자를 이
        문항에서 제외한다. 초판은 이 상태가 없어 같은 액션을 영원히 재발행했다 —
        채점 하네스에서 무한루프다.

        ★ D43 **생존 모드 부활**: 페이싱의 생존 모드(잔여예산이 남은 문항 최저가 합에 못 미치면
        강제 절약)는 `PacedPandora.min_costs` 를 읽는데, 그 리스트는 `PacedPandora.route()`
        안에서만 채워진다. 어댑터는 `pol.inner` 를 직접 쓰므로 route() 를 거치지 않아
        **배포 경로에서 생존 모드가 영구히 비활성**이었다 (측정: 예산 0.4배에서 무응답 1 → 42).
        회귀 테스트가 `pol.min_costs` 를 손으로 주입하고 있어 잡히지 않았다. 문항당 1회 기록한다.
        """
        n = len(call_history)
        # 1) 직전 요청이 반영되지 않았다 → 호스트가 거절했다
        if self._last_req is not None:
            t_prev, n_prev, m_prev = self._last_req
            if t_prev == tier and n_prev == n:
                self._refused.add(m_prev)
            self._last_req = None
        # 2) 새 문항 판정 — 이력이 비었고 직전 문항이 진행 중이 아니었다
        if n == 0 and self._prev_hist_len != 0:
            self._refused = set()
            if hasattr(pol, "min_costs"):
                pol.min_costs.append(float(np.min(np.asarray(costs, dtype=float))))
        self._prev_hist_len = n

    # ---------------- cascade-routing(1스텝 lookahead) 어댑터 (Phase 17) ----------------

    def _step_lookahead(self, pol, tier, prompt, feats, dom, costs, call_history,
                        remaining_budget) -> Action:
        """`baselines.CascadeRouting` 의 push 규약 — 내부 route() 와 같은 gain 규칙.

            q*      = max 관측 상금,  gain(m) = E[max(q*, X_m)] − q* − λ·c_m
            gain 최대가 양수면 열고, 아니면 멈춘다.
        """
        from .engine.pandora import observable_prize_grid
        bel = pol.make_belief(feats, dom)
        obs = self._replay(prompt, call_history, bel)
        afford = self._afford(obs, costs, remaining_budget)
        if afford:
            q_star = max((bel.prize(v, m) for m, v in obs.items()), default=0.0)
            vals, prob = observable_prize_grid(bel.pbar(), bel.noise)
            gains = {m: float((prob[:, m] * np.maximum(vals[:, m], q_star)).sum())
                     - q_star - float(pol.lam) * costs[m] for m in afford}
            m_star = max(gains, key=gains.get)
            if not obs or gains[m_star] > 0.0:
                return self._request(tier, call_history, m_star)
        elif not obs:
            act = self._cheapest_fallback(tier, call_history, costs, remaining_budget)
            return act if act is not None else Action("abstain", "")
        best = max(obs, key=lambda m: bel.prize(obs[m], m))
        return Action("answer", self.model_ids[best])

    # ---------------- 단일호출 정책용 어댑터 (Phase 17) ----------------

    def _step_single(self, pol, tier, feats, dom, costs, call_history,
                     remaining_budget) -> Action:
        """`baselines.LearnedRouter` 류 '예측 후 확정' 정책의 push 규약.

        `SelectiveRouter` 가 train 내 held-out 에서 학습형을 고른 경우 배포되는 경로다.
        내부 `route()` 와 같은 규칙: argmax(p̄ − λc) 순서로 감당 가능한 첫 상자 1회 호출,
        그 답을 최종 답으로 낸다.
        """
        if call_history:                                   # 이미 호출했다 → 그 답이 최종
            return Action("answer", call_history[0]["model_id"])
        pbar = np.asarray(pol.pred.predict_row(feats, dom), dtype=float)
        for m in np.argsort(-(pbar - float(pol.lam) * costs)):
            m = int(m)
            if m in self._refused:                         # D40: 호스트가 거절한 상자
                continue
            if self._plan_cost(costs, m) <= float(remaining_budget) + 1e-12:
                return self._request(tier, call_history, m)
        return Action("abstain", "")

    # ---------------- 교체 지점 ----------------

    def _features(self, prompt, tier: str) -> np.ndarray:
        """프롬프트 → 인코더 특성. 이미 벡터면 그대로 사용 (합성 세계·테스트 경로)."""
        if isinstance(prompt, np.ndarray):
            return prompt
        key = prompt if isinstance(prompt, str) else repr(prompt)
        if key not in self._encoded:
            # D36: 라우터가 적합 때 쓴 인코더를 그대로 쓴다. `LPBRouter.fit()` 이
            # `ds.text_encoder` 에서 물려받으므로, 데이터셋을 만들 때 인코더를 붙여 두면
            # 여기까지 자동으로 흘러온다 (`loader`/`gate` CLI 가 그렇게 한다).
            enc = getattr(self.routers[tier], "text_encoder", None)
            if enc is None:
                raise ValueError(
                    "프롬프트가 텍스트인데 이 라우터에 text_encoder 가 없습니다. "
                    "특성을 만든 인코더를 실어 보내야 합니다 — 셋 중 하나를 하십시오: "
                    "① 데이터셋 구성 시 `ds.text_encoder = enc` (권장, fit() 이 물려받습니다) "
                    "② `LPBRouter(..., text_encoder=enc)` "
                    "③ step() 에 텍스트 대신 이미 인코딩된 특성 벡터를 넘기기. "
                    "임의의 기본 인코더로 대체하지 않는 이유는, 적합 때와 다른 인코더를 쓰면 "
                    "특성 공간이 달라져 **조용히 다른 정책**이 되기 때문입니다")
            self._encoded[key] = np.asarray(enc.encode([prompt])[0])
        return self._encoded[key]

    def _cost_vector(self, prompt, model_metadata) -> np.ndarray:
        """후보별 이번 문항 호출 비용 (Q3 확정 시 여기만 교체).

        지원 형식
          · np.ndarray / list      : 모델 순서대로의 비용 벡터 (사전 계산 결과)
          · {model_id: float}      : 모델별 비용
          · {model_id: {"cost":…}} : 메타데이터 딕셔너리
        """
        if isinstance(model_metadata, (list, tuple, np.ndarray)):
            return np.asarray(model_metadata, dtype=float)
        out = np.empty(len(self.model_ids), dtype=float)
        for i, mid in enumerate(self.model_ids):
            v = model_metadata[mid]
            out[i] = float(v if np.isscalar(v) else v["cost"])
        return out
