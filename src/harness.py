"""오프라인 평가 하네스 — 챌린지 채점 구조의 리플레이 (PROJECT_PLAN.md Phase 1).

채점 규칙의 강제:
  1. 정책은 Session.call()로만 모델을 호출할 수 있고, 호출 즉시 비용이 과금된다.
  2. 최종 답은 '호출한 모델의 출력' 중에서만 선택 가능 (Pandora keep-best-opened 동형).
  3. 정책은 진짜 품질을 볼 수 없고 검증기 잡음 관측만 본다.
  4. tier 예산 초과 호출은 거부된다. 아무것도 못 부른 쿼리는 최저가 모델이 강제되고
     위반(violation)으로 기록된다.
"""
from dataclasses import dataclass, field
import numpy as np
from .schema import Dataset
from .cost_mirror import cost_matrix


@dataclass
class Session:
    """쿼리 1건에 대한 정책↔하네스 상호작용 프로토콜.

    비용의 두 얼굴 (Phase 17): `costs` 는 **정책이 호출 전에 보는** 비용이고
    `charge_costs` 는 **하네스가 실제로 과금하는** 비용이다. 대회 런타임은 호출 전에
    단가(비용 정책)만 주고 실제 토큰 수는 호출 후에 주므로, decode 길이를 모르는 상태의
    비용은 추정치다 (`src/cost_model.py`). 둘을 같게 두면 오라클 비용 가정이 된다.
    """
    costs: np.ndarray          # (M,) 정책이 보는 비용 (사전 추정치일 수 있다)
    verifier_row: np.ndarray   # (M,) 잡음 관측 (호출한 모델만 열람 가능)
    remaining_budget: float
    called: list[int] = field(default_factory=list)
    spent: float = 0.0
    # 대회 상세가 예산과 **별도로** 명시한 '남은 호출 수' 상한 (Q 확인 대상). None=무제한.
    # 예산은 비용 제약, max_calls는 호출 횟수 제약 — 둘 중 먼저 걸리는 것이 정지를 강제한다.
    max_calls: int | None = None
    # 실제 과금 비용 (None = costs 와 동일 = 오라클 비용 체제).
    charge_costs: np.ndarray | None = None
    # 추정 불확실성 안전마진: 계획 단계에서 비용을 (1+margin) 배로 보수적으로 잡는다.
    cost_margin: float = 0.0
    # 추정이 실제보다 낮아 호스트가 거부한 횟수 (계측용 — 파산 원인 분해에 쓴다)
    rejected: int = 0
    # 동적 관측 훅 (Phase 19): `observe_fn(m_idx, called) -> v`.
    # 왜 필요한가 — 정적 `verifier_row` 는 "이 상자의 점수"를 미리 고정하지만, 실제로 가장
    # 강한 신호인 **교차모델 답 일치**는 *이미 어떤 상자를 열었는지*에 따라 달라진다. 상한
    # 진단에서 남은 달성 가능 여지의 88% 가 관측 채널에 있었고(+0.0593), 그 여지는 2회 이상
    # 여는 tier(balanced·premium)에 집중돼 있다 — 즉 이 훅이 그 여지에 접근하는 경로다.
    observe_fn: object | None = None

    def _charge_vec(self) -> np.ndarray:
        return self.costs if self.charge_costs is None else self.charge_costs

    def _at_call_cap(self) -> bool:
        return self.max_calls is not None and len(self.called) >= self.max_calls

    def can_afford(self, m_idx: int) -> bool:
        if m_idx in self.called:
            return True                       # 재열람은 무과금·무카운트 (사전 계산 결과 재조회)
        if self._at_call_cap():
            return False                      # 호출 수 상한 도달 → 새 상자 개봉 불가
        plan = self.costs[m_idx] * (1.0 + self.cost_margin)
        return plan <= self.remaining_budget - self.spent + 1e-12

    def _observe(self, m_idx: int) -> float:
        if self.observe_fn is None:
            return self.verifier_row[m_idx]
        return float(self.observe_fn(m_idx, list(self.called)))

    def call(self, m_idx: int) -> float | None:
        """모델 호출. 예산 부족 또는 호출 상한 도달 시 None. 성공 시 검증기 관측 반환."""
        if m_idx in self.called:
            return self._observe(m_idx)      # 중복 호출은 재과금하지 않음 (사전 계산 결과 재열람)
        if not self.can_afford(m_idx):       # 예산 ∧ 호출 상한 동시 검사
            return None
        actual = float(self._charge_vec()[m_idx])
        if actual > self.remaining_budget - self.spent + 1e-12:
            # 정책은 추정 비용으로 감당 가능하다고 판단했지만 실제 비용이 예산을 넘는다
            # → 호스트가 거부한다. 안전마진의 존재 이유이며, 이 카운터가 그 값을 보여준다.
            self.rejected += 1
            return None
        self.called.append(m_idx)
        self.spent += actual
        return self._observe(m_idx)


@dataclass
class TierResult:
    tier: str
    mean_quality: float
    total_cost: float
    budget: float
    unanswered: int            # 예산 소진으로 아무 모델도 못 부른 쿼리 수
    calls_per_query: float
    rejected: int = 0          # 비용 과소추정으로 호스트가 거부한 호출 수 (Phase 17)
    # 쿼리별 품질 (Phase 18) — 정책 간 **쌍대 비교의 표준오차**를 재려면 평균만으로는 부족하다.
    # 게이트가 잡음을 승리로 오인하는 것을 막는 데 쓴다 (D21 교훈의 재적용).
    per_query: np.ndarray | None = None


def run_tier(ds: Dataset, idx: np.ndarray, policy, tier: str, budget: float,
             bankruptcy: str = "zero_quality", per_query: bool = False,
             max_calls: int | None = None, est_costs: np.ndarray | None = None,
             cost_margin: float = 0.0, dyn_observe=None) -> TierResult:
    """정책을 쿼리 순서대로 리플레이. policy.route(session, features, domain) -> 최종 모델 idx.

    bankruptcy (예산 소진 시 처리 — 실제 규칙은 Phase 0 Q6로 확인 중):
      - "zero_quality"   : 무응답 = 0점, 과금 없음 (보수적 기본값 — 파산 정책이 정직하게 벌점)
      - "force_cheapest" : 최저가 강제 호출 + 초과 과금 허용 (관대한 대안)

    per_query (B5 / Phase 0 Q5): budget의 의미론
      - False: 테스트셋 **전역 총액** (기본) — 한 쿼리의 절약이 다음 쿼리로 이월된다.
               쿼리 간 배분 문제가 존재하므로 M3 페이싱이 shadow price를 조절한다.
      - True : **문항당 상한** — 매 쿼리가 동일한 예산으로 시작하고 이월이 없다.
               배분 문제 자체가 사라지므로 페이싱은 무의미해지고 λ 고정이 정답이다.
    """
    cmat = cost_matrix(ds)
    # est_costs (Phase 17): 정책이 호출 전에 보는 비용. None 이면 실제 비용 = 오라클 가정.
    pol_cmat = cmat if est_costs is None else np.asarray(est_costs, dtype=float)
    quality_sum, spent_total, unanswered, calls, rejected = 0.0, 0.0, 0, 0, 0
    per_q = np.zeros(len(idx), dtype=float)
    cheapest = int(np.argmin(cmat[idx].mean(axis=0)))
    if hasattr(policy, "reset"):
        policy.reset(tier=tier, budget=budget * (len(idx) if per_query else 1),
                     n_queries=len(idx))

    for pos, i in enumerate(idx):
        remaining = budget if per_query else budget - spent_total
        sess = Session(costs=pol_cmat[i], verifier_row=ds.verifier[i],
                       remaining_budget=remaining, max_calls=max_calls,
                       charge_costs=None if est_costs is None else cmat[i],
                       cost_margin=cost_margin,
                       # dyn_observe(i) -> observe_fn(m, called). 정적 행렬 대신 호출 이력에
                       # 의존하는 관측을 쓸 때만 지정한다 (Phase 19 동적 교차모델 일치).
                       observe_fn=None if dyn_observe is None else dyn_observe(int(i)))
        choice = policy.route(sess, ds.features[i], int(ds.domains[i]))
        rejected += sess.rejected
        if not sess.called:
            unanswered += 1
            if bankruptcy == "force_cheapest":
                sess.called.append(cheapest)
                sess.spent += cmat[i, cheapest]
                choice = cheapest
            else:                                  # zero_quality: 무응답 0점
                spent_total += sess.spent
                if hasattr(policy, "observe_spend"):
                    policy.observe_spend(sess.spent)
                continue
        if choice not in sess.called:             # 규칙 강제: 호출한 출력 중에서만 선택
            choice = max(sess.called, key=lambda m: sess.verifier_row[m])
        per_q[pos] = ds.quality[i, choice]
        quality_sum += ds.quality[i, choice]
        spent_total += sess.spent
        calls += len(sess.called)
        if hasattr(policy, "observe_spend"):
            policy.observe_spend(sess.spent)

    return TierResult(tier, quality_sum / len(idx), spent_total, budget,
                      unanswered, calls / len(idx), rejected, per_q)


def tier_budgets(ds: Dataset, idx: np.ndarray, cfg: dict,
                 per_query: bool = False) -> dict[str, float]:
    """tier 예산 = frac × (전량 최대모델 호출 시 총비용). Phase 0 Q5 답변 시 교체.

    per_query=True (B5): 같은 frac을 **쿼리 1건 기준**으로 환산한 문항당 상한.
    전역 총액을 쿼리 수로 나눈 값과 같으므로 두 의미론의 총지출 상한이 일치하고,
    따라서 페이싱의 가치만 분리해 비교할 수 있다.
    """
    cmat = cost_matrix(ds)
    max_model_total = cmat[idx].max(axis=1).sum()
    denom = len(idx) if per_query else 1
    return {t: v["budget_frac"] * max_model_total / denom for t, v in cfg["tiers"].items()}


def combined_score(results: dict[str, TierResult], cfg: dict) -> float:
    """챌린지 종합 점수 재현: tier 가중 평균 품질 (저예산 가중치 높음)."""
    return sum(cfg["tiers"][t]["weight"] * r.mean_quality for t, r in results.items())
