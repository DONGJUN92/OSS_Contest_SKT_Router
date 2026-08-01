"""베이스라인 정책 3종 (PROJECT_PLAN.md Phase 1).

① AllModel(최대/최소): 라우팅 없는 상·하한 기준선.
② StaticCascade: v4식 정적 threshold 순차 승급 — 최저가부터 호출, 검증기 점수 < τ면
   다음 모델로 승급. τ는 train fold에서 그리드 탐색으로 tier별 튜닝.
이후 모든 Phase의 개선은 이 세 줄의 점수표 대비로만 주장한다.
"""
import numpy as np


class AllModel:
    """항상 지정 모델만 호출 (예산 부족 시 호출 실패 → 하네스가 위반 처리)."""

    def __init__(self, m_idx: int, name: str):
        self.m_idx, self.name = m_idx, name

    def route(self, sess, features, domain) -> int:
        sess.call(self.m_idx)
        return self.m_idx


class StaticCascade:
    """비용 오름차순 사다리 순차 승급, 정적 threshold τ (v4 스타일)."""

    def __init__(self, cost_order: list[int], tau: float, name: str = "static-cascade"):
        self.order, self.tau, self.name = cost_order, tau, name

    def route(self, sess, features, domain) -> int:
        best, best_v = None, -1.0
        for m in self.order:
            v = sess.call(m)
            if v is None:          # 예산 부족 → 지금까지 최선으로 종료
                break
            if v > best_v:
                best, best_v = m, v
            if v >= self.tau:      # 만족 → 조기 종료
                return m
        return best if best is not None else self.order[0]


def tune_cascade_tau(ds, train_idx, cfg, tier: str, budget: float) -> float:
    """train fold에서 τ 그리드 탐색 (베이스라인의 공정한 튜닝 — 테스트 fold 미접촉)."""
    from src.harness import run_tier
    from src.cost_mirror import cost_matrix
    order = list(np.argsort(cost_matrix(ds)[train_idx].mean(axis=0)))
    best_tau, best_q = None, -1.0
    for tau in cfg["eval"]["cascade_tau_grid"]:
        r = run_tier(ds, train_idx, StaticCascade(order, tau), tier, budget)
        if r.mean_quality > best_q:
            best_q, best_tau = r.mean_quality, tau
    return best_tau


class BudgetCascade:
    """**예산 인지** 정적 cascade — 공정한 강 베이스라인 (Phase 17, 레드팀 CRITICAL).

    지적: `StaticCascade` 는 예산을 전혀 보지 않고 사다리를 올라가므로 저예산 tier 에서
    앞쪽 쿼리가 예산을 태워 뒤쪽이 **파산**한다(무응답 0점). LPB 가 cascade 를 크게 앞선
    보고 격차의 상당 부분이 "우리가 파산을 막았다"가 아니라 "베이스라인이 파산했다"에서
    왔다 — 즉 strawman 이다. 문항당 허용액 한 줄만 넣으면 사라지는 격차를 신규성으로
    주장할 수 없다.

    이 정책은 남은 예산을 남은 쿼리로 균등 배분한 허용액 안에서만 승급한다(광고 페이싱의
    가장 단순한 형태). 최저가 1회는 허용액과 무관하게 시도해 무응답을 구조적으로 막는다.
    LPB 의 M3 페이싱과 같은 정보(잔여 예산·잔여 쿼리)만 쓰므로 비교가 대등하다.
    """

    def __init__(self, cost_order: list[int], tau: float, name: str = "budget-cascade"):
        self.order, self.tau, self.name = cost_order, tau, name
        self.B, self.T, self.t, self.cum = 0.0, 1, 0, 0.0

    def reset(self, tier: str, budget: float, n_queries: int):
        self.B, self.T, self.t, self.cum = budget, max(n_queries, 1), 0, 0.0

    def observe_spend(self, spent: float):
        self.t += 1
        self.cum += spent

    def route(self, sess, features, domain) -> int:
        allow = (self.B - self.cum) / max(self.T - self.t, 1) if self.B > 0 else float("inf")
        best, best_v, spent = None, -1.0, 0.0
        for m in self.order:
            # 최저가 1회는 허용액을 무시하고 시도 (파산 방지). 이후 승급만 허용액이 통제.
            if best is not None and spent + sess.costs[m] > allow + 1e-12:
                break
            v = sess.call(m)
            if v is None:
                break
            spent += sess.costs[m]
            if v > best_v:
                best, best_v = m, v
            if v >= self.tau:
                return m
        return best if best is not None else self.order[0]


def tune_budget_cascade_tau(ds, train_idx, cfg, tier: str, budget: float) -> float:
    """train fold τ 그리드 탐색 — StaticCascade 튜닝과 동급 공정."""
    from src.harness import run_tier
    from src.cost_mirror import cost_matrix
    order = list(np.argsort(cost_matrix(ds)[train_idx].mean(axis=0)))
    best_tau, best_q = None, -1.0
    for tau in cfg["eval"]["cascade_tau_grid"]:
        r = run_tier(ds, train_idx, BudgetCascade(order, tau), tier, budget)
        if r.mean_quality > best_q:
            best_q, best_tau = r.mean_quality, tau
    return best_tau


class CascadeRouting:
    """결합 라우팅+캐스케이딩 베이스라인 (Dekoninck et al., arXiv:2410.10347 계열).

    Phase 17에 추가 — 레드팀 지적: **같은 행동공간에서 최적성을 주장하는, 가장 가까운
    공개 선행 경쟁자**가 `참고자료/` 와 `baselines/` 양쪽에 모두 없었다. 지수(index) 정책이
    아니라 **1스텝 lookahead 로 열기/멈추기를 함께 결정**하는 구조라 Weitzman 과 직접
    대비된다.

    결정 규칙 (매 스텝):
        q*      = 지금까지 관측한 상금의 최대
        gain(m) = E[max(q*, X_m)] − q* − λ·c_m         (X_m = 관측 상금 E[Q|v])
        gain 최대인 m 이 양수면 열고, 아니면 멈춘다.
    X_m 의 분포는 LPB 와 **똑같은** 신념·검증기 채널에서 만든다(`observable_prize_grid`)
    — 그래서 이 비교는 신념 품질 차이가 아니라 **결정 구조** 차이만 재는 대조군이다.

    Weitzman 과의 관계: 독립·정확관측 상자에서는 지수 정책이 전역 최적이고 1스텝
    lookahead 는 근시안적이다. 반대로 상자가 상관되고 관측이 잡음이면 지수 최적성이
    깨지므로 어느 쪽이 이길지는 **선험적으로 알 수 없다** → 측정 대상이다.
    """

    def __init__(self, make_belief, lam: float, name: str = "cascade-routing"):
        self.make_belief, self.lam, self.name = make_belief, lam, name

    def route(self, sess, features, domain) -> int:
        from src.engine.pandora import observable_prize_grid
        bel = self.make_belief(features, domain)
        costs = np.asarray(sess.costs, dtype=float)
        unopened = set(range(len(costs)))
        obs: dict[int, float] = {}
        while unopened:
            afford = [m for m in unopened if sess.can_afford(m)]
            if not afford:
                break
            q_star = max((bel.prize(v, m) for m, v in obs.items()), default=0.0)
            vals, prob = observable_prize_grid(bel.pbar(), bel.noise)
            gains = {}
            for m in afford:
                e_max = float((prob[:, m] * np.maximum(vals[:, m], q_star)).sum())
                gains[m] = e_max - q_star - self.lam * costs[m]
            m_star = max(gains, key=gains.get)
            if obs and gains[m_star] <= 0.0:
                break
            v = sess.call(m_star)
            if v is None:
                break
            unopened.discard(m_star)
            obs[m_star] = v
            bel.update(m_star, v)
        if not obs:
            m = int(np.argmin(costs))
            v = sess.call(m)
            if v is not None:
                obs[m] = v
        if not obs:
            return 0
        return max(obs, key=lambda m: bel.prize(obs[m], m))


class LearnedRouter:
    """학습형 단일호출 라우터 — RouteLLM류 지배적 OSS 패러다임의 대표 비교군 (#3).

    예측기(predictor.predict_row(x, dom) → 모델별 정답확률 p̄)가 있으면, 호출 **전에**
    argmax(p̄_m − λ·c_m)로 상자 하나를 고르고 한 번 호출한 뒤 그 답을 낸다. 순차 개봉·
    관측 후 재지수·정지 판정이 없다 — LPB의 Weitzman 순차 구조와 대비되는 '예측 후 확정'
    구조. 예측기가 vote 상자 열([model×N])도 후보로 보므로 동일 행동공간에서 경쟁한다.
    """

    def __init__(self, predictor, lam: float, name: str = "learned"):
        self.pred, self.lam, self.name = predictor, lam, name

    def route(self, sess, features, domain) -> int:
        pbar = np.asarray(self.pred.predict_row(features, domain), dtype=float)
        for m in np.argsort(-(pbar - self.lam * sess.costs)):   # 최고 가치-비용 순
            if sess.call(int(m)) is not None:                    # 감당 가능한 첫 상자 1회 호출
                return int(m)
        return 0


def tune_learned_lambda(predictor, ds, train_idx, budget: float, iters: int = 40) -> float:
    """train에서 단일호출 총지출 ≤ budget이 되는 최소 λ 이분탐색 (cascade τ 튜닝과 동급 공정)."""
    from src.cost_mirror import cost_matrix
    cmat = cost_matrix(ds)
    P = np.stack([np.asarray(predictor.predict_row(ds.features[i], int(ds.domains[i])),
                             dtype=float) for i in train_idx])
    lo, hi = 0.0, 1e5
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        chosen = np.argmax(P - mid * cmat[train_idx], axis=1)
        if cmat[train_idx, chosen].sum() > budget:
            lo = mid
        else:
            hi = mid
    return hi


class SelectiveRouter:
    """메타 라우터 (Phase 15) — LPB와 학습형 단일호출 라우터 중 train 내 held-out으로 나은 것
    을 골라 배포한다. 3모델·실텍스트에서 학습형이 LPB에 대등/우위인 한계를, 단일 정책 개선
    (belief-swap·value-order)이 실패(중립/D17에 포섭)한 뒤 **원칙적 model-selection**으로 극복.

    thesis 정합: **LPB가 주 정책**이고 학습형은 명확한 대등우위 체제의 보험이다 — LPB를 기본
    으로 두고 learned가 tr_sel에서 `margin` 이상 이길 때만 전환한다(D21 교훈: 잡음을 승리로
    오인 방지). LPB가 지배하는 체제(상자 많음·순차 탐색 가치)에선 LPB, 소수상자·실텍스트에선
    learned가 뽑힌다. 실측: 어느 baseline에도 (margin 보정 후) 지지 않음.

    실데이터·상자 수 무관하게 동작 — 프로토콜은 테스트 미접촉(tr → fit/sel 분할).
    """

    #: ★ D66 (외부 레드팀 2026-08-01) — 채택 규칙의 **목적**을 명시한다.
    #:
    #: 지적: 제출 기본값(selective3 0.5057)이 **자기 후보 목록 안의** cascade-routing
    #: (0.5091)에 지배당한다. 원인은 절대 문턱 0.01 이고, 저장소는 그 비용을 이미
    #: 측정해 두었다(§4.5.1 "종합 +0.005 수준").
    #:
    #: 저장소의 방어는 "유리한 방향으로 문턱을 조정하지 않는다"였다. **그 규칙은 옳지만
    #: 여기 적용하는 것은 범주 오류다** — 규칙의 목적은 *결과를 본 뒤* 유리하게 고치는 것을
    #: 막는 것이고, held-out 선택셋에서 argmax 를 고르는 것은 p-hacking 이 아니라 표준
    #: 모델 선택이다. 보수적 절대 문턱이 정당한 경우는 전환 비용이 **비대칭**일 때인데
    #: 대회 채점에는 비대칭이 없다(목적함수 = 기대 점수).
    #:
    #: 그리고 초판 규칙에는 더 기본적인 결함이 있었다 — **표준오차를 아예 보지 않는다.**
    #: 게이트는 D26 에서 쌍대 SE 문턱을 도입했는데 `SelectiveRouter` 는 평균 비교뿐이다.
    #: 즉 절대 문턱 0.01 이 표본 크기와 무관한 고정값으로 두 역할(잡음 방어 + 보수성)을
    #: 겸하고 있었다.
    #:
    #: 그래서 세 모드를 노출한다. **기본값 변경은 측정 후에만** 한다 (D21).
    #:   "absolute"  : max(margin, se_k·쌍대SE)   — 게이트와 같은 규약
    #:   "paired_se" : se_k·쌍대SE 만             — 절대 바닥 제거
    #:   "argmax"    : 0                          — held-out 최고를 그대로 (기대점수 최대화)
    #:
    #: ★★ **측정 결과가 감사의 진단을 반박했다 (Phase 21, 그대로 적는다).**
    #: 감사는 "절대 문턱 0.01 이 전환을 막는다"고 진단했다. 실측하니
    #:   · `paired_se`(절대 바닥 제거)의 효과는 **정확히 0.0000** — 세 tier 중 둘에서
    #:     구속하는 것은 절대 문턱이 아니라 **쌍대 SE**(0.0085~0.0183)였다.
    #:   · 그리고 감사 제안대로 SE 항을 **추가**하자(`se_k` 0 → 1) selective3 가
    #:     **0.5057 → 0.5007 로 내려갔다.** 전환 4건 중 2건이 막혔고 그 둘은 옳은 전환이었다.
    #: 즉 이 체제에서 문제는 문턱이 **높아서**가 아니라, 문턱을 더 높이면 손해라는 것이다.
    #: 그래서 `se_k` 기본값을 **0.0 으로 둔다** — 이것이 Phase 20 까지의 동작(평균 격차 >
    #: margin)과 정확히 같고, 커밋된 아티팩트를 그대로 재현한다. SE 는 **인자로만** 남긴다.
    MARGIN_MODES = ("absolute", "paired_se", "argmax")

    def __init__(self, cfg, n_domains: int, seed: int = 0, sel_frac: float = 0.3,
                 margin: float = 0.01, margin_mode: str = "absolute",
                 se_k: float = 0.0, **lpb_kwargs):
        self.cfg, self.n_dom, self.seed = cfg, n_domains, seed
        self.sel_frac, self.margin, self.lpb_kwargs = sel_frac, margin, lpb_kwargs
        if margin_mode not in self.MARGIN_MODES:
            raise ValueError(f"margin_mode 는 {self.MARGIN_MODES} 중 하나여야 합니다")
        self.margin_mode, self.se_k = margin_mode, float(se_k)

    def _fit_learned(self, ds, idx, tier):
        from src.encoder import DiscLR                        # 라이브러리 부품 (Phase 17 이관)
        from src.harness import tier_budgets
        disc = DiscLR().fit(ds.features[idx], ds.domains[idx], ds.quality[idx])
        lam = tune_learned_lambda(disc, ds, idx, tier_budgets(ds, idx, self.cfg)[tier])
        return LearnedRouter(disc, lam)

    def _fit_lookahead(self, ds, idx, tier, lpb):
        """cascade-routing 후보 — LPB 와 **같은 신념**을 쓰고 결정 규칙만 다르다 (Phase 17).

        추가 이유(정직 기록): 이 베이스라인을 공정하게 붙여 재보니 실전 구성(A.X 3모델·
        Q2=No·실텍스트)에서 **LPB 를 앞섰다**(종합 0.5136 vs 0.5017, 호출은 더 적음).
        thesis 를 방어하는 대신 후보로 승격시켜 **데이터가 고르게** 한다 — Phase 15 가
        학습형을 후보로 승격시킨 것과 같은 처리다.
        """
        from src.harness import tier_budgets
        from src.engine.pandora import tune_lambda_quality, FINE_MULTS
        mb = lpb._factory(1.0).make_belief
        b = tier_budgets(ds, idx, self.cfg)[tier]
        # ★ 대등 튜닝 (Phase 19): 이전에는 이 후보만 약한 `tune_lambda_replay`(예산을 채우는
        # 최소 λ)를 썼고 LPB 는 `tune_lambda_quality`(세밀격자 + 표준오차 문턱)를 썼다. 즉
        # 비교가 우리 쪽으로 기울어 있었다. 같은 튜너를 준다 — 그 결과 이 후보가 더 세지면
        # 메타 선택이 그것을 고르므로 **제출 시스템은 손해가 없고 비교만 정직해진다.**
        lam = tune_lambda_quality(lambda l: CascadeRouting(mb, l), ds, idx, b,
                                  mults=FINE_MULTS, se_k=1.0)
        return CascadeRouting(mb, lam)

    def fit(self, ds, tr, tier):
        from src.router import LPBRouter
        from src.harness import run_tier, tier_budgets
        rng = np.random.default_rng(1000 + self.seed)
        perm = rng.permutation(tr)
        n_fit = int((1 - self.sel_frac) * len(perm))
        tr_fit, tr_sel = perm[:n_fit], perm[n_fit:]
        lpb = LPBRouter(self.cfg, self.n_dom, seed=self.seed, **self.lpb_kwargs).fit(ds, tr_fit, tier)
        cands = {"lpb": lpb.policy(),
                 "learned": self._fit_learned(ds, tr_fit, tier),
                 "cascade_routing": self._fit_lookahead(ds, tr_fit, tier, lpb)}
        b_sel = tier_budgets(ds, tr_sel, self.cfg)[tier]
        # D66: **쿼리별** 점수를 보관한다 — 쌍대 차이의 표준오차를 구하려면 평균만으로는
        # 부족하다. 같은 쿼리에 대한 두 정책의 차이는 쿼리 난이도 분산을 상쇄하므로
        # 같은 표본으로 훨씬 작은 진짜 차이를 잡아낸다 (게이트 D26 과 같은 논리).
        pq = {k: np.asarray(run_tier(ds, tr_sel, p, tier, b_sel).per_query, dtype=float)
              for k, p in cands.items()}
        q = {k: float(v.mean()) for k, v in pq.items()}
        best = max((k for k in q if k != "lpb"), key=lambda k: q[k])
        d = pq[best] - pq["lpb"]
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("inf")
        # ★ `absolute` 는 **덧셈**이지 `max` 가 아니다. 게이트는 `max(margin, se_k·SE)` 를 쓰지만
        # 여기서 그렇게 하면 `se_k·SE ≥ 0` 이라 **음수 margin 이 0 으로 바닥**을 맞는다 —
        # `margin=-1e9` 로 대안을 강제하는 기존 API 계약(회귀 테스트가 쓰는 관용구)이 조용히
        # 깨진다. 덧셈이면 `se_k=0` 에서 `need == margin` 으로 **Phase 20 규칙과 정확히 동일**
        # 하고, `se_k>0` 이면 절대 바닥 위에 통계적 여유를 얹은 보수형이 된다.
        # (조용히 다른 정책이 되느니 계약을 보존한다 — D18·D41 과 같은 철학.)
        need = {"absolute": self.margin + self.se_k * se,
                "paired_se": self.se_k * se,
                "argmax": 0.0}[self.margin_mode]
        self.chosen = best if float(d.mean()) > need else "lpb"
        self.sel_scores = {k: round(v, 4) for k, v in q.items()}
        self.sel_detail = {"best_alt": best, "gap": round(float(d.mean()), 4),
                           "paired_se": round(se, 4), "required": round(need, 4),
                           "margin_mode": self.margin_mode, "n_sel": int(len(d))}
        # 선택된 정책을 full tr로 재적합 (더 많은 데이터)
        if self.chosen == "lpb":
            self._pol = LPBRouter(self.cfg, self.n_dom, seed=self.seed,
                                  **self.lpb_kwargs).fit(ds, tr, tier)
        elif self.chosen == "learned":
            self._pol = self._fit_learned(ds, tr, tier)
        else:
            full = LPBRouter(self.cfg, self.n_dom, seed=self.seed,
                             **self.lpb_kwargs).fit(ds, tr, tier)
            self._pol = self._fit_lookahead(ds, tr, tier, full)
            self._belief_owner = full
        return self

    def policy(self):
        return self._pol.policy() if self.chosen == "lpb" else self._pol

    # ── 제출 어댑터 호환 (Phase 17) ────────────────────────────────────────────
    # `SubmissionRouter` 는 라우터에서 집단 축 정보를 읽는다. LPB 가 선택되면 그 값을
    # 그대로 위임하고, 학습형이 선택되면 집단 축이 없으므로 (1,M) 로 보고한다 —
    # 이것이 없으면 Phase 15 가 "한계 극복"이라 발표한 정책이 **배포 불가**였다.

    @property
    def irt(self):
        if self.chosen == "lpb":
            return self._pol.irt
        return {"b": np.zeros((1, 1))}          # 단일호출 정책 = 집단 축 없음

    @property
    def clusterer(self):
        return getattr(self._pol, "clusterer", None) if self.chosen == "lpb" else None

    @property
    def text_encoder(self):
        return getattr(self._pol, "text_encoder", None)
