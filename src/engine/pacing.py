"""M3: 쌍대가격 예산 페이싱 (v5 설계도 §5) — v2 (D10 수정).

v1(쿼리 단위 mirror descent)은 지출의 lumpiness(싼 쿼리 다수 + 비싼 승급 소수)에
발진하여 natural 순서에서 λ 폭락 → 후반 파산을 유발했다 (phase4 reflection D10).

v2 컨트롤러 (광고 예산 페이싱의 표준 구조):
  err_t = (누적지출 / B) − (t / T)          # 누적 페이싱 오차 (완만한 신호)
  λ_t   = λ0 · exp(κ⁺·max(err,0) + κ⁻·min(err,0))   # 비대칭: 과소비 억제 > 과소지출 완화
  생존 모드: 잔여예산 < 잔여쿼리 × 최저비용추정 × margin → λ = λ0 × clip (강제 절약)

λ0는 오프라인 리플레이 튜닝값(웜스타트) — natural 순서에서 err≈0이면 λ≈λ0로
정적 정책과 일치(보험료 0), 순서 이동 시에만 개입한다.
"""
import numpy as np


class PacedPandora:
    def __init__(self, inner_factory, lam0: float, k_over: float = 8.0,
                 k_under: float = 2.0, clip: float = 64.0, survival_margin: float = 1.05,
                 name: str = "pandora-paced"):
        self.inner_factory, self.lam0 = inner_factory, lam0
        self.k_over, self.k_under, self.clip = k_over, k_under, clip
        self.survival_margin = survival_margin
        self.name = name
        self.inner = inner_factory(lam0)
        # reset() 없이 호출돼도 죽지 않게 기본 상태를 미리 세운다 (Phase 17: 주최측
        # 하네스가 begin_tier 훅을 호출해 주지 않을 수 있고, 그때 조용히 λ 고정 정책으로
        # 퇴화하는 것이 AttributeError 보다 낫다 — 지출을 모르면 페이싱할 것도 없다).
        self.B, self.T, self.t, self.cum = 0.0, 1, 0, 0.0
        self.min_costs: list[float] = []

    def reset(self, tier: str, budget: float, n_queries: int):
        self.B, self.T = budget, max(n_queries, 1)
        self.t, self.cum = 0, 0.0
        self.min_costs: list[float] = []       # 쿼리별 최저 호출비용 이력
        self.inner = self.inner_factory(self.lam0)
        if hasattr(self, "stats"):             # 선택적 계측: reset을 넘어 집계 유지
            self.inner.stats = self.stats

    def observe_spend(self, spent: float):
        self.t += 1
        self.cum += spent
        if self.B <= 0:
            return
        # 생존 모드 (v3, D15 수정): 남은 쿼리 전부에 최저가 1회조차 빠듯하면 강제 절약.
        #  - 예비비 추정: 전역 min(낙관 편향 — 로그정규 입력에서 과소 추정)이 아니라
        #    쿼리별 최저비용의 0.9 분위수 (보수적)
        #  - 생존 λ: λ0×clip(λ0이 작으면 무력)이 아니라 절대값 1.2/c_min —
        #    모든 상자의 예약지수 σ = 1−λc/p̄ ≤ 0 이 되어 정확히 1회 호출 후 정지
        remaining_b = self.B - self.cum
        remaining_q = self.T - self.t
        if self.min_costs and remaining_q > 0:
            c_reserve = float(np.quantile(self.min_costs, 0.9))
            if remaining_b < remaining_q * c_reserve * self.survival_margin:
                c_floor = max(min(self.min_costs), 1e-9)
                self.inner.lam = max(self.lam0 * self.clip, 1.2 / c_floor)
                return
        err = self.cum / self.B - self.t / self.T
        k = self.k_over if err > 0 else self.k_under
        lam = self.lam0 * float(np.exp(k * err))
        self.inner.lam = float(np.clip(lam, self.lam0 / self.clip, self.lam0 * self.clip))

    def route(self, sess, features, domain) -> int:
        self.min_costs.append(float(np.min(sess.costs)))
        return self.inner.route(sess, features, domain)
