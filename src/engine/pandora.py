"""M2: 베이지안 재지수 Weitzman 엔진 (v5 설계도 §4).

정책 4단계:
  1. 미개봉 상자의 예약지수 σ_m = 1 − λ c_m / p̄_m (Bernoulli 상금 닫힌형)
  2. 최고 지수 상자 열기 → 검증기 관측 v
  3. θ 그리드 사후 갱신 (우도비) → 전 상자 p̄ 재계산 (상관 Pandora의 핵심)
  4. 정지: max 관측 상금 추정 ≥ max 잔여 σ → 연 상자 중 최고 답 반환

신념(belief)과 지수 엔진을 분리 — GridBelief(IRT, 갱신 가능) / FixedBelief(판별식,
갱신 구조적 불가) 를 같은 엔진에 꽂아 ablation이 공정해진다.
"""
import numpy as np

from .prize import bernoulli_reservation, beta_reservation, perbox_reservation

Z_GRID = np.linspace(-4.0, 4.0, 41)          # 표준화 θ 그리드
_PRIOR_W = np.exp(-0.5 * Z_GRID ** 2)
_PRIOR_W = _PRIOR_W / _PRIOR_W.sum()

# 검증기 관측 v 의 그리드 (이 코드베이스의 v 는 항상 [0,1]).
# 관측 상금 예약값(`prize_reservation="observable"`)에서 X=E[Q|v] 의 분포를 이산화하는 데 쓴다.
V_GRID = np.linspace(0.0, 1.0, 41)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def observable_prize_grid(pbar_vec: np.ndarray, noise):
    """상자별 **관측 상금** X=E[Q|v] 의 이산 분포 → (지지 (L,M), 확률 (L,M)).

        v ~ f_m(v) = p̄_m·f(v|1) + (1−p̄_m)·f(v|0)   (적합된 검증기 채널)
        X_m = prize(v, m) = P(y=1|v)                 (정지 규칙·최종 선택이 쓰는 바로 그 값)

    신념 종류(2PL / Beta / GRM)와 무관하게 p̄=E[Q] 만 받으므로 두 신념 클래스가 공유한다.
    이진이 아닌 품질에서는 채널이 (y=1 대비 y=0) 이진 조건분포로 적합돼 있으므로 근사이며,
    L=2(이진)에서는 정확하다 — `GradedGridBelief` 의 보간 근사와 같은 성격의 기록이다.
    """
    p = np.clip(np.asarray(pbar_vec, dtype=float), 1e-9, 1.0 - 1e-9)
    M, L = len(p), len(V_GRID)
    vals = np.empty((L, M))
    prob = np.empty((L, M))
    for m in range(M):
        l1, l0 = noise.liks_grid(V_GRID, m)
        f = p[m] * l1 + (1.0 - p[m]) * l0
        tot = float(f.sum())
        prob[:, m] = f / tot if tot > 1e-300 else 1.0 / L
        vals[:, m] = noise.p_correct_grid(V_GRID, m)
    return vals, prob


class NoiseModel:
    """검증기 관측 채널: P(y=1|v) 캘리브레이션 + 우도비 f(v|y=1)/f(v|y=0).

    train fold의 (v, y) 쌍으로만 적합 — 생성기의 잡음 모수를 엔진이 훔쳐보지 않음.
    """

    def fit(self, v: np.ndarray, y: np.ndarray, steps: int = 2000, lr: float = 0.5):
        w, b = 0.0, 0.0
        for _ in range(steps):
            p = _sigmoid(w * v + b)
            w -= lr * float(((p - y) * v).mean())
            b -= lr * float((p - y).mean())
        self.w, self.b = w, b
        # 클래스별 적률을 y로 **가중**해 구한다 — y∈{0,1}이면 기존 마스킹과 정확히
        # 동일하고, Q4=연속(y∈[0,1])에서도 부분점수를 그대로 흡수한다 (B4).
        self.m1, self.s1 = self._moments(v, y)
        self.m0, self.s0 = self._moments(v, 1.0 - y)
        return self

    @staticmethod
    def _moments(v: np.ndarray, wt: np.ndarray) -> tuple[float, float]:
        tot = float(wt.sum())
        if tot <= 1e-9:                                # 한쪽 클래스 부재 → 전역으로 폴백
            return float(v.mean()), max(float(v.std()), 1e-3)
        m = float((wt * v).sum() / tot)
        var = float((wt * (v - m) ** 2).sum() / tot)
        return m, max(np.sqrt(var), 1e-3)

    def p_correct(self, v: float, m: int | None = None) -> float:
        return float(_sigmoid(self.w * v + self.b))

    def liks(self, v: float, m: int | None = None) -> tuple[float, float]:
        """(f(v|y=1), f(v|y=0)) — 사후 상금 모드용 클래스별 우도."""
        l1 = np.exp(-0.5 * ((v - self.m1) / self.s1) ** 2) / self.s1
        l0 = np.exp(-0.5 * ((v - self.m0) / self.s0) ** 2) / self.s0
        return float(l1), float(max(l0, 1e-12))

    def lik_ratio(self, v: float, m: int | None = None) -> float:
        l1, l0 = self.liks(v)
        return l1 / l0

    # ---- 벡터화 (관측 상금 예약값용 — 그리드 전체를 한 번에) ----

    def p_correct_grid(self, v: np.ndarray, m: int | None = None) -> np.ndarray:
        return _sigmoid(self.w * np.asarray(v, dtype=float) + self.b)

    def liks_grid(self, v: np.ndarray, m: int | None = None):
        v = np.asarray(v, dtype=float)
        l1 = np.exp(-0.5 * ((v - self.m1) / self.s1) ** 2) / self.s1
        l0 = np.exp(-0.5 * ((v - self.m0) / self.s0) ** 2) / self.s0
        return l1, np.maximum(l0, 1e-12)


def fit_platt_per_model(pbar_mat: np.ndarray, y_mat: np.ndarray, steps: int = 800,
                        lr: float = 0.1):
    """모델별 Platt 재보정 (s_m, t_m) — z' = s·z + t, z = logit(p̄).

    보정셋의 예측 p̄ 와 실제 정오로 모델마다 독립 로지스틱을 적합한다. s=1, t=0 에서 출발하므로
    보정이 필요 없으면 그 자리에 머문다 (무해). 표본이 부족하거나 한쪽 클래스만 있으면 항등.
    """
    P = np.clip(np.asarray(pbar_mat, dtype=float), 1e-6, 1 - 1e-6)
    Y = np.asarray(y_mat, dtype=float)
    Z = np.log(P / (1.0 - P))
    M = P.shape[1]
    s = np.ones(M)
    t = np.zeros(M)
    for m in range(M):
        y = Y[:, m]
        if len(y) < 30 or y.sum() < 5 or (len(y) - y.sum()) < 5:
            continue                                   # 항등 유지
        z = Z[:, m]
        sm, tm = 1.0, 0.0
        for _ in range(steps):
            p = _sigmoid(sm * z + tm)
            g = p - y
            sm -= lr * float((g * z).mean())
            tm -= lr * float(g.mean())
        if np.isfinite(sm) and np.isfinite(tm):
            s[m], t[m] = sm, tm
    return s, t


class PerModelNoise:
    """모델별 검증기 보정 (Phase 7 개선, D-corr 대응).

    LLM-judge류 검증기의 편향은 모델마다 다르다(대형 모델의 유창한 오답 과대평가 등).
    전역 단일 보정은 이 이질성을 평균 내버려 상금 추정을 오염시킨다 —
    모델(상자) 열마다 독립 NoiseModel을 적합하고, 표본 부족/퇴화 열은 전역으로 폴백.
    """

    def fit(self, v_mat: np.ndarray, y_mat: np.ndarray):
        self.glob = NoiseModel().fit(v_mat.ravel(), y_mat.ravel())
        self.per = []
        for m in range(v_mat.shape[1]):
            y = y_mat[:, m]
            if 5 <= y.sum() <= len(y) - 5:              # 양쪽 클래스 표본 확보 시만
                self.per.append(NoiseModel().fit(v_mat[:, m], y))
            else:
                self.per.append(self.glob)
        return self

    def _pick(self, m):
        return self.glob if m is None or m >= len(self.per) else self.per[m]

    def p_correct(self, v: float, m: int | None = None) -> float:
        return self._pick(m).p_correct(v)

    def liks(self, v: float, m: int | None = None) -> tuple[float, float]:
        return self._pick(m).liks(v)

    def lik_ratio(self, v: float, m: int | None = None) -> float:
        return self._pick(m).lik_ratio(v)

    def p_correct_grid(self, v: np.ndarray, m: int | None = None) -> np.ndarray:
        return self._pick(m).p_correct_grid(v)

    def liks_grid(self, v: np.ndarray, m: int | None = None):
        return self._pick(m).liks_grid(v)


class ExactNoise:
    """3a 정확성 검증용: 관측 = 진짜 상금 (고전 Pandora 가정)."""

    def p_correct(self, v, m=None):
        return float(v)

    def liks(self, v, m=None):
        return float(v), float(1.0 - v + 1e-12)       # 정확 관측: 사후가 v 자체로 붕괴

    def lik_ratio(self, v, m=None):
        return 1.0

    def p_correct_grid(self, v, m=None):
        return np.asarray(v, dtype=float)             # 관측 상금 == 잠재 상금

    def liks_grid(self, v, m=None):
        v = np.asarray(v, dtype=float)
        return v, np.maximum(1.0 - v, 1e-12)


class GridBelief:
    """IRT 신념: θ 그리드 사후 + 2PL 곡선. update()가 전 상자 p̄를 동시 갱신.

    prize_mode (Phase 8, R2):
      "calibrated": q = P(y=1|v) 전역/모델별 로지스틱 — 검증기가 예리할 때 충분
      "posterior" : q = p̄_m·f(v|1) / (p̄_m·f(v|1) + (1−p̄_m)·f(v|0)) — 사전(θ 사후
                    반영)과 관측을 결합. 검증기가 잡음일 때(실텍스트 AUC~0.9) 선택
                    오류를 줄인다. 이것이 E[Q|v]의 올바른 형태.
    """

    def __init__(self, mu, s, a, b_dom, noise, do_update=True,
                 prize_mode: str = "calibrated", kappa=None,
                 prize_reservation: str = "latent", platt=None, logit_offset=None):
        self.nodes = mu + s * Z_GRID
        self.wts = _PRIOR_W.copy()
        z = a[None, :] * (self.nodes[:, None] - b_dom[None, :])                 # (K,M)
        if logit_offset is not None:
            # Phase 19: 모델별 잔차 r_m(x). 단일 θ 로는 표현 불가한 "모델 X 는 이 종류에
            # 강하다"를 담는다. θ 축이 아니라 로짓에 얹히므로 **모든 θ 노드에 동일 상수**로
            # 더해진다 — 즉 θ 사후 갱신 구조는 그대로 유지된다 (r 은 θ 와 독립).
            z = z + np.asarray(logit_offset, dtype=float)[None, :]
        if platt is not None:
            # Phase 18 (fast tier 레버): 모델별 Platt 재보정. fast tier 는 1회 호출로
            # 퇴화하므로 점수가 argmax(p̄ − λc) 정확도에 전부 걸리고, **모델 간 상대적**
            # 오보정이 그대로 오선택이 된다. ICC 의 기울기·절편을 보정셋으로 다시 맞춘다.
            s_m, t_m = platt
            z = s_m[None, :] * z + t_m[None, :]
        self.P = _sigmoid(z)
        self.noise = noise
        self.do_update = do_update
        self.prize_mode = prize_mode
        self.kappa = kappa            # None=Bernoulli 상금 / (M,)=Beta 상금 (Q4 연속)
        # "latent"     : E[(Q−σ)⁺]=λc  — 잠재 품질로 푼다 (Phase 16까지의 동작)
        # "observable" : E[(X−σ)⁺]=λc, X=E[Q|v] — **상자를 열어 실제로 얻는 상금**으로 푼다
        #                (Phase 17 결함 수정, 근거는 `prize.perbox_reservation` 문서 참조)
        self.prize_reservation = prize_reservation
        # 관측 전 사전확률 캐시 — posterior 상금이 v 를 이중계산하지 않게 한다 (Phase 17).
        self._pre_pbar: dict[int, float] = {}

    def pbar(self) -> np.ndarray:
        return self.wts @ self.P

    def reservation(self, target: np.ndarray) -> np.ndarray:
        """Weitzman 예약값 (B4 일반해). target = λ·c_m.

        Bernoulli는 양 구간 닫힌형(벡터 1회 — 기존 지연 유지), 연속(Beta)은 θ 사후
        혼합에 대한 안전장치 Newton. 어느 쪽이든 σ<0 구간이 정확히 처리된다.

        prize_reservation="observable" 이면 잠재 Q 대신 관측 상금 X=E[Q|v] 분포로 푼다.
        """
        if self.prize_reservation == "observable":
            vals, prob = observable_prize_grid(self.pbar(), self.noise)
            return perbox_reservation(target, vals, prob)
        if self.kappa is None:
            return bernoulli_reservation(self.pbar(), target)
        return beta_reservation(target, self.P, self.kappa, self.wts)

    def update(self, m: int, v: float):
        if not self.do_update:
            return
        # v 를 흡수하기 **전** 사전확률을 남긴다 — posterior 상금이 이 값을 써야 한다.
        self._pre_pbar[m] = float(self.wts @ self.P[:, m])
        r = self.noise.lik_ratio(v, m)
        lik = self.P[:, m] * r + (1.0 - self.P[:, m])
        self.wts = self.wts * lik
        tot = self.wts.sum()
        if tot > 1e-300:
            self.wts = self.wts / tot

    def prize(self, v: float, m: int | None = None) -> float:
        if self.prize_mode == "posterior" and m is not None:
            # ★ Phase 17 수정: 예전에는 `self.wts @ self.P[:,m]` 을 사전확률로 썼는데,
            # route() 는 `update(m,v)` 를 먼저 호출하므로 그 사후에는 **이미 v 가 들어가
            # 있다** → 베이즈 결합에서 v 를 두 번 세었다. 관측 전 사전확률을 쓴다.
            pbar = self._pre_pbar.get(m)
            if pbar is None:
                pbar = float(self.wts @ self.P[:, m])   # 갱신 이력이 없으면 현재값이 사전
            l1, l0 = self.noise.liks(v, m)
            return pbar * l1 / max(pbar * l1 + (1 - pbar) * l0, 1e-12)
        return self.noise.p_correct(v, m)


class GradedGridBelief:
    """등급반응모형(GRM) 신념 — 다수준 이산 품질 (B4 잔여 / Q4 점질량 체제).

    GridBelief가 (K,M) 정답확률을 들고 있는 자리에 (L,K,M) **등급확률**을 들고,
    예약값은 이산 지지 정확해(`discrete_reservation`, DP 최적성 검증 완료)로 푼다.
    Beta 경로와 달리 0·1의 점질량을 자연히 표현하며 특수함수를 쓰지 않는다.

    θ 사후 갱신의 우도 (근사 1건 — 정직 기록):
      검증기 잡음 모델은 이진 (v|y=1), (v|y=0) 두 조건분포로 적합돼 있다. 부분점수
      등급 v_l 에 대한 f(v | Q=v_l) 은 관측되지 않으므로 **선형 보간**을 쓴다:
          f(v | Q=v_l) ≈ v_l·f(v|1) + (1−v_l)·f(v|0)
      "부분적으로 맞은 답의 검증기 점수 분포는 정답과 오답 사이에 있다"는 가정이며,
      L=2 에서는 보간이 항등이 되어 기존 이진 갱신과 **정확히 일치**한다.
    """

    def __init__(self, mu, s, grade_probs, values, noise, do_update=True,
                 prize_reservation: str = "latent"):
        self.nodes = mu + s * Z_GRID
        self.wts = _PRIOR_W.copy()
        self.PR = grade_probs                      # (L, K, M) — P(Q=v_l | θ_k) for model m
        self.values = np.asarray(values, dtype=float)
        self.noise = noise
        self.do_update = do_update
        self.prize_reservation = prize_reservation

    def _mix(self) -> np.ndarray:
        """θ 사후로 가중한 등급확률 (L, M)."""
        return np.einsum("k,lkm->lm", self.wts, self.PR)

    def pbar(self) -> np.ndarray:
        """모델별 기대 품질 E[Q] — 진단·정지 비교용."""
        return self.values @ self._mix()

    def reservation(self, target: np.ndarray) -> np.ndarray:
        from .prize import discrete_reservation
        if self.prize_reservation == "observable":
            vals, prob = observable_prize_grid(self.pbar(), self.noise)
            return perbox_reservation(target, vals, prob)
        return discrete_reservation(target, self.values, self._mix())

    def update(self, m: int, v: float):
        if not self.do_update:
            return
        l1, l0 = self.noise.liks(v, m)
        # f(v | Q=v_l) 선형 보간 → L(v|θ) = Σ_l P(Q=v_l|θ)·f(v|Q=v_l)
        f_grade = self.values * l1 + (1.0 - self.values) * l0      # (L,)
        lik = np.einsum("l,lk->k", f_grade, self.PR[:, :, m])
        self.wts = self.wts * np.maximum(lik, 1e-300)
        tot = self.wts.sum()
        if tot > 1e-300:
            self.wts = self.wts / tot

    def prize(self, v: float, m: int | None = None) -> float:
        """E[Q|v] — 검증기 보정이 연속 라벨로 적합돼 있으므로 그대로 기대 품질이다."""
        return self.noise.p_correct(v, m)


class FixedBelief:
    """판별식 신념: p̄ 고정 — 잠재변수가 없어 관측 후 갱신이 구조적으로 불가능."""

    def __init__(self, pbar_vec, noise):
        self._pbar = np.asarray(pbar_vec, dtype=float)
        self.noise = noise

    def pbar(self):
        return self._pbar

    def reservation(self, target: np.ndarray) -> np.ndarray:
        return bernoulli_reservation(self._pbar, target)

    def update(self, m, v):
        pass

    def prize(self, v, m=None):
        return self.noise.p_correct(v, m)


class PandoraPolicy:
    """하네스 호환 정책. make_belief(x, dom) -> belief 객체.

    stop_margin (M4): 정지 조건을 q* ≥ σ* + margin으로 보수화 — 관측이 잡음일 때
    조기 정지의 품질 손실 위험을 conformal risk control로 통제하는 손잡이.
    """

    def __init__(self, make_belief, lam: float, name: str = "pandora",
                 stop_margin: float = 0.0, open_order: str = "sigma",
                 allow_abstain: bool = False):
        self.make_belief, self.lam, self.name = make_belief, lam, name
        self.stop_margin = stop_margin
        # ★ 전략적 기권 (Phase 19). 기본 off — 채택은 측정으로 판단한다.
        #
        # 왜 필요한가: 예산 스윕에서 극단적 저예산(frac 0.03·0.06)에서 **네 정책 모두 달성폭의
        # 0%** 를 가져갔다(각 0.24·0.23 방치). 오라클은 "어느 쿼리에 돈을 쓸지"를 고르는데,
        # 우리 정책은 아래 `if not obs:` fallback 때문에 **p̄ < λc 인 절망적 쿼리에도 최저가를
        # 무조건 산다.** 그것은 Lagrangian 위반이다 — 그 지출은 더 쉬운 뒷쿼리에 썼어야 한다.
        # 저예산 tier 가 최고 가중인 이 챌린지에서 정확히 반대 방향의 결함이다.
        #
        # 규칙 정합성: 아무것도 호출하지 않으면 최종 답이 없고 하네스는 무응답=0점으로 센다
        # (`bankruptcy="zero_quality"`). 즉 "호출한 출력 중 하나"를 어기지 않는다.
        # ⚠ 주최측 규칙이 무응답을 금지하거나(Q6=force_cheapest) 페널티를 준다면 끄면 된다.
        self.allow_abstain = allow_abstain
        # ★ 발동 카운터 (Phase 19 D32). 이 세션에서 **기능이 조용히 발동하지 않아 "효과 0"으로
        # 측정된 사례가 세 번** 나왔다: ① `open_order` 인자 미전달로 A/B 미실행(D23)
        # ② `agree_ref` 슬롯이 무정보라 동적 재계산이 무변화 ③ 기권 검사를 도달 불가 블록에 배치.
        # 그래서 **null 결과를 받아들이기 전에 기구가 실제로 발동했는지 증명**하는 계측을 둔다.
        # 규칙: 발동 횟수가 0 이면 그 측정은 "효과 없음"이 아니라 "미실행"으로 읽어야 한다.
        self.n_abstain = 0
        self.n_route = 0
        # open_order (Phase 15): 개봉 순서.
        #  "sigma" = Weitzman 예약지수 (무예산 전역 최적, 기본) — 상자 많고 순차 탐색이
        #            가치 있는 체제에 강함.
        #  "value" = 즉시 순가치 argmax(p̄−λc) — 빠듯한 예산·소수 상자·잡음 검증기 체제에서
        #            더 낫다. Weitzman σ의 옵션가치가 무의미해지고(≈1회 개봉) 탐색비용·
        #            선택오류가 이득을 잠식하는 경우(진단: textworld 3모델 +0.034 vs irt
        #            −0.022 tradeoff). knapsack/PILOT(2508.21141) 관점. 정지는 σ 기준 유지.
        self.open_order = open_order

    def route(self, sess, features, domain) -> int:
        self.n_route += 1
        bel = self.make_belief(features, domain)
        M = len(sess.costs)
        unopened = set(range(M))
        obs: dict[int, float] = {}                      # 연 상자의 관측 v (상금은 매회 재계산)
        costs = np.asarray(sess.costs, dtype=float)

        while unopened:
            afford = [m for m in unopened if sess.can_afford(m)]
            if not afford:
                break
            # B4: 예약값은 신념이 상금 분포로부터 유도한다 (이진=닫힌형 / 연속=Beta 혼합).
            # 기존 σ=1−λc/p̄ 는 σ≥0 구간의 특수해였고 λc>p̄ 에서 순서를 틀렸다.
            res = bel.reservation(self.lam * costs)
            max_sigma = max(float(res[m]) for m in afford)   # 잔여 최대 예약값 (정지 기준)
            if self.open_order == "value":                   # 즉시 순가치 개봉 (Phase 15)
                pbar = bel.pbar()
                m_star = max(afford, key=lambda m: float(pbar[m]) - self.lam * costs[m])
            else:                                            # Weitzman σ 개봉 (기본)
                m_star = max(afford, key=lambda m: float(res[m]))
            # 상금 추정은 현재 신념으로 매회 재계산 (posterior 모드에서 θ 갱신 반영)
            if obs and max(bel.prize(v, m) for m, v in obs.items()) \
                    >= max_sigma + self.stop_margin:
                break                                   # Weitzman 정지 규칙 (+ M4 마진)
            if not obs and self.allow_abstain:
                # ★ 첫 개봉 전 순가치 판정 (Phase 19 위치 수정). 초판은 이 검사를 루프 **밖**
                # `if not obs:` 폴백에 뒀는데, 그 블록은 예산이 남아 있는 동안 도달하지 않는다
                # (위 정지 검사가 `obs and` 로 가드돼 obs 가 비면 무조건 한 번 연다). 그래서
                # 기권이 한 번도 발동하지 않았고 측정이 전 구간 0 이었다 — 개념이 아니라
                # **배치가 틀렸다.** Lagrangian 대로 최선 상자의 순가치가 음수면 여기서 멈춘다.
                pbar = bel.pbar()
                if max(float(pbar[m]) - self.lam * float(costs[m]) for m in afford) <= 0.0:
                    self.n_abstain += 1
                    break
            v = sess.call(m_star)
            if v is None:
                break
            unopened.discard(m_star)
            obs[m_star] = v
            bel.update(m_star, v)
            if hasattr(self, "stats"):                  # 선택적 계측 (상자 사용 분포)
                self.stats[m_star] = self.stats.get(m_star, 0) + 1

        if not obs:
            # 강제 최소 응답 시도 (최저가) — 단 allow_abstain 이면 순가치가 음수일 때 기권한다.
            m = int(np.argmin(sess.costs))
            if self.allow_abstain and sess.can_afford(m):
                pbar = bel.pbar()
                if float(pbar[m]) - self.lam * float(sess.costs[m]) <= 0.0:
                    return 0                            # 아무것도 호출하지 않음 = 무응답(0점)
            v = sess.call(m)
            if v is not None:
                obs[m] = v
        if not obs:
            return 0
        return max(obs, key=lambda m: bel.prize(obs[m], m))


def tune_lambda_replay(policy_factory, ds, idx, budget, iters: int = 14,
                       lam_hint: float | None = None, per_query: bool = False,
                       est_costs=None, cost_margin: float = 0.0) -> float:
    """train 리플레이 기반 λ 튜닝: 자연 지출(무제약) ≤ budget이 되는 최소 λ.

    D6 수정: 고정 [0, 1e5] 이분탐색은 반복 수 부족 시 λ를 수십 배 과대 추정
    (예약지수 전면 붕괴 → 즉시 정지). 지수적 브래킷 탐색으로 spend가 budget을
    가로지르는 구간 [hi/4, hi]를 먼저 찾은 뒤 그 안에서 이분탐색한다.

    lam_hint (B4): 브래킷 탐색의 시작점. λ가 작을수록 정책이 상자를 많이 열어
    리플레이가 비싸지므로, 0.5에서 올라오는 기본 경로는 **가장 비싼 구간을 먼저**
    지난다. 힌트(예: 평균정합 Bernoulli 튜닝값)에서 출발하면 그 낭비만 사라지고
    **수렴하는 해는 동일하다** — 근사가 아니라 탐색 순서 최적화다.
    """
    from src.harness import run_tier
    # per_query=True (B5): budget은 **문항당** 상한이므로 총지출 목표는 budget×|idx| 다.
    # 이 환산 없이 총지출을 문항당 예산과 직접 비교하면 λ가 폭증해 정책이 최저가 1회로
    # 퇴화한다 (개발 중 실측: fast 품질 0.92 → 0.16).
    free = budget * 1e6                                # 무제약 리플레이용 (의미론 동일)
    budget = budget * (len(idx) if per_query else 1)   # 이후 비교는 총지출 기준

    def spend(lam):
        return run_tier(ds, idx, policy_factory(lam), "tune", free,
                        per_query=per_query, est_costs=est_costs,
                        cost_margin=cost_margin).total_cost

    start = 0.5 if lam_hint is None else max(float(lam_hint), 1e-6)
    hi = start
    if spend(hi) > budget:
        while spend(hi) > budget and hi < 1e7:        # 지출 ≤ 예산이 될 때까지 λ 증가
            hi *= 4.0
        lo = 0.0 if hi == start else hi / 4.0
    else:                                             # 힌트가 이미 충분히 큼 → 아래로
        lo = hi / 4.0
        while lo > 1e-6 and spend(lo) <= budget:      # 예산을 넘기는 lo를 찾을 때까지
            hi, lo = lo, lo / 4.0
        lo = 0.0 if lo <= 1e-6 else lo
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if spend(mid) > budget:
            lo = mid
        else:
            hi = mid
    return hi


#: 기존 λ 후보 배수 — 로그 간격이 2배로 거칠다 (fast tier 에서 문제가 된다, 아래 참조)
COARSE_MULTS = (1, 2, 4, 8, 16)
#: Phase 18 신설. fast tier 는 종합 점수 가중치의 절반을 차지하는데, 그 구간에서 정책은
#: σ<0 체제로 밀려 **1회 호출**로 퇴화한다. 그러면 점수는 "그 1회를 누구로 고르는가"
#: = argmax(p̄ − λc) 뿐이고, 이는 **λ 값에 직접 의존**한다. 2배 간격 격자는 그 선택 경계를
#: 성기게 훑어 최적 λ 를 지나칠 수 있다. 1 부터 아래로도 내려가고(0.5·0.7) 간격을 √2 로 줄인다.
FINE_MULTS = (0.5, 0.71, 1, 1.41, 2, 2.83, 4, 5.66, 8, 11.3, 16)


def tune_lambda_quality(policy_factory, ds, idx, budget, mults=COARSE_MULTS,
                        lam_hint: float | None = None,
                        per_query: bool = False,
                        est_costs=None, cost_margin: float = 0.0,
                        se_k: float = 0.0) -> float:
    """D11 수정: '예산을 다 쓰는 최소 λ'가 아니라 '예산 내 train 품질 최대 λ'를 선택.

    품질이 예산보다 싸게 포화되는 체제(4c World B)에서 최소-λ는 이득 없는 지출을
    강제한다. 예산-실현가능 최소 λ에서 출발해 배수 후보를 예산 제약 하에 리플레이하고
    품질 최대(동률 시 큰 λ = 절약·저지연)를 채택한다.

    se_k > 0 (Phase 18): 후보 간 차이가 **쌍대 표준오차**의 se_k 배를 넘지 않으면 동률로 보고
    큰 λ 를 택한다. 거친 격자를 촘촘하게 만들면 후보가 늘어나 잡음 최댓값을 고르는 위험
    (multiple comparisons)이 커지므로, 격자를 좁히는 것과 문턱을 세우는 것은 함께 가야 한다.
    D21 의 교훈을 λ 탐색 자체에 적용한 것이다.

    ★ D34 (래칫 결함 수정): 초판의 se_k 분기는 후보를 **직전에 채택된 후보**와 비교하고,
    채택될 때마다 기준선(best_q/best_pq)을 그 후보로 옮겼다. 그래서 "유의하지 않은 손실"이
    **누적**됐다 — 오름차순 격자에서는 매 단계 최대 1×SE 씩 잃으면서 끝까지 올라가,
    참 최적이 격자 최솟값이어도 최댓값을 고른다(시뮬레이션: 단계당 0.004 손실이면 종합
    0.04 를 포기하고 lam_mult=16 을 선택). λ 과대 → σ<0 증가 → 1회 호출 퇴화이므로
    fast tier 퇴화의 일부가 이 결함이었다.

    수정: 기준은 **항상 전역 최고 후보**다. 전 후보를 먼저 평가하고, 최고 대비 유의하게
    나쁘지 않은 후보 중 **가장 큰 λ** 를 고른다. se_k=0 이면 정확히 동률만 허용하므로
    기존 COARSE 경로(argmax, 동률→큰 λ)와 동일하다.
    """
    from src.harness import run_tier
    lam_min = tune_lambda_replay(policy_factory, ds, idx, budget, lam_hint=lam_hint,
                                 per_query=per_query, est_costs=est_costs,
                                 cost_margin=cost_margin)
    cands = []                                       # (lam, quality, per_query)
    for m in mults:
        lam = lam_min * m
        r = run_tier(ds, idx, policy_factory(lam), "tune", budget,
                     per_query=per_query, est_costs=est_costs, cost_margin=cost_margin)
        pq = None if r.per_query is None else np.asarray(r.per_query, dtype=float)
        cands.append((float(lam), float(r.mean_quality), pq))
    return select_lambda(cands, se_k) if cands else lam_min


def select_lambda(cands, se_k: float = 0.0) -> float:
    """λ 후보 중 채택값 선택 — `tune_lambda_quality` 의 순수 함수 부분 (D34).

    cands: [(lam, quality, per_query 배열 또는 None), ...]

    규칙: **전역 최고 후보를 기준선으로 고정**하고, 그 기준 대비 유의하게 나쁘지 않은
    후보 중 가장 큰 λ 를 고른다 (큰 λ = 절약·저지연·분포이동 방어).

    ★ 기준선이 고정이라는 점이 D34 수정의 핵심이다. 초판은 기준선을 직전 채택자로 옮겨
    "유의하지 않은 손실"을 무한 누적시켰고, 오름차순 격자에서 항상 최대 λ 로 밀려 올라갔다.
    리플레이가 필요 없는 순수 함수라 `tests/test_deploy_fixes.py` 가 직접 고정한다.
    """
    cands = sorted(cands, key=lambda c: c[0])        # λ 오름차순 (격자 순서 비의존)
    lam_star, q_star, pq_star = max(cands, key=lambda c: c[1])   # 기준선 — 이후 불변
    if se_k <= 0.0 or pq_star is None:
        # 기존 동작: 정확히 동률인 후보 중 가장 큰 λ
        return max((lam for lam, q, _ in cands if q >= q_star), default=lam_star)
    best_lam = lam_star
    for lam, q, pq in cands:
        if pq is None or lam <= best_lam:
            continue
        d = np.asarray(pq, dtype=float) - np.asarray(pq_star, dtype=float)
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
        # `>=` 여야 한다: 완전 동률(se=0, q==q_star)에서 엄격 부등호는 동률을 **거부**해
        # "동률이면 큰 λ" 라는 원래 규칙을 깨뜨린다 (신규 테스트가 잡은 결함).
        if (q - q_star) >= -se_k * se:               # **최고 대비** 유의하게 나쁘지 않다
            best_lam = lam
    return best_lam
