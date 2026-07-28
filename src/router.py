"""LPB-Router 최종 조립 (Phase 6) — M1+M2+M3+M4의 단일 진입점.

구성 (Phase 1~5 실증을 통과한 최종형):
  M1  Amortized IRT 신념망 (다집단 2PL MML + 인코더)      [Phase 2]
  M2  Weitzman 예약지수 엔진 (DP 대조로 최적성 검증)        [Phase 3]
  M3  쌍대가격 페이싱 v2 (누적·비대칭·생존 모드)            [Phase 4]
  +   quality-first λ 선택, [모델×N표] 투표 상자            [Phase 4]
  M4  conformal 인증서 (정책 무개입 위험 상한)              [Phase 5]

기각된 구성(무비용 플래그로만 잔존): 베이지안 재지수(D7), 정지 마진 CRC(D12),
보수화 관측 채널(D13) — 근거는 docs/reflections/phase{3,5}.md.

실데이터 적용 시 교체 지점: (1) Dataset 로더, (2) IRTEncoder의 백본을 텍스트
인코더(mDeBERTa/bge 계열)로, (3) cost_mirror를 주최측 산식으로 (Phase 0 Q3).
"""
import numpy as np
from scipy.stats import beta as beta_dist

from .irt.mml import fit_mml
from .encoder import IRTEncoder
from .harness import tier_budgets, Session, run_tier
from .cost_mirror import cost_matrix
from .engine.pandora import (NoiseModel, PerModelNoise, GridBelief, PandoraPolicy,
                             tune_lambda_quality, COARSE_MULTS, FINE_MULTS)
from .engine.pacing import PacedPandora


class LPBRouter:
    """tier별로 fit() 후 policy()가 하네스 호환 정책을 반환한다."""

    def __init__(self, cfg, n_domains: int, cal_frac: float = 0.3,
                 cert_conf: float = 0.95, seed: int = 0,
                 per_model_verifier: bool = True, tune_headroom: float = 0.97,
                 prize_mode: str = "calibrated", prize_family: str = "bernoulli",
                 pacing: bool = True, per_query_budget: bool = False,
                 use_domain: bool = True, open_order: str = "sigma",
                 prize_reservation: str = "observable", cert_frac: float = 0.5,
                 cost_mode: str = "oracle", cost_margin: float = 0.0,
                 n_clusters: int = 8, calibrate_pbar: bool = True,
                 enc_hidden: int = 0, enc_residual: bool = False,
                 allow_abstain: bool = False,
                 lam_mults=None, lam_se_k: float = 1.0,
                 text_encoder=None):
        # ★ D36: 배포 어댑터가 **텍스트 프롬프트**를 특성으로 바꿀 때 쓸 인코더.
        # 명시하지 않으면 `fit()` 이 `ds.text_encoder` 에서 물려받는다 — 즉 특성을 만든
        # 그 인코더가 그대로 배포에 실린다. 초판은 이 경로가 없어 `SubmissionRouter._features`
        # 가 텍스트 입력에서 항상 ValueError 였다 (72개 테스트가 전부 ndarray 만 넘겨 못 잡았다).
        self.text_encoder = text_encoder
        # Phase 18 — fast tier 전용 레버 2종 (가중치 0.5 구간이 1회 호출로 퇴화하므로,
        # 그 1회의 선택 정확도가 곧 점수다).
        #   calibrate_pbar : 모델별 Platt 재보정 (모델 간 상대 오보정 = 직접적 오선택)
        #   lam_mults      : λ 후보 격자. None → FINE_MULTS(√2 간격 + 1 미만 포함).
        #                    기존 2배 간격은 1회 선택의 λ 경계를 성기게 훑었다
        #   lam_se_k       : λ 후보 비교의 쌍대 표준오차 문턱 (격자를 좁히면 후보가 늘어
        #                    잡음 최댓값을 고를 위험이 커지므로 함께 필요)
        # ★ 기본값 on 근거 = 측정 (`eval/results/probe_fast_tier.json`, 3-fold, A.X 3모델·
        #   Q2=No·textworld·증강 검증기): fast 0.3555 → 0.3638 (+0.0083 = **2.1×쌍대 SE**),
        #   종합 0.5017 → 0.5051. 두 레버 각각도 1×SE 를 넘었다. D21 기준(표준오차 문턱)을
        #   통과했으므로 기본값을 바꾼다 — 통과하지 못했다면 유지했을 것이다.
        #   부수 관찰: 세밀 격자가 σ<0 비율을 0.70 → 0.58 로 낮춘다 (일부 상자를 승급 가능
        #   구간에 남긴다) — fast tier 퇴화를 완화하는 방향이 실제로 작동했다.
        self.calibrate_pbar = calibrate_pbar
        self.lam_mults = lam_mults
        self.lam_se_k = lam_se_k
        self.platt = None
        # use_domain="cluster" 일 때 pseudo-domain 개수 (무감독 프롬프트 군집, Phase 17).
        self.n_clusters = n_clusters
        # held-out 중 **인증 전용**으로 떼어 둘 비율 (λ₀ 선택에 관여하지 않는 split).
        self.cert_frac = cert_frac
        # Phase 19 — 예측기(p̄) 용량. 오라클 치환 진단 결과 남은 격차의 **82.8%** 가 예측 몫이고
        # 결정 구조는 3.6% 였다(`eval/results/probe_predictor_ceiling.json`). 그래서 여기가
        # 남은 최대 레버다. 기본 off — 채택은 `eval/probe_predictor.py` 측정으로 판단한다.
        #   enc_hidden   : μ·σ 헤드 앞 tanh 은닉층 폭 (0=선형, 기존과 동일)
        #   enc_residual : 모델별 로짓 잔차 r_m(x). 단일 잠재 θ 로는 "모델 X 는 이 종류에
        #                  강하다"를 표현할 수 없는 구조적 한계를 라벨 없이 푼다
        self.enc_hidden, self.enc_residual = int(enc_hidden), bool(enc_residual)
        # 전략적 기권 (Phase 19) — 근거·규칙 정합성은 `engine/pandora.py` PandoraPolicy 참조.
        self.allow_abstain = bool(allow_abstain)
        # Phase 17 (레드팀 MAJOR): 호출 전 비용은 실제로는 **추정치**다 (decode 길이 미지).
        #   cost_mode "oracle" — 실제 비용 (Phase 16 까지의 암묵 가정, ablation 상한)
        #             "ridge"  — 프롬프트 특성 → 길이 회귀 (배포에 정직한 기본 후보)
        #             "mean"   — 모델별 평균 길이 (특성 없이도 가능한 최소 기준)
        #   cost_margin — 계획 단계에서 비용을 (1+margin) 배로 보수화. 과소추정으로 호스트가
        #             호출을 거부하는 사건(`TierResult.rejected`)을 줄인다.
        # 기본은 "oracle" 로 두어 기존 점수표가 불변이고, 실배포·측정은 명시적으로 켠다
        # (`eval/probe_cost_uncertainty.py`). 실데이터에서 무엇을 쓸지는 그 측정이 결정한다.
        self.cost_mode, self.cost_margin = cost_mode, cost_margin
        self._est_cmat = None
        self.cfg, self.n_dom = cfg, n_domains
        # Phase 17: 예약값을 무엇의 분포로 풀 것인가.
        #   "observable" (기본) — X=E[Q|v], 상자를 열어 **실제로 얻는** 상금. 정지 규칙과
        #       최종 선택이 쓰는 양과 일치하므로 이것이 Weitzman 방정식의 올바른 입력이다.
        #   "latent"          — 잠재 품질 Q (Phase 16까지의 동작). 재현·ablation 전용.
        # 근거·반례: `src/engine/prize.py:perbox_reservation`, 측정: `eval/probe_observable_prize.py`
        self.prize_reservation = prize_reservation
        # 개봉 순서 (Phase 15): "sigma"=Weitzman 예약지수(기본) · "value"=즉시 순가치 argmax(p̄−λc)
        #  · "auto"=train-cal replay로 두 순서 중 나은 것을 fold별 선택(엿보기 없이).
        # 진단: value는 빠듯한 예산·소수 상자·잡음 검증기(textworld 3모델 +0.034)에서 낫고,
        # sigma는 순차 탐색이 가치 있는 체제(irt −0.022)에서 낫다 → 체제 의존이라 auto가 정답.
        self.open_order = open_order
        self.open_order_eff = "sigma" if open_order == "auto" else open_order
        # 챌린지 런타임은 task/도메인 라벨을 제공하지 않는다 (대회 상세: "benchmark 또는
        # task 이름 미제공"). use_domain=False 는 다집단 2PL(b_{d,m})을 단일집단(b_m)으로
        # 퇴화시켜 추론 시 도메인이 없어도 동작하게 한다 — 리스크 레지스터 R2/R5식
        # 우아한 퇴화. irt["b"]가 (1,M)이 되어 submission 어댑터의 도메인 필수 예외(D18)도
        # 자동 해소된다 (n_domains==1 → domain_of 기본값). 적합은 여전히 공개 데이터로
        # 하되 도메인 축을 풀링한다. 비용은 eval/probe_domain_free.py 로 측정.
        self.use_domain = use_domain
        # B5 / Phase 0 Q5: 예산이 문항당 상한이면 쿼리 간 배분 문제가 없으므로 페이싱
        # (shadow price의 온라인 조절)은 무의미해진다 → λ 고정 모드로 우아하게 퇴화.
        # 리스크 레지스터 R5의 설계대로이며, 정책 형태는 그대로다 (λ만 상수).
        self.per_query_budget = per_query_budget
        self.pacing = pacing and not per_query_budget
        self.cal_frac, self.cert_conf, self.seed = cal_frac, cert_conf, seed
        self.per_model_verifier = per_model_verifier   # Phase 7: 모델별 검증기 보정
        self.prize_mode = prize_mode                   # Phase 8 R2: 사후 상금 모드
        # B4 / Phase 0 Q4: "bernoulli"=이진 품질(닫힌형) · "beta"=연속 품질(Beta 반응모형).
        # 로더의 validate_dataset()이 quality_kind로 어느 쪽인지 알려준다.
        self.prize_family = prize_family
        # M4 인증서의 '유의미한 손해' 기준. 이진은 0(기존 정의와 등가), 연속은 부분점수
        # 차이가 늘 존재하므로 재료성 문턱을 세워야 인증서가 정보를 갖는다.
        self.cert_margin = 0.0 if prize_family == "bernoulli" else 0.1
        # Phase 7 R1: λ0 튜닝 목표에 3% 안전 여유 — 보정셋과 테스트의 비용 구성
        # 이질성(fold 간 in_tokens 분포 차)이 유발하는 잔여 파산을 제거.
        # 품질-지출 곡선이 상단에서 평탄하므로 품질 비용 ≈ 0 (corr/balanced 실측 근거)
        self.tune_headroom = tune_headroom

    def fit(self, ds, train_idx: np.ndarray, tier: str):
        # D36: 특성을 만든 인코더를 데이터셋에서 물려받는다 (명시 인자가 우선).
        if self.text_encoder is None:
            self.text_encoder = getattr(ds, "text_encoder", None)
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(train_idx)
        n_fit = int((1 - self.cal_frac) * len(perm))
        tr_fit, held = perm[:n_fit], perm[n_fit:]
        # ★ Phase 17 (레드팀 MAJOR): 예전에는 held-out 전체가 **λ₀ 선택과 인증서 산출을
        # 동시에** 담당했다. λ₀(그리고 open_order)가 그 집합에서 골라진 뒤 같은 집합의
        # 후회율로 Clopper–Pearson 상한을 매기면 in-sample 이라 보증이 아니다.
        # → held-out 을 튜닝용/인증용으로 쪼갠다. 인증 split 은 정책 선택에 일절 관여하지 않는다.
        n_cert = max(1, int(round(self.cert_frac * len(held))))
        tr_cal, tr_cert = held[:len(held) - n_cert], held[len(held) - n_cert:]
        if len(tr_cal) == 0:                              # 극소 표본 방어
            tr_cal, tr_cert = held, held

        # ---- 집단 축 결정 (Phase 17: "cluster" = 무감독 pseudo-domain) ----
        # 런타임에 task 라벨이 없으므로 다집단 이득을 버려야 했는데(단일집단 퇴화),
        # 프롬프트 특성은 항상 있으니 거기서 집단을 만든다. 근거·측정: src/cluster.py.
        if self.use_domain == "cluster":
            from .cluster import PromptClusterer
            self.clusterer = PromptClusterer(k=self.n_clusters, seed=self.seed).fit(
                ds.features[tr_fit])
            self._doms = self.clusterer.predict(ds.features)
            n_dom_eff, pdb = self.clusterer.n_clusters, True
        else:
            self.clusterer = None
            self._doms = np.asarray(ds.domains)
            n_dom_eff, pdb = self.n_dom, bool(self.use_domain)

        if self.prize_family == "grm":
            from .irt.grm_mml import fit_grm_mml
            self.irt = fit_grm_mml(ds.quality[tr_fit], self._doms[tr_fit], n_dom_eff,
                                   per_domain_b=pdb)
            self.kappa = None
            # IRTEncoder는 (a, b) 2PL 형태로 θ 헤드를 학습한다. GRM에는 단일 b가 없으므로
            # **중앙 문턱**을 유효 난이도 대리값으로 준다 — 인코더는 θ 척도만 맞추면 되고
            # 실제 상금 계산은 전 문턱을 쓰는 GradedGridBelief가 담당한다.
            self.irt["b"] = self.irt["thr"][:, :, self.irt["thr"].shape[2] // 2]
        elif self.prize_family == "beta":
            from .irt.beta_mml import fit_beta_mml
            self.irt = fit_beta_mml(ds.quality[tr_fit], self._doms[tr_fit], n_dom_eff,
                                    per_domain_b=pdb)
            self.kappa = self.irt["kappa"]
        else:
            self.irt = fit_mml(ds.quality[tr_fit], self._doms[tr_fit], n_dom_eff,
                               per_domain_b=pdb)
            self.kappa = None
        self.enc = IRTEncoder(self.irt["a"], self.irt["b"], hidden=self.enc_hidden,
                              residual=self.enc_residual).fit(
            ds.features[tr_fit], self._doms[tr_fit], ds.quality[tr_fit], seed=self.seed)
        if self.calibrate_pbar:                # 보정셋으로 모델별 ICC 재보정 (Phase 18)
            from .engine.pandora import fit_platt_per_model
            self.platt = fit_platt_per_model(
                self.enc.predict(ds.features[tr_cal], self._doms[tr_cal]),
                ds.quality[tr_cal])
        if self.per_model_verifier:
            self.noise = PerModelNoise().fit(ds.verifier[tr_fit], ds.quality[tr_fit])
        else:
            self.noise = NoiseModel().fit(ds.verifier[tr_fit].ravel(),
                                          ds.quality[tr_fit].ravel())

        if self.cost_mode != "oracle":          # 사전 비용 추정기는 train fold 로만 적합
            from .cost_model import DecodeLengthEstimator, estimated_cost_matrix
            self.cost_est = DecodeLengthEstimator(mode=self.cost_mode).fit(
                ds.features[tr_fit], ds.out_tokens[tr_fit])
            self._est_cmat = estimated_cost_matrix(ds, self.cost_est)
        b_cal = tier_budgets(ds, tr_cal, self.cfg, self.per_query_budget)[tier]
        # λ₀ 튜너는 보정셋을 20여 회 리플레이하고, λ가 작을수록 정책이 상자를 많이
        # 열어 리플레이가 비싸진다. 연속(Beta) 경로에서는 호출마다 특수함수 평가가
        # 들어가 이 구간의 비용이 폭발한다(측정: 튜닝 1,800s vs 모형 적합 45s).
        # → 값싼 평균정합 Bernoulli 튜닝으로 **브래킷 시작점만** 잡고, 실제 탐색·해는
        #   정확한 Beta 정책으로 구한다. 근사가 아니라 탐색 순서 최적화이므로
        #   수렴하는 λ₀는 완전탐색과 동일하다 (eval/probe_tune_family.py가 대조 검증).
        b_tune = b_cal * self.tune_headroom
        pq = self.per_query_budget
        hint = None
        if self.prize_family == "beta":            # GRM·Bernoulli는 예약값이 이미 값싸다
            hint = tune_lambda_quality(self._factory_tune, ds, tr_cal, b_tune, per_query=pq)
        if self.open_order == "auto":
            # 두 개봉 순서를 각자의 λ₀로 튜닝해 **train-cal 품질로 선택** (테스트 미접촉).
            # 진단이 보인 regime 의존(value: 빠듯·소수상자 / sigma: 순차탐색 가치)을 fold별
            # 데이터로 해소한다 — 어느 쪽이 이 세계에서 나은지 미리 알 수 없으므로.
            cands = []
            for order in ("sigma", "value"):
                self.open_order_eff = order
                lam_o = tune_lambda_quality(self._factory, ds, tr_cal, b_tune,
                                            lam_hint=hint, per_query=pq,
                                            est_costs=self._est_cmat,
                                            cost_margin=self.cost_margin,
                                            mults=self.lam_mults or FINE_MULTS,
                                            se_k=self.lam_se_k)
                q = run_tier(ds, tr_cal, self._factory(lam_o), tier, b_cal,
                             per_query=pq, est_costs=self._est_cmat,
                             cost_margin=self.cost_margin).mean_quality
                cands.append((q, order, lam_o))
            best = max(cands, key=lambda c: c[0])
            self.open_order_eff, self.lam0 = best[1], best[2]
            self._order_cands = cands                           # 진단용
        else:
            self.open_order_eff = self.open_order
            self.lam0 = tune_lambda_quality(self._factory, ds, tr_cal, b_tune,
                                            lam_hint=hint, per_query=pq,
                                            est_costs=self._est_cmat,
                                            cost_margin=self.cost_margin,
                                            mults=self.lam_mults or FINE_MULTS,
                                            se_k=self.lam_se_k)
        self._tr_cal, self._b_tune, self._hint = tr_cal, b_tune, hint   # 진단용
        self._tr_cert = tr_cert
        b_cert = tier_budgets(ds, tr_cert, self.cfg, self.per_query_budget)[tier]
        self.certificate = self._certify(ds, tr_cert, b_cert)
        self.diagnostics = self._diagnose(ds, tr_cert, b_cert)
        return self

    def _make_factory(self, kappa):
        if self.prize_family == "grm":
            from .irt.grm_mml import grm_probs
            from .engine.pandora import GradedGridBelief, Z_GRID

            def factory(lam: float) -> PandoraPolicy:
                def make_belief(x, dom):
                    mu, s = self.enc.belief_row(x)
                    if self.clusterer is not None:
                        dom = self.clusterer.predict_row(x)
                    pr = grm_probs(self.irt, mu + s * Z_GRID, dom)     # (L,K,M)
                    return GradedGridBelief(mu, s, pr, self.irt["values"], self.noise,
                                            prize_reservation=self.prize_reservation)
                return PandoraPolicy(make_belief, lam, name="lpb",
                                     open_order=self.open_order_eff)
            return factory

        def factory(lam: float) -> PandoraPolicy:
            def make_belief(x, dom):
                mu, s = self.enc.belief_row(x)
                # 단일집단(use_domain=False)이면 b가 (1,M) — 도메인 인덱스를 0으로 클램프.
                # cluster 모드는 **프롬프트에서** 집단을 정한다 → 런타임 라벨 불필요.
                if self.clusterer is not None:
                    d = self.clusterer.predict_row(x)
                else:
                    d = dom if self.irt["b"].shape[0] > 1 else 0
                return GridBelief(mu, s, self.irt["a"], self.irt["b"][d], self.noise,
                                  prize_mode=self.prize_mode, kappa=kappa,
                                  prize_reservation=self.prize_reservation,
                                  platt=self.platt,
                                  logit_offset=self.enc.residual_row(x))
            # ★ Phase 17 결함: 여기서 open_order 를 넘기지 않아 bernoulli/beta(=배포 경로)
            # 에서는 `open_order="value"|"auto"` 가 **조용히 무시**됐다. GRM 분기만 넘기고
            # 있었다. 그래서 Phase 15 의 "sigma≡value 완전 동일(0.6129 vs 0.6129)" 과
            # `probe_ax3_collapse.json` 의 `auto_minus_sigma=0.0` 은 두 팔이 모두 sigma 로
            # 돈 결과였다 — A/B 가 실행되지 않았다. 인자를 연결해 실제로 비교되게 한다.
            return PandoraPolicy(make_belief, lam, name="lpb",
                                 open_order=self.open_order_eff,
                                 allow_abstain=self.allow_abstain)
        return factory

    def _factory(self, lam: float) -> PandoraPolicy:
        """배포 정책 — 상금족에 맞는 정확한 예약값."""
        return self._make_factory(self.kappa)(lam)

    def _factory_tune(self, lam: float) -> PandoraPolicy:
        """λ₀ 탐색 전용 — 평균정합 Bernoulli 예약값 (kappa=None)."""
        return self._make_factory(None)(lam)

    def policy(self):
        """배포 정책. 전역 예산이면 페이싱, 문항당 예산이면 λ 고정 (B5).

        λ 고정 모드에서도 정책의 **형태는 동일**하다 — 예약지수 σ = 1 − λc/p̄ 로 열고
        멈추는 구조 그대로이며, λ가 온라인으로 움직이지 않을 뿐이다. 문항당 예산에서는
        쿼리 간 이월이 없어 조절할 대상 자체가 없기 때문이다 (R5 설계).
        """
        if not self.pacing:
            return self._factory(self.lam0)
        return PacedPandora(self._factory, self.lam0, name="lpb-router")

    def _cost_views(self, ds):
        """(정책이 보는 비용, 실제 과금 비용) — cost_mode="oracle" 이면 동일 행렬."""
        true = cost_matrix(ds)
        return (true if self._est_cmat is None else self._est_cmat), true

    def _diagnose(self, ds, idx, budget) -> dict:
        """배포 체제 진단 (Phase 17, 레드팀 CRITICAL #2 대응 — 방어가 아니라 계측).

        보고하는 것
          sigma_neg_frac  : 감당 가능한 미개봉 상자 중 **예약값 σ<0** 인 비율.
                            1.0 이면 정지 규칙이 첫 관측 직후 무조건 성립하므로 정책은
                            "1회 호출 = argmax(p̄−λc)" 로 퇴화한다 — 즉 그 tier 에서는
                            Weitzman 순차 탐색이 비활성이고 학습형 단일호출과 같은
                            정책 클래스가 된다. fast tier(가중치 최대)에서 실제로 발생한다.
          calls_per_query : 실제 호출 수. 위 진단의 독립 확인.
          single_call_frac: 정확히 1회만 호출한 쿼리 비율.
        이 값들을 숨기지 않고 제출 문서에 싣는 것이 이 결함에 대한 정직한 대응이다.
        """
        pol_cmat, cmat = self._cost_views(ds)
        lam = float(self.lam0)
        neg, tot, calls, singles = 0, 0, 0, 0
        for i in idx:
            bel = self._factory(lam).make_belief(ds.features[i], int(ds.domains[i]))
            res = np.asarray(bel.reservation(lam * pol_cmat[i]), dtype=float)
            afford = pol_cmat[i] <= budget + 1e-12
            if afford.any():
                neg += int((res[afford] < 0).sum())
                tot += int(afford.sum())
            sess = Session(costs=pol_cmat[i], verifier_row=ds.verifier[i],
                           remaining_budget=budget,
                           charge_costs=None if self._est_cmat is None else cmat[i],
                           cost_margin=self.cost_margin)
            self._factory(lam).route(sess, ds.features[i], int(ds.domains[i]))
            calls += len(sess.called)
            singles += int(len(sess.called) == 1)
        n = max(len(idx), 1)
        return {"sigma_neg_frac": round(neg / max(tot, 1), 4),
                "calls_per_query": round(calls / n, 3),
                "single_call_frac": round(singles / n, 4),
                "lam0": lam, "n_diag": int(n)}

    def _certify(self, ds, cal_idx, budget) -> dict:
        """M4: **인증 전용 split** 의 조기정지 후회율 Clopper–Pearson 상한 (정책 무개입).

        유효성 범위 (Phase 17 정정 — 과대주장 제거):
          · 이 split 은 λ₀·open_order 선택에 관여하지 않는다 (fit 의 3분할). 그래서
            "튜닝셋에서 잰 in-sample 수치"라는 결함은 해소됐다.
          · 그러나 **전역 예산 체제에서 후회 사건은 독립이 아니다** — 쿼리 i 의 결정이
            1..i−1 의 지출을 통해 λ 에 의존한다(M3 페이싱). Clopper–Pearson 은 i.i.d.
            베르누이를 가정하므로, 엄밀히는 "교환가능성 가정 하의 상한"이다. 문항당 예산
            (`per_query_budget=True`)에서는 의존성이 사라져 가정이 정확히 성립한다.
          · 따라서 표기는 "분포무관 유한표본 보증"이 아니라 **"독립 인증 split 에서 측정한
            후회율과 그 이항 상한"** 이다. `dependent` 필드가 이를 기계가독 형태로 남긴다.
        """
        pol_cmat, cmat = self._cost_views(ds)
        pol = self.policy()
        if hasattr(pol, "reset"):                      # λ 고정 모드에는 페이싱 상태가 없다
            pol.reset(tier="cert", budget=budget * (len(cal_idx) if self.per_query_budget
                                                    else 1), n_queries=len(cal_idx))
        events, spent_total = 0, 0.0
        for i in cal_idx:
            remaining = budget if self.per_query_budget else budget - spent_total
            sess = Session(costs=pol_cmat[i], verifier_row=ds.verifier[i],
                           remaining_budget=remaining,
                           charge_costs=None if self._est_cmat is None else cmat[i],
                           cost_margin=self.cost_margin)
            choice = pol.route(sess, ds.features[i], int(ds.domains[i]))
            if sess.called:
                if choice not in sess.called:
                    choice = max(sess.called, key=lambda m: sess.verifier_row[m])
                # 후회 사건 = "예산 안에서 열 수 있었던 더 나은 상자를 두고 멈췄다".
                # B4 일반화: 이진(τ=0)에서는 기존 정의(0을 골랐는데 1이 있었음)와
                # 정확히 등가이고, 연속 품질에서는 τ가 '유의미한 손해'의 기준이 된다.
                left = remaining - sess.spent
                alts = [ds.quality[i, m] for m in range(ds.m)
                        if m not in sess.called and cmat[i, m] <= left]
                if alts and max(alts) > ds.quality[i, choice] + self.cert_margin:
                    events += 1
            spent_total += sess.spent
            if hasattr(pol, "observe_spend"):
                pol.observe_spend(sess.spent)
        n = len(cal_idx)
        upper = float(beta_dist.ppf(self.cert_conf, events + 1, n - events)) \
            if events < n else 1.0
        return {"risk_hat": events / n, "risk_upper": upper,
                "confidence": self.cert_conf, "n_cal": n,
                # 정직 메타데이터 (Phase 17)
                "split": "held-out cert split (λ₀ 선택과 분리)",
                "dependent": not self.per_query_budget,
                "note": ("전역 예산 페이싱으로 사건 간 의존성 존재 → 교환가능성 가정 하 상한"
                         if not self.per_query_budget else
                         "문항당 예산이라 사건 독립 → 이항 상한 가정 정확")}
