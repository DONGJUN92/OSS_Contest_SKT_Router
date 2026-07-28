"""Phase 17 결함 수정의 회귀 방벽.

레드팀이 찾은 결함은 전부 "조용히 다른 정책이 되는" 종류였다 — 크래시가 아니라 잘못된 값을
내면서 테스트를 통과했다. 그래서 각 수정에 **그 결함이 재발하면 반드시 실패하는** 테스트를
붙인다. D22 교훈: 회귀 테스트는 옳음의 증거가 아니라 현재 동작의 증거이므로, 무엇을
고정하는지 명시해야 한다.
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))


# ---------- 1. 관측 상금 예약값 (잠재 Q 로 풀던 결함) ----------

class _Chan:
    """가우시안 검증기 채널 — 상자별 예리함 지정."""

    def __init__(self, sds):
        self.sds = list(sds)

    def _sd(self, m):
        return self.sds[0] if m is None or m >= len(self.sds) else self.sds[m]

    def liks_grid(self, v, m=None):
        s = self._sd(m)
        v = np.asarray(v, dtype=float)
        return (np.exp(-0.5 * ((v - 1.0) / s) ** 2) / s,
                np.maximum(np.exp(-0.5 * ((v - 0.0) / s) ** 2) / s, 1e-12))

    def p_correct_grid(self, v, m=None):
        l1, l0 = self.liks_grid(v, m)
        return l1 / np.maximum(l1 + l0, 1e-12)


def test_observable_reservation_satisfies_its_definition():
    """E[(X−σ)⁺] = λc 를 σ 에서 실제로 만족하는가 (정의 잔차)."""
    from src.engine.pandora import observable_prize_grid
    from src.engine.prize import perbox_reservation, perbox_excess
    p = np.array([0.3, 0.6, 0.85])
    t = np.array([0.05, 0.10, 0.02])
    vals, prob = observable_prize_grid(p, _Chan([0.4, 0.8, 1.6]))
    sig = perbox_reservation(t, vals, prob)
    exc, _ = perbox_excess(sig, vals, prob)
    assert np.allclose(exc, t, atol=1e-6), f"정의 잔차 {exc - t}"


def test_latent_reservation_overstates_and_can_flip_order():
    """잠재 Q 로 풀면 σ 가 과대되고, 예리함이 다르면 상자 순서가 뒤집힌다.

    이것이 수정 이유다 — 재발하면 이 테스트가 실패한다.
    """
    from src.engine.pandora import observable_prize_grid
    from src.engine.prize import bernoulli_reservation, perbox_reservation
    # (a) 과대: 잡음이 커질수록 격차가 커진다
    gaps = []
    for sd in (0.2, 0.6, 2.0):
        p, t = np.array([0.6]), np.array([0.05])
        lat = float(bernoulli_reservation(p, t)[0])
        obs = float(perbox_reservation(t, *observable_prize_grid(p, _Chan([sd])))[0])
        gaps.append(lat - obs)
    assert gaps[0] >= -1e-9 and gaps[-1] > 0.3, f"과대 격차 추이 {gaps}"
    assert gaps[0] < gaps[1] < gaps[2], "잡음이 커질수록 과대가 심해져야 한다"
    # (b) 순서 역전
    p, t = np.array([0.55, 0.58]), np.array([0.12, 0.12])
    lat = bernoulli_reservation(p, t)
    obs = perbox_reservation(t, *observable_prize_grid(p, _Chan([0.30, 1.50])))
    assert int(np.argmax(lat)) != int(np.argmax(obs)), "순서 역전 사례가 재현되어야 한다"
    assert int(np.argmax(obs)) == 0, "관측 상금 기준으로는 예리한 상자를 열어야 한다"


def test_sharp_channel_recovers_bernoulli_closed_form():
    """검증기가 예리해질수록 관측 상금 해가 잠재 Q 닫힌형으로 수렴해야 한다 (일관성).

    ★ 왜 `ExactNoise` 로 하지 않는가 (Phase 17에 발견한 경계): `ExactNoise` 는 3a DP 검증용
    스텁이라 `liks` 가 **밀도가 아니다**(정확 관측을 `(v, 1−v)` 로 인코딩). 따라서 연속 v
    그리드 위의 분포를 만드는 `observable_prize_grid` 와는 의미론이 맞지 않는다. 배포 경로의
    `NoiseModel`/`PerModelNoise` 는 정규 클래스조건 밀도이므로 문제가 없다.
    수렴은 **sd→0 인 실제 밀도 채널**로 확인하는 것이 옳다.
    """
    from src.engine.pandora import observable_prize_grid
    from src.engine.prize import bernoulli_reservation, perbox_reservation
    p = np.array([0.25, 0.5, 0.9])
    t = np.array([0.05, 0.2, 0.6])
    lat = bernoulli_reservation(p, t)
    prev = None
    for sd in (0.5, 0.2, 0.05):
        obs = perbox_reservation(t, *observable_prize_grid(p, _Chan([sd] * 3)))
        gap = float(np.abs(obs - lat).max())
        if prev is not None:
            assert gap < prev, f"sd={sd} 에서 수렴하지 않음 (gap {gap} >= {prev})"
        prev = gap
    assert prev < 0.05, f"예리한 채널에서 닫힌형과 일치해야 한다 (gap {prev})"


# ---------- 2. posterior 상금의 v 이중계산 ----------

def test_posterior_prize_uses_pre_update_prior():
    """update() 후 prize() 를 부를 때 v 가 두 번 반영되면 안 된다."""
    from src.engine.pandora import GridBelief, NoiseModel
    rng = np.random.default_rng(0)
    v = rng.uniform(size=400)
    y = (v > 0.5).astype(float)
    noise = NoiseModel().fit(v, y, steps=400)
    a, b = np.array([1.0, 1.2]), np.array([0.0, 0.3])
    kw = dict(a=a, b_dom=b, noise=noise, prize_mode="posterior")
    bel_a = GridBelief(0.0, 1.0, **kw)
    prize_no_update = bel_a.prize(0.9, 0)          # 갱신 없이 (사전 = 현재 사후)
    bel_b = GridBelief(0.0, 1.0, **kw)
    bel_b.update(0, 0.9)
    prize_after = bel_b.prize(0.9, 0)              # 갱신 후 — 같은 사전을 써야 한다
    assert abs(prize_no_update - prize_after) < 1e-9, (
        "posterior 상금이 관측 전 사전확률을 쓰지 않는다 = v 이중계산")


# ---------- 3. open_order 인자가 실제로 정책에 전달되는가 ----------

def test_open_order_reaches_the_policy():
    """router 가 만든 정책이 open_order 를 들고 있어야 한다 (미전달 결함 재발 방지)."""
    from src.router import LPBRouter
    from tests.test_packaging import _tiny_dataset, MIN_CFG
    ds = _tiny_dataset()
    tr = np.arange(60, 240)
    for order in ("sigma", "value"):
        r = LPBRouter(MIN_CFG, 1, seed=0, use_domain=False, open_order=order).fit(ds, tr, "fast")
        pol = r._factory(r.lam0)
        assert getattr(pol, "open_order", None) == order, (
            f"open_order={order} 가 정책에 전달되지 않았다 (A/B 가 조용히 무효화됨)")


# ---------- 4. 비용 추정/과금 분리와 안전마진 ----------

def test_charge_costs_separate_from_planning_costs():
    from src.harness import Session
    s = Session(costs=np.array([1.0]), verifier_row=np.array([0.5]),
                remaining_budget=1.5, charge_costs=np.array([2.0]))
    assert bool(s.can_afford(0))                   # 계획상 감당 가능(추정 1.0)
    assert s.call(0) is None                       # 실제 2.0 → 호스트 거부
    assert s.rejected == 1 and s.spent == 0.0 and s.called == []


def test_cost_margin_prevents_rejection():
    from src.harness import Session
    kw = dict(verifier_row=np.array([0.5]), remaining_budget=1.1,
              costs=np.array([1.0]), charge_costs=np.array([1.05]))
    assert Session(**kw).call(0) is not None                 # 마진 0 → 통과 (실제 1.05 ≤ 1.1)
    s = Session(**kw, cost_margin=0.5)                       # 마진 50% → 계획 1.5 > 1.1
    assert s.call(0) is None and s.rejected == 0             # 계획 단계에서 거른다 (거부 아님)


def test_cost_estimator_is_honest_and_bounded():
    from src.cost_model import DecodeLengthEstimator, estimated_cost_matrix, estimation_report
    from tests.test_packaging import _tiny_dataset
    ds = _tiny_dataset()
    # 출력 길이는 항상 양수여야 한다 (음수 길이면 상대오차가 발산해 테스트가 무의미해진다)
    ds.out_tokens = (120 + 30 * np.abs(ds.features[:, :1])
                     + np.arange(ds.m)[None, :] * 10).astype(int)
    est = DecodeLengthEstimator(mode="ridge").fit(ds.features[:150], ds.out_tokens[:150])
    cm = estimated_cost_matrix(ds, est)
    assert cm.shape == (ds.n, ds.m) and (cm > 0).all()
    rep = estimation_report(ds, cm)
    assert 0.0 <= rep["mae_rel"] < 1.0 and 0.0 <= rep["frac_underestimated"] <= 1.0


# ---------- 5. 무감독 군집 pseudo-domain ----------

def test_cluster_pseudo_domain_is_deterministic_and_runtime_only():
    """군집은 시드 고정 결정론이고, **프롬프트 특성만으로** 집단을 정해야 한다."""
    from src.cluster import PromptClusterer
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(m, 0.3, size=(80, 5)) for m in (-2.0, 0.0, 2.0)])
    a = PromptClusterer(k=3, seed=0).fit(X)
    b = PromptClusterer(k=3, seed=0).fit(X)
    assert (a.predict(X) == b.predict(X)).all(), "동일 시드에서 결정론이어야 한다"
    lab = a.predict(X)
    assert len(set(lab.tolist())) == 3
    # 같은 덩어리는 같은 군집에 (분리 가능한 데이터에서)
    for g in range(3):
        seg = lab[g * 80:(g + 1) * 80]
        assert (seg == seg[0]).mean() > 0.9
    assert a.predict_row(X[0]) == int(lab[0])


def test_cluster_mode_router_needs_no_domain_label():
    """use_domain="cluster" 라우터는 route() 에 도메인을 안 줘도 동일 결정을 내려야 한다."""
    from src.router import LPBRouter
    from src.harness import Session
    from src.cost_mirror import cost_matrix
    from tests.test_packaging import _tiny_dataset, MIN_CFG
    ds = _tiny_dataset(n=300)
    tr = np.arange(80, 300)
    r = LPBRouter(MIN_CFG, 1, seed=0, use_domain="cluster", n_clusters=3).fit(ds, tr, "fast")
    assert r.clusterer is not None and r.irt["b"].shape[0] == r.clusterer.n_clusters
    cmat = cost_matrix(ds)
    pol = r._factory(r.lam0)
    for i in range(20):
        outs = []
        for dom in (0, 7, 123):                    # 엉뚱한 도메인을 줘도 무시해야 한다
            sess = Session(costs=cmat[i], verifier_row=ds.verifier[i], remaining_budget=1e6)
            outs.append((pol.route(sess, ds.features[i], dom), tuple(sess.called)))
        assert len(set(outs)) == 1, "군집 모드가 외부 도메인 라벨에 반응했다"


# ---------- 6. 인증서 split 분리 ----------

def test_certificate_split_is_disjoint_from_tuning_split():
    from src.router import LPBRouter
    from tests.test_packaging import _tiny_dataset, MIN_CFG
    ds = _tiny_dataset(n=400)
    r = LPBRouter(MIN_CFG, 1, seed=0, use_domain=False).fit(ds, np.arange(100, 400), "fast")
    assert len(np.intersect1d(r._tr_cal, r._tr_cert)) == 0, "λ 튜닝셋과 인증셋이 겹친다"
    assert r.certificate["n_cal"] == len(r._tr_cert)
    assert "dependent" in r.certificate and "note" in r.certificate
    d = r.diagnostics
    assert 0.0 <= d["sigma_neg_frac"] <= 1.0 and d["calls_per_query"] >= 1.0 - 1e-9


# ---------- 7. 페이싱이 reset 없이도 동작 ----------

def test_answer_wrappers_are_unwrapped_before_matching():
    """실 응답의 포장(`['D']`)을 벗기지 않으면 구조 검사와 일치 판정이 통째로 죽는다.

    Phase 18 진단: RouterBench 객관식 응답이 `['D']` 로 오는데 검사기가 이를 못 읽어
    mmlu/arc/hellaswag 4,686셀 전부에서 0(std 0.000)이었다. 파서 버그를 "구조 검사는
    무용"으로 오독할 뻔했던 사건이라, 포장 해제와 정규형 일치를 테스트로 고정한다.
    """
    from src.verifier import _struct_choice, _struct_number, _norm_tail, _unwrap
    assert _unwrap("['D']") == "D" and _unwrap('("42")') == "42"
    p = "Q\n(A) x\n(B) y\n(C) z\n(D) w"
    assert _struct_choice(p, "['D']") == 1.0
    assert _struct_choice(p, "D") == 1.0
    assert _struct_number("", "['1,234']") == 1.0
    # 정규형 일치: 표기가 달라도 같은 답이면 같아야 한다
    assert _norm_tail("['D']", "mmlu") == _norm_tail("The answer is D.", "mmlu") == "d"
    assert _norm_tail("so we get 1,234", "gsm8k") == _norm_tail("['1234']", "gsm8k") == "1234"
    # 태스크를 모르면 기존 문자열 정규화로 퇴화 (무해)
    assert _norm_tail("['D']", "unknown_task") == "d"


def test_paced_policy_survives_missing_reset():
    """주최측 하네스가 begin_tier 훅을 부르지 않아도 죽지 않아야 한다."""
    from src.engine.pacing import PacedPandora

    class _Dummy:
        def __init__(self, lam):
            self.lam = lam

        def route(self, sess, f, d):
            return 0

    pol = PacedPandora(lambda l: _Dummy(l), lam0=0.5)
    pol.observe_spend(0.3)                          # reset 없이 호출
    assert pol.inner.lam == 0.5                     # 예산을 모르면 λ 고정으로 퇴화
