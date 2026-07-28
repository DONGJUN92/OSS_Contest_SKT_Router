"""고도화용 신규 세계 3종 (Phase 7) — 실데이터의 위험 요소를 모사한 적대적 입력.

World C "corr"     : 모델 간 오류 상관(중첩 실패) + 모델 크기에 비례하는 심판 편향.
                     실제 LLM들은 같은 쿼리에서 같이 틀리고(escalation 가치 하락),
                     LLM-judge는 대형 모델의 유창한 오답을 과대평가한다.
World D "crossing" : 문항특성곡선(ICC) 교차 — 고판별(a↑) 모델은 쉬운 문제에 강하고
                     어려운 문제에서 급락, 평탄(a↓) 모델은 어디서나 중간.
                     비용-능력의 단조 서열이 깨진다 (D7 기각의 예언 조건).
World E "nosignal" : 쿼리 특성이 난이도와 완전 무상관 — 프로파일러 최악 조건.
                     라우터는 관측(호출 결과)에만 의존해야 한다.

모든 세계는 기존 스키마를 그대로 사용하며 extras["true_p"]를 저장해
synth_ext.extend_with_votes(투표 상자 확장)와 호환된다.
"""
import numpy as np
from .schema import Dataset, ModelMeta
from .synth import _common, _out_tokens, _sigmoid


def _assemble(cfg, world, rng, domains, in_tokens, p, features, verifier_bias=None,
              extras=None):
    model_ids = list(cfg["models"].keys())
    n, m = len(domains), len(model_ids)
    out_tokens = _out_tokens(cfg, rng, n, model_ids)
    quality = (rng.uniform(size=(n, m)) < p).astype(float)
    sigma_v = cfg["synth"]["verifier_noise"]
    v = quality + rng.normal(0, sigma_v, size=quality.shape)
    if verifier_bias is not None:                     # 오답에만 얹히는 모델별 편향
        v = v + (1 - quality) * verifier_bias[None, :]
    verifier = np.clip(v, 0.0, 1.0)
    models = {mid: ModelMeta(mid, cfg["models"][mid]["prefill_price"],
                             cfg["models"][mid]["decode_price"]) for mid in model_ids}
    ex = {"true_p": p, **(extras or {})}
    return Dataset(model_ids=model_ids, models=models, domains=domains,
                   features=features, in_tokens=in_tokens, out_tokens=out_tokens,
                   quality=quality, verifier=verifier, world=world, extras=ex)


def make_world2(cfg: dict, world: str, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    n = cfg["synth"]["n_queries"]
    d = cfg["synth"]["feature_dim"]
    n_dom = cfg["synth"]["n_domains"]
    domains, in_tokens = _common(cfg, rng, n)
    M = len(cfg["models"])

    if world == "corr":
        # 중첩 실패: 전역 난이도 u, 능력 c_m, 급경사 임계 → 오류가 모델 간 강상관
        u = rng.uniform(0, 1, size=n)
        cap = np.linspace(0.32, 0.88, M)
        k = 9.0
        p = _sigmoid(k * (cap[None, :] - u[:, None]))
        p = np.clip(p * 0.97 + 0.015, 0.02, 0.985)     # 소량의 개별 요동
        judge_bias = np.linspace(0.0, 0.28, M)         # 대형일수록 오답 과대평가
        latent = -u                                     # 특성은 난이도의 잡음 관측
        w = rng.normal(0, 1, size=d)
        features = latent[:, None] * w[None, :] + rng.normal(0, 0.7, size=(n, d))
        features = np.hstack([features, np.eye(n_dom)[domains]])
        return _assemble(cfg, "corr", rng, domains, in_tokens, p, features,
                         verifier_bias=judge_bias,
                         extras={"u": u, "cap": cap, "judge_bias": judge_bias})

    if world == "crossing":
        # ICC 교차: (a, b)를 비용 서열과 비단조로 배치 — 값싼 평탄형이 tail에서 우위.
        # 5모델 기본값. 모델 수가 다르면(예: A.X 3모델) config synth.crossing.{a,b}로 교체.
        theta = rng.normal(0, 1, size=n)
        cw = cfg["synth"].get("crossing", {})
        a = np.array(cw.get("a", [0.45, 2.8, 0.6, 3.2, 1.0]))
        b = np.array(cw.get("b", [-0.6, 0.1, -1.1, -0.4, -1.6]))
        assert len(a) == M and len(b) == M, \
            f"crossing a/b 길이({len(a)},{len(b)})가 모델 수 M={M}와 불일치 — config synth.crossing 확인"
        p = _sigmoid(a[None, :] * (theta[:, None] - b[None, :]))
        w = rng.normal(0, 1, size=d)
        features = theta[:, None] * w[None, :] + rng.normal(0, 0.7, size=(n, d))
        features = np.hstack([features, np.eye(n_dom)[domains]])
        return _assemble(cfg, "crossing", rng, domains, in_tokens, p, features,
                         extras={"theta": theta, "a": a, "b": b})

    if world == "nosignal":
        # World A와 동일한 응답 구조, 그러나 특성은 순수 잡음 (θ와 무상관)
        dom_shift = rng.normal(0, 0.4, size=n_dom)
        theta = rng.normal(0, 1, size=n) + dom_shift[domains]
        a_lo, a_hi = cfg["synth"]["irt"]["a_range"]
        a = rng.uniform(a_lo, a_hi, size=M)
        b = np.array(cfg["synth"]["irt"]["b_values"])
        p = _sigmoid(a[None, :] * (theta[:, None] - b[None, :]))
        features = rng.normal(0, 1, size=(n, d))       # 신호 없음
        features = np.hstack([features, np.eye(n_dom)[domains]])
        return _assemble(cfg, "nosignal", rng, domains, in_tokens, p, features,
                         extras={"theta": theta, "a": a, "b": b})

    raise ValueError(f"unknown world2: {world}")
