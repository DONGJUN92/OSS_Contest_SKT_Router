"""합성 데이터 생성기 — 두 개의 세계 (self-reflection 장치).

World A "irt"        : 1D 잠재 난이도 θ가 실제로 존재. LPB-Router의 가정이 성립하는 세계.
World B "specialist" : 도메인 특화 소형 모델이 대형 모델을 특정 도메인에서 추월.
                       1D θ 가정이 구조적으로 깨지는 세계 — 우리 방법의 자기확증 편향을
                       차단하기 위한 적대적(misspecified) 시뮬레이션.

두 세계 모두에서 이기지 못하는 설계는 채택하지 않는다는 것이 Phase별 검증 원칙.
"""
import numpy as np
from .schema import Dataset, ModelMeta


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _common(cfg, rng, n):
    n_dom = cfg["synth"]["n_domains"]
    domains = rng.integers(0, n_dom, size=n)
    in_tokens = np.clip(rng.lognormal(mean=5.3, sigma=0.6, size=n), 50, 4000).astype(int)
    return domains, in_tokens


def _out_tokens(cfg, rng, n, model_ids):
    means = np.array([cfg["models"][mid]["avg_out_tokens"] for mid in model_ids])
    return np.clip(
        rng.normal(means[None, :], means[None, :] * 0.25, size=(n, len(model_ids))), 20, None
    ).astype(int)


def _verifier(quality, sigma, rng):
    """검증기 관측 = 품질 + 가우시안 잡음, [0,1] 클립. 실데이터에선 검증기 모델 출력."""
    return np.clip(quality + rng.normal(0, sigma, size=quality.shape), 0.0, 1.0)


def make_world(cfg: dict, world: str, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    model_ids = list(cfg["models"].keys())
    n, m = cfg["synth"]["n_queries"], len(model_ids)
    d = cfg["synth"]["feature_dim"]
    domains, in_tokens = _common(cfg, rng, n)
    out_tokens = _out_tokens(cfg, rng, n, model_ids)
    extras = {}

    if world == "irt":
        # 1D 잠재 용이도 θ (높을수록 쉬움) + 도메인별 평균 이동
        dom_shift = rng.normal(0, 0.4, size=cfg["synth"]["n_domains"])
        theta = rng.normal(0, 1, size=n) + dom_shift[domains]
        a_lo, a_hi = cfg["synth"]["irt"]["a_range"]
        a = rng.uniform(a_lo, a_hi, size=m)
        b = np.array(cfg["synth"]["irt"]["b_values"])
        p = _sigmoid(a[None, :] * (theta[:, None] - b[None, :]))
        extras = {"theta": theta, "a": a, "b": b}
    elif world == "specialist":
        # 모델별 기본 정답률 + 도메인 특화 부스트: 도메인 k의 전담 소형 모델이 대형을 추월
        base = np.array(cfg["synth"]["specialist"]["base_skill"])
        boost = cfg["synth"]["specialist"]["specialist_boost"]
        n_dom = cfg["synth"]["n_domains"]
        skill = np.tile(base, (n_dom, 1))                    # (D, M)
        for dom in range(n_dom):
            specialist = dom % (m - 1)                        # 최상위 모델 제외한 소형에 특화 부여
            skill[dom, specialist] = min(0.97, base[specialist] + boost)
        slope = cfg["synth"]["specialist"]["difficulty_slope"]
        difficulty = rng.beta(2, 2, size=n)                   # 쿼리별 난이도 (0 쉬움, 1 어려움)
        p = np.clip(skill[domains] - slope * (difficulty[:, None] - 0.5), 0.02, 0.98)
        extras = {"skill": skill, "difficulty": difficulty}
    else:
        raise ValueError(f"unknown world: {world}")

    quality = (rng.uniform(size=(n, m)) < p).astype(float)
    verifier = _verifier(quality, cfg["synth"]["verifier_noise"], rng)

    # 쿼리 특성: 잠재 변수의 잡음 섞인 관측 + 도메인 one-hot (실데이터에선 인코더 출력)
    latent = extras.get("theta", extras.get("difficulty"))
    w = rng.normal(0, 1, size=d)
    features = latent[:, None] * w[None, :] + rng.normal(0, 0.7, size=(n, d))
    dom_onehot = np.eye(cfg["synth"]["n_domains"])[domains]
    features = np.hstack([features, dom_onehot])

    models = {
        mid: ModelMeta(mid, cfg["models"][mid]["prefill_price"], cfg["models"][mid]["decode_price"])
        for mid in model_ids
    }
    return Dataset(
        model_ids=model_ids, models=models, domains=domains, features=features,
        in_tokens=in_tokens, out_tokens=out_tokens, quality=quality,
        verifier=verifier, world=world, extras=extras,
    )
