"""B4 지연 A/B — σ 일반해 도입이 tie-break 자산(지연)을 깎았는가?

동일 데이터·동일 정책에서 예약값 계산부만 구/신으로 갈아끼워 측정한다.
(구식은 σ<0 구간이 틀렸으므로 성능 비교가 아니라 **지연 비용**만 보는 목적이다.)

사용법: python eval/probe_b4_latency.py
"""
import sys, pathlib, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from src.engine import pandora
from src.engine.pandora import PandoraPolicy
from src.harness import run_tier, tier_budgets
from src.router import LPBRouter
from src.synth import make_world
from src.textworld import build_textworld, to_dataset
from src.text_encoder import get_encoder
from src.verifier import feature_matrix, fold_verifier_matrix
from run_phase2 import CFG


def _route_old(self, sess, features, domain) -> int:
    """B4 이전의 route (예약값 = 1 − λc/p̄ 인라인)."""
    bel = self.make_belief(features, domain)
    M = len(sess.costs)
    unopened, obs = set(range(M)), {}
    while unopened:
        pbar = bel.pbar()
        afford = [m for m in unopened if sess.can_afford(m)]
        if not afford:
            break
        sigma = {m: 1.0 - self.lam * sess.costs[m] / max(pbar[m], 1e-9) for m in afford}
        m_star = max(sigma, key=sigma.get)
        if obs and max(bel.prize(v, m) for m, v in obs.items()) \
                >= sigma[m_star] + self.stop_margin:
            break
        v = sess.call(m_star)
        if v is None:
            break
        unopened.discard(m_star)
        obs[m_star] = v
        bel.update(m_star, v)
    if not obs:
        m = int(np.argmin(sess.costs))
        v = sess.call(m)
        if v is not None:
            obs[m] = v
    if not obs:
        return 0
    return max(obs, key=lambda m: bel.prize(obs[m], m))


def bench(ds, tr, te, tier, label, reps=3):
    router = LPBRouter(CFG, len(np.unique(ds.domains)), seed=0).fit(ds, tr, tier)
    budget = tier_budgets(ds, te, CFG)[tier]
    out = {}
    new_route = PandoraPolicy.route
    for name, fn in (("신(일반해)", new_route), ("구(1−λc/p̄)", _route_old)):
        PandoraPolicy.route = fn
        best = min(_timeit(ds, te, router, tier, budget) for _ in range(reps))
        out[name] = best
    PandoraPolicy.route = new_route
    print(f"  {label:<26}" + "   ".join(f"{k} {v:.4f} ms/q" for k, v in out.items())
          + f"   차이 {out['신(일반해)'] - out['구(1−λc/p̄)']:+.4f}")


def _timeit(ds, te, router, tier, budget):
    pol = router.policy()
    t0 = time.perf_counter()
    run_tier(ds, te, pol, tier, budget)
    return (time.perf_counter() - t0) / len(te) * 1e3


if __name__ == "__main__":
    print("[probe] 예약값 계산부 구/신 지연 비교 (동일 라우터·동일 데이터)\n")
    a = make_world(CFG, "irt", seed=7)
    bench(a, np.arange(1000, 2000), np.arange(0, 600), "fast", "World A (상자 5)")

    tw = build_textworld(CFG, seed=42)
    f, meta = to_dataset(tw, CFG)
    f.features = get_encoder("hashing").encode(meta["prompts"])
    tr = np.arange(0, 1600)
    f.verifier, _ = fold_verifier_matrix(feature_matrix(meta), f.quality, tr)
    for tier in ("fast", "balanced"):
        bench(f, tr, np.arange(1600, 2000), tier, f"World F (상자 15, {tier})")
