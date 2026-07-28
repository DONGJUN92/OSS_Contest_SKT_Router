"""B4 사전 조사 ②: 음수 σ 구간이 **실제 라우터 실행 중에** 발생하는가?

probe_negative_sigma.py가 닫힌형의 λc>p̄ 구간 결함을 확인했다. 그러나 그 구간을
실제로 밟지 않는다면 휴면 결함이다. 여기서는 학습된 λ와 실제 p̄로 도는 라우터의
매 결정 시점을 계측한다.

계측 항목
  neg_all : 후보 상자 전부가 σ<0 (→ 강제 1회 개봉 후 즉시 정지. argmax가 곧 점수)
  neg_any : 일부 상자가 σ<0
  argmax_diff : 현행 닫힌형과 참 예약값의 argmax 불일치 (= 실제 오선택)

사용법: python eval/probe_sigma_regime.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from src.engine.pandora import PandoraPolicy
from src.engine.pacing import PacedPandora
from src.harness import run_tier, tier_budgets
from src.router import LPBRouter
from run_phase2 import CFG
from run_phase4 import ext_worlds

TIERS = ["fast", "balanced", "premium"]
STATS = {}


class ProbePandora(PandoraPolicy):
    """route()를 그대로 복제하되 결정 시점마다 σ 체제를 기록한다."""

    def route(self, sess, features, domain) -> int:
        bel = self.make_belief(features, domain)
        M = len(sess.costs)
        unopened, obs = set(range(M)), {}
        while unopened:
            pbar = bel.pbar()
            afford = [m for m in unopened if sess.can_afford(m)]
            if not afford:
                break
            c = np.array([sess.costs[m] for m in afford])
            p = np.maximum(np.array([pbar[m] for m in afford]), 1e-9)
            s_cur = 1.0 - self.lam * c / p
            s_exact = np.where(s_cur >= 0.0, s_cur, p - self.lam * c)
            STATS["decisions"] += 1
            if (s_cur < 0).all():
                STATS["neg_all"] += 1
            elif (s_cur < 0).any():
                STATS["neg_any"] += 1
            if int(np.argmax(s_cur)) != int(np.argmax(s_exact)):
                STATS["argmax_diff"] += 1
                if (s_cur < 0).all():
                    STATS["argmax_diff_negall"] += 1

            sigma = {m: s_cur[j] for j, m in enumerate(afford)}
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


def main():
    worlds, ext, folds = ext_worlds()
    print("[probe] 실행 중 σ 체제 계측 (학습된 λ · 실제 p̄, 5-fold)\n")
    print(f"  {'world':<12}{'tier':<10}{'결정수':>8}{'전부음수%':>11}{'일부음수%':>11}"
          f"{'argmax불일치%':>14}{'그중전부음수':>13}")
    for wname in worlds:
        eds = ext[wname]
        for tier in TIERS:
            for k in ("decisions", "neg_all", "neg_any", "argmax_diff",
                      "argmax_diff_negall"):
                STATS[k] = 0
            for f in range(CFG["eval"]["k_folds"]):
                te = folds[wname][f]
                tr = np.setdiff1d(np.arange(eds.n), te)
                r = LPBRouter(CFG, CFG["synth"]["n_domains"], seed=f).fit(eds, tr, tier)
                pol = PacedPandora(
                    lambda l: ProbePandora(r._factory(l).make_belief, l), r.lam0)
                run_tier(eds, te, pol, tier, tier_budgets(eds, te, CFG)[tier])
            d = max(STATS["decisions"], 1)
            print(f"  {wname:<12}{tier:<10}{STATS['decisions']:>8}"
                  f"{100 * STATS['neg_all'] / d:>11.2f}{100 * STATS['neg_any'] / d:>11.2f}"
                  f"{100 * STATS['argmax_diff'] / d:>14.2f}"
                  f"{STATS['argmax_diff_negall']:>13}")


if __name__ == "__main__":
    main()
