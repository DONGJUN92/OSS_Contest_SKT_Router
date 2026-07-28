"""고도화 후보 검증: 중첩 실패(corr) 체제에서 θ 베이지안 갱신의 가치 재시험.

배경: D7(Phase 3)은 완만한 ICC 세계(A/B)에서 "갱신 무효"를 판정했다. corr 세계는
ICC가 급경사(중첩 실패)라 갱신의 예언된 유효 체제 — 싼 모델의 실패 관측이 θ 사후를
급락시켜 중간 모델의 지수를 무너뜨리면 '승급 스킵'이 자동으로 일어나야 한다.

가설 H-skip: corr(급경사)에서 update-on > update-off. 기각되면 명시적 스킵 휴리스틱은
추가 검토 없이 폐기(신념 경로로 안 되는 것이 휴리스틱으로 될 근거 없음 + 과적합 위험).

스택: 최종형과 동일 (투표 상자 + 페이싱 v3 + quality-first λ + PerModelNoise).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from src.harness import run_tier, tier_budgets
from src.irt.mml import fit_mml
from src.engine.pandora import PerModelNoise, GridBelief, PandoraPolicy, tune_lambda_quality
from src.engine.pacing import PacedPandora
from phase2_stages import IRTEncoder
from run_phase2 import CFG
from run_phase7 import build_world

N_DOM = CFG["synth"]["n_domains"]


def make_policy_factory(irt, enc, noise, do_update):
    def fac(lam):
        def make_belief(x, dom):
            mu, s = enc.belief_row(x)
            return GridBelief(mu, s, irt["a"], irt["b"][dom], noise, do_update=do_update)
        return PandoraPolicy(make_belief, lam, name=f"upd={do_update}")
    return fac


def run(world):
    base, eds, folds = build_world(world)
    print(f"\n[probe-update | World: {world}] (최종 스택, 5-fold)")
    # 적합된 판별도 확인: 급경사 체제인지 (D7 예언 조건의 사전 점검)
    tr0 = np.setdiff1d(np.arange(eds.n), folds[0])
    irt0 = fit_mml(eds.quality[tr0], eds.domains[tr0], N_DOM, per_domain_b=True)
    print(f"  적합 판별도 a: min={irt0['a'].min():.2f} median={np.median(irt0['a']):.2f} "
          f"max={irt0['a'].max():.2f}  (World A 대비 급경사 여부)")
    for tier in ["fast", "balanced"]:
        res = {}
        for upd in [True, False]:
            qs, calls = [], []
            for f in range(CFG["eval"]["k_folds"]):
                test_idx = folds[f]
                tr = np.setdiff1d(np.arange(eds.n), test_idx)
                rng = np.random.default_rng(f)
                perm = rng.permutation(tr)
                tr_fit = perm[:int(0.7 * len(perm))]
                tr_cal = perm[int(0.7 * len(perm)):]
                irt = fit_mml(eds.quality[tr_fit], eds.domains[tr_fit], N_DOM,
                              per_domain_b=True)
                enc = IRTEncoder(irt["a"], irt["b"]).fit(
                    eds.features[tr_fit], eds.domains[tr_fit], eds.quality[tr_fit])
                noise = PerModelNoise().fit(eds.verifier[tr_fit], eds.quality[tr_fit])
                fac = make_policy_factory(irt, enc, noise, upd)
                b_cal = tier_budgets(eds, tr_cal, CFG)[tier]
                lam0 = tune_lambda_quality(fac, eds, tr_cal, b_cal)
                pol = PacedPandora(fac, lam0, name=f"upd={upd}")
                b_te = tier_budgets(eds, test_idx, CFG)[tier]
                r = run_tier(eds, test_idx, pol, tier, b_te)
                qs.append(r.mean_quality)
                calls.append(r.calls_per_query)
            res[upd] = (np.mean(qs), np.std(qs), np.mean(calls))
        d = res[True][0] - res[False][0]
        print(f"  {tier:<10} update-on {res[True][0]:.4f}+-{res[True][1]:.4f} "
              f"(calls {res[True][2]:.2f})  vs  off {res[False][0]:.4f}+-{res[False][1]:.4f} "
              f"(calls {res[False][2]:.2f})   delta={d:+.4f}")


if __name__ == "__main__":
    for w in (sys.argv[1:] or ["corr"]):
        run(w)
