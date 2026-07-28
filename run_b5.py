"""B5 — λ 고정 모드 (Q5가 문항당 예산일 경우) 리플레이 검증.

가설 (리스크 레지스터 R5): 예산이 **문항당 상한**이면 쿼리 간 이월이 없어 배분 문제
자체가 사라지고, M3 페이싱(shadow price의 온라인 조절)은 **무의미해진다**. 정책 형태는
그대로이며 λ만 상수가 된다 — "손실"이 아니라 "무의미화"라는 것이 설계 주장이었다.

이 주장을 데이터로 확인한다. 세 팔을 같은 문항당 예산에서 비교:
  ① λ고정   : pacing=False (B5 본체 — 이 체제의 정답이라고 주장한 구성)
  ② 페이싱  : 전역 예산용 컨트롤러를 문항당 체제에 그대로 켠 것 (무해한가? 해로운가?)
  ③ cascade : 정적 threshold 베이스라인

**공정성**: 문항당 예산 = 전역 예산 / 쿼리 수 이므로 총지출 상한이 두 의미론에서 같다.
따라서 ①과 전역 기준 점수의 차이는 곧 **쿼리 간 이월의 가치**를 뜻한다.

사용법: python run_b5.py
"""
import sys, pathlib, json

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

from src.cost_mirror import cost_matrix
from src.harness import run_tier, tier_budgets
from src.router import LPBRouter
from src.synth import make_world
from src.synth2 import make_world2
from src.synth_ext import extend_with_votes
from baselines.policies import StaticCascade, tune_cascade_tau
from run_phase2 import CFG, OUT

TIERS = ["fast", "balanced", "premium"]
W = {t: CFG["tiers"][t]["weight"] for t in CFG["tiers"]}
WORLDS = ["irt", "specialist", "corr", "crossing", "nosignal"]
# 전역 예산(현행 의미론) 기준 점수 — 이월의 가치를 재는 대조
GLOBAL_REF = {"irt": 0.9486, "specialist": 0.9982, "corr": 0.9259,
              "crossing": 0.9907, "nosignal": 0.9615}


def build(wname):
    base = make_world(CFG, wname, seed=CFG["seed"]) if wname in ("irt", "specialist") \
        else make_world2(CFG, wname, seed=CFG["seed"])
    conc = 0.6 if wname == "crossing" else None
    return extend_with_votes(base, CFG, conc=conc), \
        base.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])


def eval_arm(ds, folds, tier, pacing):
    qs, uns = [], []
    for f in range(CFG["eval"]["k_folds"]):
        te = folds[f]
        tr = np.setdiff1d(np.arange(ds.n), te)
        r = LPBRouter(CFG, CFG["synth"]["n_domains"], seed=f,
                      per_query_budget=True, pacing=pacing).fit(ds, tr, tier)
        b_te = tier_budgets(ds, te, CFG, per_query=True)[tier]
        res = run_tier(ds, te, r.policy(), tier, b_te, per_query=True)
        qs.append(res.mean_quality)
        uns.append(100.0 * res.unanswered / len(te))
    return float(np.mean(qs)), float(np.std(qs)), float(np.mean(uns))


def main():
    print("[b5] 문항당 예산 체제 (Q5=per-query) — λ 고정 vs 페이싱 vs cascade\n")
    print(f"  {'world':<12}{'λ고정':>9}{'페이싱':>9}{'차이':>8}{'cascade':>9}"
          f"{'전역기준':>10}{'이월가치':>10}{'무응답%':>9}")
    out = {}
    for w in WORLDS:
        ds, folds = build(w)
        order = list(np.argsort(cost_matrix(ds).mean(axis=0)))
        fx, px, cs = {}, {}, {}
        for tier in TIERS:
            fx[tier] = eval_arm(ds, folds, tier, pacing=False)
            px[tier] = eval_arm(ds, folds, tier, pacing=True)
            qs = []
            for f in range(CFG["eval"]["k_folds"]):
                te = folds[f]
                tr = np.setdiff1d(np.arange(ds.n), te)
                b_tr = tier_budgets(ds, tr, CFG, per_query=True)[tier]
                tau = tune_cascade_tau(ds, tr, CFG, tier, b_tr * len(tr))
                b_te = tier_budgets(ds, te, CFG, per_query=True)[tier]
                qs.append(run_tier(ds, te, StaticCascade(order, tau), tier, b_te,
                                   per_query=True).mean_quality)
            cs[tier] = float(np.mean(qs))
        cf = sum(W[t] * fx[t][0] for t in TIERS)
        cp = sum(W[t] * px[t][0] for t in TIERS)
        cc = sum(W[t] * cs[t] for t in TIERS)
        un = float(np.mean([fx[t][2] for t in TIERS]))
        out[w] = {"fixed": round(cf, 4), "paced": round(cp, 4), "cascade": round(cc, 4),
                  "global_ref": GLOBAL_REF[w], "unans_pct": round(un, 2),
                  "tiers": {t: {"fixed": round(fx[t][0], 4), "fixed_std": round(fx[t][1], 4),
                                "paced": round(px[t][0], 4), "cascade": round(cs[t], 4)}
                            for t in TIERS}}
        print(f"  {w:<12}{cf:>9.4f}{cp:>9.4f}{cp - cf:>+8.4f}{cc:>9.4f}"
              f"{GLOBAL_REF[w]:>10.4f}{GLOBAL_REF[w] - cf:>+10.4f}{un:>9.2f}")

    d = [out[w]["paced"] - out[w]["fixed"] for w in WORLDS]
    carry = [out[w]["global_ref"] - out[w]["fixed"] for w in WORLDS]
    print(f"\n  페이싱 효과 (문항당 체제): 평균 {np.mean(d):+.4f}  "
          f"최대 {max(d, key=abs):+.4f}  → {'무의미(R5 주장 확인)' if abs(np.mean(d)) < 0.005 else '유의미 — R5 재검토 필요'}")
    print(f"  쿼리 간 이월의 가치      : 평균 {np.mean(carry):+.4f}  "
          f"(전역 예산이 문항당보다 이만큼 유리)")
    wins = sum(1 for w in WORLDS if out[w]["fixed"] >= out[w]["cascade"] - 0.003)
    print(f"  λ고정 최상위/동률        : {wins}/{len(WORLDS)}")
    json.dump(out, open(OUT / "b5.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
