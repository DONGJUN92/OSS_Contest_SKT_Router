"""예산 수준 스윕 (Phase 19) — 우리 우위가 **예산 가정에 의존하는가**.

왜 필요한가: tier 예산 수준(`budget_frac` 0.08/0.25/0.60)은 주최측 미답변(Q7)이고 **우리 가정**
이다. 그런데 달성 가능 폭이 예산 수준에 극단적으로 의존한다 — frac 0.08 에서 fast tier 의
오라클−하한은 **0.072** 뿐이라 총 달성폭의 12.6% 밖에 안 된다. 즉 "저예산 tier 가 승부처"라는
이 프로젝트의 출발 전제가 **예산 수준에 따라 참/거짓이 갈린다.**

그래서 정책 서열이 예산 수준 전 구간에서 유지되는지 재고, 어느 구간이 실제 승부처인지
(달성폭이 큰 곳) 함께 보고한다. 이것은 리스크 대응이면서 제출 자산이다 — "우리 우위는 예산
수준 가정에 의존하지 않는다"를 수치로 말할 수 있게 된다.

사용법: python eval/probe_budget_sweep.py [--folds 2] [--n 1000]
"""
import sys, pathlib, json, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from src.textworld import build_textworld, to_dataset
from src.features import RichEncoder
from src.verifier import augmented_feature_matrix, fold_verifier_matrix
from src.harness import run_tier
from src.cost_mirror import cost_matrix
from src.router import LPBRouter
from src.encoder import DiscLR
from src.engine.pandora import tune_lambda_quality, FINE_MULTS
from baselines.policies import (LearnedRouter, tune_learned_lambda, BudgetCascade,
                                tune_budget_cascade_tau, CascadeRouting, AllModel)

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config_ax3.yaml", encoding="utf-8"))
OUT = ROOT / "eval" / "results"
FRACS = [0.03, 0.06, 0.12, 0.25, 0.50, 1.0]
POLS = ["lpb", "lpb_abstain", "learned", "cascade_budget", "cascade_routing"]


def _band(ds, cmat, idx, budget):
    """(하한, 상한) — **둘 다 예산 제약을 지킨다.**

    ★ 초판 결함 (Phase 19 자체 발견): 하한을 "예산 무시 always-cheapest 평균"으로 계산했다.
    그런데 아주 빠듯한 예산(frac 0.03·0.06)에서는 최저가조차 전 쿼리에 못 쓰므로 그 하한이
    **실현 불가능**했고, 달성폭이 0 으로 나와 백분율이 −2e10% 같은 쓰레기가 됐다. 또 상한
    그리디도 그 실현 불가능한 배정에서 출발해 틀린 값을 냈다.

    수정: 하한 = 예산 안에서 최저가를 살 수 있는 만큼만 사고 나머지는 무응답(0점).
          상한 = (쿼리, 모델) 후보를 품질/비용 비로 정렬해 예산 안에서 담는 그리디
                 (쿼리당 최대 1개, 아무것도 안 사는 것도 허용) — 그룹당 1개 배낭의 LP 완화.
    """
    q, c = ds.quality[idx], cmat[idx]
    n = len(idx)
    # 하한: 최저가를 예산 소진까지
    ch_lo = np.argmin(c, axis=1)
    cost_lo = c[np.arange(n), ch_lo]
    spent, qs = 0.0, np.zeros(n)
    for r in range(n):
        if spent + cost_lo[r] <= budget + 1e-12:
            spent += cost_lo[r]
            qs[r] = q[r, ch_lo[r]]
    floor = float(qs.mean())
    # 상한: 품질/비용 비 그리디 (쿼리당 1개, 미구매 허용)
    cands = [(q[r, m] / max(c[r, m], 1e-12), r, m)
             for r in range(n) for m in range(ds.m) if q[r, m] > 0]
    cands.sort(reverse=True)
    taken = np.zeros(n, dtype=bool)
    spent, qh = 0.0, np.zeros(n)
    for _, r, m in cands:
        if taken[r] or spent + c[r, m] > budget + 1e-12:
            continue
        taken[r] = True
        spent += c[r, m]
        qh[r] = q[r, m]
    return floor, float(qh.mean())


def main(nf=2, n_queries=1000):
    t0 = time.time()
    cfg = dict(CFG, synth=dict(CFG["synth"], n_queries=n_queries))
    tw = build_textworld(cfg, seed=42)
    ds, meta = to_dataset(tw, cfg, Ns=(1,))
    ds.features = RichEncoder().fit(meta["prompts"]).encode(meta["prompts"])
    F = augmented_feature_matrix(meta, text_dim=64, use_prompt=True, use_agreement=True)
    cmat = cost_matrix(ds)
    folds = ds.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])
    order = list(np.argsort(cmat.mean(axis=0)))
    print(f"[probe_budget_sweep] n={ds.n} M={ds.m} folds={nf}", flush=True)
    print(f"\n  {'frac':<7}{'하한':<8}{'상한':<8}{'달성폭':<8}" +
          "".join(f"{p:<17}" for p in POLS) + "1위")
    res = {}
    for frac in FRACS:
        acc = {p: [] for p in POLS}
        band = []
        for f in range(nf):
            te = folds[f]
            tr = np.setdiff1d(np.arange(ds.n), te)
            ds.verifier, _ = fold_verifier_matrix(F, ds.quality, tr)
            b_te = frac * cmat[te].max(axis=1).sum()
            b_tr = frac * cmat[tr].max(axis=1).sum()
            band.append(_band(ds, cmat, te, b_te))
            # tier 이름은 예산을 직접 넘기므로 무의미 — config 의 frac 을 우회한다
            cfg_f = dict(CFG, tiers={"x": {"budget_frac": frac, "weight": 1.0}})
            r = LPBRouter(cfg_f, 1, seed=f, use_domain=False).fit(ds, tr, "x")
            acc["lpb"].append(run_tier(ds, te, r.policy(), "x", b_te).mean_quality)
            # 전략적 기권 arm (Phase 19): p̄ < λc 인 쿼리는 아예 사지 않고 예산을 아낀다
            ra = LPBRouter(cfg_f, 1, seed=f, use_domain=False,
                           allow_abstain=True).fit(ds, tr, "x")
            acc["lpb_abstain"].append(run_tier(ds, te, ra.policy(), "x", b_te).mean_quality)
            disc = DiscLR().fit(ds.features[tr], ds.domains[tr], ds.quality[tr])
            acc["learned"].append(run_tier(
                ds, te, LearnedRouter(disc, tune_learned_lambda(disc, ds, tr, b_tr)),
                "x", b_te).mean_quality)
            acc["cascade_budget"].append(run_tier(
                ds, te, BudgetCascade(order, tune_budget_cascade_tau(ds, tr, cfg_f, "x", b_tr)),
                "x", b_te).mean_quality)
            mb = r._factory(1.0).make_belief
            lam = tune_lambda_quality(lambda l: CascadeRouting(mb, l), ds, tr, b_tr,
                                      mults=FINE_MULTS, se_k=1.0)   # 대등 튜닝
            acc["cascade_routing"].append(run_tier(ds, te, CascadeRouting(mb, lam), "x",
                                                   b_te).mean_quality)
        lo = float(np.mean([b[0] for b in band]))
        hi = float(np.mean([b[1] for b in band]))
        q = {p: float(np.mean(acc[p])) for p in POLS}
        win = max(q, key=q.get)
        res[str(frac)] = {"floor": round(lo, 4), "ceiling": round(hi, 4),
                          "span": round(hi - lo, 4),
                          "scores": {p: round(q[p], 4) for p in POLS},
                          "capture_pct": {p: round((q[p] - lo) / max(hi - lo, 1e-9) * 100, 1)
                                          for p in POLS},
                          "winner": win}
        print(f"  {frac:<7}{lo:<8.4f}{hi:<8.4f}{hi - lo:<8.4f}" +
              "".join(f"{q[p]:.4f}({(q[p]-lo)/max(hi-lo,1e-9)*100:4.0f}%) " for p in POLS)
              + win, flush=True)

    wins = [res[k]["winner"] for k in res]
    lpb_ranks = []
    for k in res:
        s = res[k]["scores"]
        lpb_ranks.append(sorted(s, key=lambda p: -s[p]).index("lpb") + 1)
    res["_verdict"] = {
        "winners_by_frac": {k: res[k]["winner"] for k in res if not k.startswith("_")},
        "lpb_rank_by_frac": lpb_ranks,
        "lpb_rank_stable": len(set(lpb_ranks)) == 1,
        "largest_span_frac": max((k for k in res if not k.startswith("_")),
                                 key=lambda k: res[k]["span"]),
        "note": ("달성폭(span)이 큰 frac 이 실제 승부처다. lpb_rank 가 frac 전 구간에서 "
                 "동일하면 우위가 예산 가정에 의존하지 않는다."),
    }
    print(f"\n  LPB 순위(frac 순): {lpb_ranks}  → 안정: {res['_verdict']['lpb_rank_stable']}")
    print(f"  달성폭 최대 구간: frac {res['_verdict']['largest_span_frac']} "
          f"(span {res[res['_verdict']['largest_span_frac']]['span']})")
    print(f"  구간별 1위: {res['_verdict']['winners_by_frac']}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "probe_budget_sweep.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"  ({time.time() - t0:.0f}s) → eval/results/probe_budget_sweep.json")
    return res


if __name__ == "__main__":
    nf = int(sys.argv[sys.argv.index("--folds") + 1]) if "--folds" in sys.argv else 2
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 1000
    main(nf, n)
