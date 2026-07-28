"""런타임 도메인 부재 대응 — 단일집단 퇴화(use_domain=False)의 실제 비용 측정.

배경: 대회 상세("런타임에 benchmark/task 이름 미제공")상 라우터는 추론 시 도메인
라벨을 볼 수 없다. 그러나 현행은 다집단 2PL(b_{d,m})이고 submission 어댑터는 도메인
미지정 시 예외를 던진다(D18). 오프라인 하네스는 ds.domains가 있어 이 갭이 가려진다.

이 프로브는 LPBRouter(use_domain=False) — 다집단을 단일집단(b_m)으로 퇴화 — 의 종합
점수를 현행(use_domain=True)과 6검증 세계에서 fold별로 비교한다. "도메인을 못 쓰면
얼마를 잃는가"를 수치화해 설계 결정(단일집단 퇴화 vs 프롬프트→도메인 분류기 vs
도메인 주변화)의 근거를 만든다.

★ World F(textworld)가 가장 현실적인 세계다: 특성이 텍스트 인코더 출력이라 합성 세계와
달리 도메인 원-핫이 특성에 새지 않는다 — 도메인 상실의 순수 효과가 드러난다.
(합성 A~E는 features에 도메인 원-핫이 있어 인코더가 부분 보상 → 비용을 과소평가할 수 있음.
그 confound 자체를 표에 함께 드러낸다.)

사용법: python eval/probe_domain_free.py [--folds N] [--worlds irt,specialist,...,textworld]
"""
import sys, pathlib, json, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from src.harness import run_tier, tier_budgets
from src.router import LPBRouter
from src.verifier import fold_verifier_matrix
from run_phase2 import CFG, OUT

TIERS = ["fast", "balanced", "premium"]
W = {t: CFG["tiers"][t]["weight"] for t in CFG["tiers"]}
SYNTH = ["irt", "specialist", "corr", "crossing", "nosignal"]


def _composite(qs_by_tier):
    return sum(W[t] * float(np.mean(qs_by_tier[t])) for t in TIERS)


def _run_synth(which):
    """합성 5세계(irt/specialist/corr/crossing/nosignal): run_phase7의 정식 기계
    (build_world + eval_router, 5-fold)에서 use_domain True/False 비교 — 정준 매트릭스와
    동일 데이터·동일 fold를 쓴다."""
    from run_phase7 import build_world, eval_router
    out = {}
    for wname in [w for w in SYNTH if w in which]:
        base, eds, folds = build_world(wname)
        modes = {}
        for use_dom in (True, False):
            qbt = {}
            for tier in TIERS:
                q, _sd, _cats, _oa = eval_router(eds, folds, tier,
                                                 router_kwargs={"use_domain": use_dom})
                qbt[tier] = [q]                         # eval_router가 이미 fold 평균
            modes[use_dom] = qbt
        out[wname] = modes
    return out


def _run_textworld(n_folds):
    """World F: 특성=텍스트 인코더 출력(도메인 원-핫 없음) — 가장 현실적."""
    from run_phase8 import build_f
    tw, ds, meta, F, folds = build_f()
    modes = {}
    for use_dom in (True, False):
        qbt = {t: [] for t in TIERS}
        for tier in TIERS:
            for f in range(n_folds):
                te = folds[f]
                tr = np.setdiff1d(np.arange(ds.n), te)
                V, _ = fold_verifier_matrix(F, ds.quality, tr)
                ds.verifier = V
                router = LPBRouter(CFG, len(np.unique(ds.domains)), seed=f,
                                   use_domain=use_dom).fit(ds, tr, tier)
                b_te = tier_budgets(ds, te, CFG)[tier]
                r = run_tier(ds, te, router.policy(), tier, b_te)
                qbt[tier].append(r.mean_quality)
        modes[use_dom] = qbt
    return {"textworld": modes}


def main(n_folds=5, which=None):
    which = which or (SYNTH + ["textworld"])
    t0 = time.time()
    results = {}
    if any(w in SYNTH for w in which):
        results.update(_run_synth(which))              # 합성은 항상 5-fold (eval_router)
    if "textworld" in which:
        results.update(_run_textworld(n_folds))

    print(f"\n[probe_domain_free] 단일집단 퇴화 비용 (n_folds={n_folds}, "
          f"{time.time() - t0:.0f}s)")
    print(f"  {'world':<12}{'multigroup':<13}{'single-group':<14}{'Δ(단일−다집단)':<16}"
          f"{'tier별 Δ (fast/bal/prem)':<28}")
    summary = {}
    for wname in which:
        if wname not in results:
            continue
        m = results[wname]
        c_true = _composite(m[True])
        c_false = _composite(m[False])
        tier_d = [float(np.mean(m[False][t]) - np.mean(m[True][t])) for t in TIERS]
        summary[wname] = {"multigroup": round(c_true, 4),
                          "single_group": round(c_false, 4),
                          "delta": round(c_false - c_true, 4),
                          "tier_delta": [round(x, 4) for x in tier_d]}
        td = "  ".join(f"{x:+.4f}" for x in tier_d)
        print(f"  {wname:<12}{c_true:<13.4f}{c_false:<14.4f}{c_false - c_true:<+16.4f}{td}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"n_folds": n_folds, "summary": summary},
              open(OUT / "probe_domain_free.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"  → eval/results/probe_domain_free.json")
    return summary


if __name__ == "__main__":
    nf = 5
    which = None
    if "--folds" in sys.argv:
        nf = int(sys.argv[sys.argv.index("--folds") + 1])
    if "--worlds" in sys.argv:
        which = sys.argv[sys.argv.index("--worlds") + 1].split(",")
    main(nf, which)
