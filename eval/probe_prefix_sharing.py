"""prefix-sharing 비용 토글의 prize 측정 (Phase 13 — 논문 레버 검토 산물).

레버: KV-cache 프리픽스 공유가 사실이면 [모델×N표] 상자의 프롬프트 프리필을 N회가 아니라
1회만 과금한다(`cost.prefix_sharing`). 적대적 검증 결론: 이득은 조건부이고, host가 실제로
N-prefill을 청구하면 정확히 0(정책이 어떤 비용에도 λ로 적응하므로 "남는 것"이 없다).
게다가 tier 예산 = frac×(최대 상자 비용)이라 공유가 켜지면 예산도 함께 줄어 상쇄가 있다.
→ 산술 상한이 아니라 **실측**으로 prize를 확정한다.

이 프로브는 prefix_sharing OFF(현행 배포 기본) vs ON에서 6세계 종합점수를 비교한다.
양쪽 모두 자기 비용 체제에서 λ를 재튜닝하므로, 차이 = "공유가 사실일 때 더 싼 투표가
푸는 순품질". Q3 확정 전까지 기본은 OFF이며, 이 수치는 Q3 확인을 우선순위화할지 판단용.

사용법: python eval/probe_prefix_sharing.py [--worlds irt,...,textworld]
"""
import sys, pathlib, json, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from src.harness import run_tier, tier_budgets
from src.router import LPBRouter
from src.verifier import feature_matrix, fold_verifier_matrix
from run_phase2 import CFG, OUT

TIERS = ["fast", "balanced", "premium"]
W = {t: CFG["tiers"][t]["weight"] for t in CFG["tiers"]}
SYNTH = ["irt", "specialist", "corr", "crossing", "nosignal"]


def _set_sharing(flag: bool):
    CFG.setdefault("cost", {})["prefix_sharing"] = flag


def _composite(qbt):
    return sum(W[t] * float(np.mean(qbt[t])) for t in TIERS)


def _run_synth(which):
    """합성 5세계: build_world가 현재 CFG의 prefix_sharing으로 비용을 굽는다 (캐시 없음)."""
    from run_phase7 import build_world, eval_router
    out = {}
    for wname in [w for w in SYNTH if w in which]:
        modes = {}
        for flag in (False, True):
            _set_sharing(flag)
            base, eds, folds = build_world(wname)         # 현재 flag로 재구축
            qbt = {t: [eval_router(eds, folds, t)[0]] for t in TIERS}
            modes[flag] = qbt
        out[wname] = modes
    return out


def _run_textworld():
    """World F: build_f 캐시를 우회해 flag마다 재구축 (가격이 to_dataset 시점에 결정됨)."""
    from src.textworld import build_textworld, to_dataset
    from src.text_encoder import get_encoder
    modes = {}
    for flag in (False, True):
        _set_sharing(flag)
        tw = build_textworld(CFG, seed=42)
        ds, meta = to_dataset(tw, CFG)                    # 현재 flag로 vote-box 가격 결정
        ds.features = get_encoder("hashing").encode(meta["prompts"])
        F = feature_matrix(meta)
        folds = ds.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])
        qbt = {t: [] for t in TIERS}
        for tier in TIERS:
            for f, te in enumerate(folds):
                tr = np.setdiff1d(np.arange(ds.n), te)
                V, _ = fold_verifier_matrix(F, ds.quality, tr)
                ds.verifier = V
                router = LPBRouter(CFG, len(np.unique(ds.domains)), seed=f).fit(ds, tr, tier)
                b_te = tier_budgets(ds, te, CFG)[tier]
                qbt[tier].append(run_tier(ds, te, router.policy(), tier, b_te).mean_quality)
        modes[flag] = qbt
    return {"textworld": modes}


def main(which=None):
    which = which or (SYNTH + ["textworld"])
    t0 = time.time()
    results = {}
    try:
        if any(w in SYNTH for w in which):
            results.update(_run_synth(which))
        if "textworld" in which:
            results.update(_run_textworld())
    finally:
        _set_sharing(False)                               # 전역 CFG 원복 (기본 OFF)

    print(f"\n[probe_prefix_sharing] 공유 비용 prize (5-fold, {time.time() - t0:.0f}s)")
    print(f"  {'world':<12}{'OFF(현행)':<12}{'ON(공유)':<12}{'Δ(ON−OFF)':<14}"
          f"{'tier별 Δ (fast/bal/prem)':<28}")
    summary = {}
    for wname in which:
        if wname not in results:
            continue
        m = results[wname]
        c_off, c_on = _composite(m[False]), _composite(m[True])
        tier_d = [float(np.mean(m[True][t]) - np.mean(m[False][t])) for t in TIERS]
        summary[wname] = {"off": round(c_off, 4), "on": round(c_on, 4),
                          "delta": round(c_on - c_off, 4),
                          "tier_delta": [round(x, 4) for x in tier_d]}
        td = "  ".join(f"{x:+.4f}" for x in tier_d)
        print(f"  {wname:<12}{c_off:<12.4f}{c_on:<12.4f}{c_on - c_off:<+14.4f}{td}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"note": "prefix_sharing ON vs OFF 종합점수. Δ>0 = 공유가 사실일 때의 상한 prize "
                       "(Q3 확정 전 기본 OFF). 상세: docs/reflections/phase13.md",
               "summary": summary},
              open(OUT / "probe_prefix_sharing.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("  → eval/results/probe_prefix_sharing.json")
    return summary


if __name__ == "__main__":
    which = None
    if "--worlds" in sys.argv:
        which = sys.argv[sys.argv.index("--worlds") + 1].split(",")
    main(which)
