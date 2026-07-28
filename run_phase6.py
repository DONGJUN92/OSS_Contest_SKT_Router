"""Phase 6 실행기: 통합 · Ablation · 재현성.

  6a: 최종 라우터(LPBRouter) end-to-end 점수표 + 결정론적 재현성 검증 (2회 실행 비교)
  6b: 구성요소 ablation — 풀스택에서 하나씩 제거한 종합 점수 기여도
      (효율화: λ0는 fold 0 보정셋에서 variant별 1회 튜닝 후 fold 공유 —
       페이싱의 λ0-둔감성(4b)을 근거로 하며, 그 자체가 강건성 재검증이 된다)

데이터: Phase 4 확장 세계 (Phase 1 보존). 사용법: python run_phase6.py {6a|6b}
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
from src.harness import run_tier, tier_budgets
from src.cost_mirror import cost_matrix
from src.engine.pandora import tune_lambda_quality, tune_lambda_replay
from src.engine.pacing import PacedPandora
from src.router import LPBRouter
from run_phase2 import load_worlds_and_folds, CFG, OUT
from run_phase3 import make_pandora, make_pandora_disc
from run_phase4 import ext_worlds, ext_fold_models
from phase2_stages import DiscLR

W = {t: CFG["tiers"][t]["weight"] for t in CFG["tiers"]}
TIERS = ["fast", "balanced", "premium"]


def final_replay(tag=""):
    """최종 라우터의 전체 리플레이 → {world: {tier: [fold quality...]}, cert}."""
    worlds, ext, folds = ext_worlds()
    out = {}
    for wname in worlds:
        eds = ext[wname]
        out[wname] = {"tiers": {}, "cert": []}
        for tier in TIERS:
            qs = []
            for f in range(CFG["eval"]["k_folds"]):
                test_idx = folds[wname][f]
                tr = np.setdiff1d(np.arange(eds.n), test_idx)
                router = LPBRouter(CFG, CFG["synth"]["n_domains"], seed=f).fit(eds, tr, tier)
                b_te = tier_budgets(eds, test_idx, CFG)[tier]
                r = run_tier(eds, test_idx, router.policy(), tier, b_te)
                qs.append(round(r.mean_quality, 6))
                if tier == "fast":
                    out[wname]["cert"].append(router.certificate)
            out[wname]["tiers"][tier] = qs
    return out


def combined(res_world):
    return sum(W[t] * float(np.mean(qs)) for t, qs in res_world["tiers"].items())


def stage_6a():
    r1 = final_replay("run1")
    r2 = final_replay("run2")
    json.dump(r1, open(OUT / "final_scores.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[6a] 최종 라우터 end-to-end (5-fold 평균) + 재현성")
    for wname in r1:
        print(f"  World: {wname}")
        for tier in TIERS:
            q1 = np.mean(r1[wname]["tiers"][tier])
            q2 = np.mean(r2[wname]["tiers"][tier])
            print(f"    {tier:<10} quality={q1:.4f}   재현 delta={abs(q1 - q2):.2e}")
        c1, c2 = combined(r1[wname]), combined(r2[wname])
        cert = r1[wname]["cert"][0]
        print(f"    종합점수={c1:.4f}  (재현 delta={abs(c1 - c2):.2e})")
        print(f"    M4 인증서(fast, fold0): 후회율≤{cert['risk_upper']:.4f} "
              f"@ {cert['confidence']:.0%} (n={cert['n_cal']})")


def _variant_policies(eds_or_ds, wname, f, folds, tier, variant, lam_cache, use_ext):
    """ablation variant별 (λ0 튜닝 fold0 공유) 정책 생성."""
    tr, test_idx, irt, enc, noise = ext_fold_models(eds_or_ds, wname, f, folds) \
        if use_ext else _base_models(eds_or_ds, wname, f, folds)
    if variant == "-irt(disc)":
        disc = DiscLR().fit(eds_or_ds.features[tr], eds_or_ds.domains[tr],
                            eds_or_ds.quality[tr])
        fac = lambda l: make_pandora_disc(disc, noise, l)
    else:
        fac = lambda l: make_pandora(irt, enc, noise, l)

    key = (wname, tier, variant)
    if key not in lam_cache:
        rng = np.random.default_rng(0)
        tr0 = np.setdiff1d(np.arange(eds_or_ds.n), folds[wname][0])
        sub = rng.choice(tr0, size=600, replace=False)
        b_sub = tier_budgets(eds_or_ds, sub, CFG)[tier]
        tuner = tune_lambda_replay if variant == "-qualityfirst" else tune_lambda_quality
        lam_cache[key] = tuner(fac, eds_or_ds, sub, b_sub)
    lam0 = lam_cache[key]
    if variant == "-pacing(static)":
        return fac(lam0)
    return PacedPandora(fac, lam0, name=variant)


_BASE_CACHE = {}


def _base_models(ds, wname, f, folds):
    """투표 상자 제거 variant용: 원본 5-모델 세계의 fold 모델."""
    from src.irt.mml import fit_mml
    from src.engine.pandora import NoiseModel
    from phase2_stages import IRTEncoder
    key = (wname, f)
    if key not in _BASE_CACHE:
        test_idx = folds[wname][f]
        tr = np.setdiff1d(np.arange(ds.n), test_idx)
        irt = fit_mml(ds.quality[tr], ds.domains[tr], CFG["synth"]["n_domains"],
                      per_domain_b=True)
        enc = IRTEncoder(irt["a"], irt["b"]).fit(ds.features[tr], ds.domains[tr],
                                                 ds.quality[tr])
        noise = NoiseModel().fit(ds.verifier[tr].ravel(), ds.quality[tr].ravel())
        _BASE_CACHE[key] = (tr, test_idx, irt, enc, noise)
    return _BASE_CACHE[key]


def stage_6b():
    worlds, ext, folds = ext_worlds()
    variants = ["full", "-pacing(static)", "-votes(5box)", "-irt(disc)", "-qualityfirst"]
    lam_cache = {}
    report = {}
    for wname in worlds:
        eds, ds = ext[wname], worlds[wname]
        scores = {}
        for variant in variants:
            data = ds if variant == "-votes(5box)" else eds
            vkey = "full" if variant == "-votes(5box)" else variant
            total = 0.0
            for tier in TIERS:
                qs = []
                for f in range(CFG["eval"]["k_folds"]):
                    pol = _variant_policies(data, wname, f, folds, tier,
                                            vkey if variant != "-votes(5box)" else "full+5box",
                                            lam_cache, use_ext=(variant != "-votes(5box)"))
                    test_idx = folds[wname][f]
                    b_te = tier_budgets(data, test_idx, CFG)[tier]
                    r = run_tier(data, test_idx, pol, tier, b_te)
                    qs.append(r.mean_quality)
                total += W[tier] * float(np.mean(qs))
            scores[variant] = round(total, 4)
        report[wname] = scores
    json.dump(report, open(OUT / "phase6b_ablation.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[6b] 구성요소 ablation — 종합 점수 (λ0 fold0 공유)")
    print(f"  {'variant':<20}" + "".join(f"{w:<14}" for w in report))
    for variant in variants:
        line = f"  {variant:<20}"
        for wname in report:
            full = report[wname]["full"]
            v = report[wname][variant]
            d = f" ({v - full:+.4f})" if variant != "full" else ""
            line += f"{v:.4f}{d:<8}"[:14].ljust(14)
        print(line)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "6a"
    {"6a": stage_6a, "6b": stage_6b}[stage]()
