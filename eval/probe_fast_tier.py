"""fast tier 전용 설계 실측 (Phase 18) — 가중치 0.5 구간을 정면으로 다룬다.

배경: Phase 17 이 확인한 구조적 사실 — 빠듯한 예산에서 λ 가 커지면 모든 예약값이 음수가 되고,
정지 조건이 첫 관측 직후 성립해 정책이 **1회 호출**로 퇴화한다(호출/쿼리 1.01, σ<0 비율 0.70).
이건 버그가 아니라 최적 정책의 귀결이므로 "고칠" 대상이 아니다. 고칠 수 있는 것은
**그 1회를 더 잘 고르는 것**뿐이고, 1회 선택은 argmax(p̄_m − λ·c_m) 이므로 레버는 두 개다.

  ① λ 격자      : 현행은 lam_min×{1,2,4,8,16} (2배 간격). 1회 선택의 경계가 λ 에 직접 걸리는데
                   2배 간격은 그 경계를 성기게 훑는다 → √2 간격 + 1 미만(0.5·0.71) 포함.
                   격자를 좁히면 후보가 늘어 잡음 최댓값을 고를 위험이 커지므로 **쌍대 표준오차
                   문턱**(se_k)을 함께 건다.
  ② p̄ 재보정    : 선택은 모델 간 **상대** p̄ 로 결정되므로 모델별 오보정이 그대로 오선택이 된다.
                   보정셋으로 모델별 Platt(s_m, t_m) 을 적합해 ICC 를 다시 맞춘다.

측정은 fast tier 를 주로 보되, 다른 tier 회귀가 없는지 함께 확인한다 (레버가 fast 만 건드리는지).
사용법: python eval/probe_fast_tier.py [--folds 3] [--n 1200]
"""
import sys, pathlib, json, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from src.textworld import build_textworld, to_dataset
from src.text_encoder import get_encoder
from src.verifier import augmented_feature_matrix, fold_verifier_matrix
from src.harness import run_tier, tier_budgets
from src.router import LPBRouter
from src.engine.pandora import FINE_MULTS, COARSE_MULTS

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config_ax3.yaml", encoding="utf-8"))
OUT = ROOT / "eval" / "results"
TIERS = ["fast", "balanced", "premium"]
W = {t: CFG["tiers"][t]["weight"] for t in CFG["tiers"]}

# ★ D55 (Phase 20 수정): 초판의 기준 팔은 `{}` — 즉 **LPBRouter 기본값**이었다. 그런데
# Phase 18 이 이 프로브의 결론을 받아 두 레버를 **기본값으로 승격**시켰으므로(현재 기본값은
# `lam_mults=None→FINE_MULTS`, `lam_se_k=1.0`, `calibrate_pbar=True`), 그 순간부터 이 프로브는
# **기본값을 기본값과 비교**하게 됐다. 실측 결과 다섯 팔이 전부 동일하고 Δ가 +0.0000 이다.
# 즉 §2.5 가 채택 근거로 인용한 "+0.0083 (2.1×SE)" 는 **인용된 그 스크립트로 재현되지 않는다.**
# D32("기구가 발동했는지 증명하라")·D51("테스트 픽스처가 부재를 은폐한다")와 같은 부류다.
# 수정: 각 팔이 레버를 **명시적으로 고정**한다. 기준 팔은 Phase 17 상태(레버 이전)다.
_PRE = {"lam_mults": COARSE_MULTS, "lam_se_k": 0.0, "calibrate_pbar": False}
ARMS = {
    "기준 (Phase 17 상태)": dict(_PRE),
    "① λ 세밀격자 (SE 문턱 없이)": {**_PRE, "lam_mults": FINE_MULTS},
    "① λ 세밀격자 + SE 문턱": {**_PRE, "lam_mults": FINE_MULTS, "lam_se_k": 1.0},
    "② p̄ Platt 재보정": {**_PRE, "calibrate_pbar": True},
    "①+② 결합 (= 현행 기본값)": {"lam_mults": FINE_MULTS, "lam_se_k": 1.0,
                                "calibrate_pbar": True},
}


def main(nf=3, n_queries=1200):
    t0 = time.time()
    cfg = dict(CFG, synth=dict(CFG["synth"], n_queries=n_queries))
    tw = build_textworld(cfg, seed=42)
    ds, meta = to_dataset(tw, cfg, Ns=(1,))                 # 실전 조건 (Q2=No)
    ds.features = get_encoder("hashing").encode(meta["prompts"])
    F = augmented_feature_matrix(meta, text_dim=64, use_prompt=True, use_agreement=True)
    folds = ds.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])
    print(f"[probe_fast_tier] n={ds.n} M={ds.m} folds={nf}", flush=True)

    res = {"n": ds.n, "n_folds": nf, "arms": {}}
    base_pq = {}
    print(f"\n  {'구성':<28}{'fast':<9}{'bal':<9}{'prem':<9}{'종합':<9}"
          f"{'fast호출':<9}{'σ<0':<7}{'fastΔ(SE)':<12}")
    for name, kw in ARMS.items():
        per_tier, calls, sneg = {}, {}, []
        pq_by_tier = {}
        for tier in TIERS:
            qs, cl, pqs = [], [], []
            for f in range(nf):
                te = folds[f]
                tr = np.setdiff1d(np.arange(ds.n), te)
                ds.verifier, _ = fold_verifier_matrix(F, ds.quality, tr)
                r = LPBRouter(CFG, 1, seed=f, use_domain=False, **kw).fit(ds, tr, tier)
                out = run_tier(ds, te, r.policy(), tier, tier_budgets(ds, te, CFG)[tier])
                qs.append(out.mean_quality)
                cl.append(out.calls_per_query)
                pqs.append(np.asarray(out.per_query, dtype=float))
                if tier == "fast":
                    sneg.append(r.diagnostics["sigma_neg_frac"])
            per_tier[tier] = float(np.mean(qs))
            calls[tier] = float(np.mean(cl))
            pq_by_tier[tier] = np.concatenate(pqs)
        comp = sum(W[t] * per_tier[t] for t in TIERS)
        # fast tier 의 기준 대비 쌍대 표준오차 (동일 fold·동일 쿼리 순서라 쌍대 비교가 유효)
        if not base_pq:
            base_pq = pq_by_tier
            se_txt = "—"
        else:
            d = pq_by_tier["fast"] - base_pq["fast"]
            se = float(d.std(ddof=1) / np.sqrt(len(d)))
            se_txt = f"{d.mean():+.4f}({se:.4f})"
        res["arms"][name] = {"per_tier": {k: round(v, 4) for k, v in per_tier.items()},
                             "composite": round(comp, 4),
                             "calls": {k: round(v, 3) for k, v in calls.items()},
                             "sigma_neg_fast": round(float(np.mean(sneg)), 4),
                             "fast_delta_vs_base": se_txt}
        print(f"  {name:<28}{per_tier['fast']:<9.4f}{per_tier['balanced']:<9.4f}"
              f"{per_tier['premium']:<9.4f}{comp:<9.4f}{calls['fast']:<9.2f}"
              f"{np.mean(sneg):<7.2f}{se_txt:<12}", flush=True)

    # D55: 기준 팔 이름을 하드코딩하지 않는다 — ARMS 의 **첫 항목**이 기준이다.
    # (이름을 바꿨을 때 여기서 KeyError 로 죽어 아티팩트가 아예 안 써졌다.)
    base_name = next(iter(ARMS))
    base = res["arms"][base_name]
    best = max(res["arms"], key=lambda k: res["arms"][k]["composite"])
    res["verdict"] = {
        "baseline_arm": base_name,
        "baseline_composite": base["composite"],
        "best_arm": best, "best_composite": res["arms"][best]["composite"],
        "delta": round(res["arms"][best]["composite"] - base["composite"], 4),
        "note": ("채택 기준: fast tier 쌍대 차이가 표준오차를 넘어야 한다. "
                 "넘지 못하면 기본값 유지 (D21)."),
    }
    print(f"\n  최고: {best} (종합 {res['arms'][best]['composite']}, "
          f"기준 대비 {res['verdict']['delta']:+.4f})")
    print(f"  → 채택 판정은 fast Δ(SE) 열이 결정한다: 차이 ≤ SE 면 기본값 유지")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "probe_fast_tier.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"  ({time.time() - t0:.0f}s) → eval/results/probe_fast_tier.json")
    return res


if __name__ == "__main__":
    nf = int(sys.argv[sys.argv.index("--folds") + 1]) if "--folds" in sys.argv else 3
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 1200
    main(nf, n)
