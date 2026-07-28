"""λ 선택 문턱 `se_k` 스윕 (Phase 20) — D34 수정이 남긴 유일한 열린 손잡이를 측정으로 닫는다.

배경 (`docs/reflections/phase20.md` §2.5)
----------------------------------------
D34 수정은 λ 후보 비교의 **기준선**을 "직전 채택자"에서 "전역 최고"로 고정했다. 기준선 고정
자체는 논쟁 대상이 아니다 — 움직이는 기준선과 비교하면 유의하지 않은 손실이 누적돼 λ 가
격자 끝까지 밀린다. 그런데 수정 후 실측에서 **σ<0(fast) 이 0.58 → 0.84 로 올랐다.**

기전: 채택 조건은 `q − q* > −se_k·SE(쌍대차이)` 인데, **전역 최고와 멀리 떨어진 λ 는 두 정책의
결정이 많이 달라 쌍대 분산이 크다.** 즉 멀수록 SE 가 커서 문턱이 넓어지고, 큰 λ 가 통과한다.
`se_k=1.0` 은 Phase 18 이 "격자를 좁히면 문턱을 세워야 한다"는 논리로 고른 값이고 **측정으로
고른 값이 아니다.** 그래서 여기서 측정한다.

세 팔
  se_k=0 : 순수 argmax(품질) + 정확 동률 시 큰 λ. 문턱 없음 → 품질 최댓값을 직접 고른다.
  se_k=1 : 현행 기본값.
  se_k=2 : 더 관대 → 더 큰 λ (더 절약·더 강한 퇴화). 방향 확인용 대조.

판정 규칙 (D21)
  기본값(se_k=1)을 이기려면 **종합 쌍대 차이가 max(절대 0.005, 1×쌍대 SE)** 를 넘어야 한다.
  넘지 못하면 기본값을 유지한다 — 잡음을 승리로 오인하지 않는다.

사용법: python eval/probe_se_k.py [--folds 3] [--n 1200]
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
from src.engine.pandora import FINE_MULTS

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config_ax3.yaml", encoding="utf-8"))
OUT = ROOT / "eval" / "results"
TIERS = ["fast", "balanced", "premium"]
W = {t: CFG["tiers"][t]["weight"] for t in CFG["tiers"]}

#: 절대 문턱 — 이보다 작은 차이는 크기만으로 채택하지 않는다 (D21).
#
# ★ D58 (Phase 20): 초판은 `se_k` 만 3-fold·단일 시드로 쟀고, 이득 +0.0011 이 이 문턱에
# 미달해 "기각"으로 끝냈다. 그런데 **문턱 미달이 곧 '차이가 없다'는 뜻은 아니다** — 검정력이
# 부족하면 어떤 실제 차이도 문턱을 못 넘는다. 그래서 두 가지를 바꾼다:
#   ① 축을 2개로 (`se_k` × `calibrate_pbar`) — 두 레버가 상호작용하는지 본다
#   ② **다중 시드 × 5-fold** 로 검정력을 올린다 (유효 표본 n_seeds×n 배)
# 그래도 문턱을 못 넘으면 그때는 "검정력을 올려도 구분되지 않는다"고 말할 수 있다 —
# 그것이 "쟀는데 미달"과 다른, 실제로 닫힌 결론이다.
ABS_MARGIN = 0.005
BASE_ARM = "se_k=1 · Platt on (현행 기본값)"
#: (이름, se_k, calibrate_pbar)
GRID = [
    ("se_k=0 · Platt on", 0.0, True),
    (BASE_ARM, 1.0, True),
    ("se_k=2 · Platt on", 2.0, True),
    ("se_k=0 · Platt off", 0.0, False),
    ("se_k=1 · Platt off", 1.0, False),
    ("se_k=2 · Platt off", 2.0, False),
]
ARMS = {name: se_k for name, se_k, _ in GRID}          # 하위호환 (이름→se_k)


def main(nf=5, n_queries=1200, seeds=(0, 1, 2)):
    t0 = time.time()
    cfg = dict(CFG, synth=dict(CFG["synth"], n_queries=n_queries))
    tw = build_textworld(cfg, seed=42)
    ds, meta = to_dataset(tw, cfg, Ns=(1,))                 # 실전 조건 (Q2=No)
    ds.features = get_encoder("hashing").encode(meta["prompts"])
    ds.text_encoder = get_encoder("hashing")                # D36: 배포 경로 정합
    F = augmented_feature_matrix(meta, text_dim=64, use_prompt=True, use_agreement=True)
    folds = ds.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])
    print(f"[probe_se_k] n={ds.n} M={ds.m} folds={nf}  (A.X 3모델 · Q2=No · 증강 검증기)",
          flush=True)

    res = {"n": ds.n, "n_folds": nf, "seeds": list(seeds), "abs_margin": ABS_MARGIN,
           "grid": [[n, s, p] for n, s, p in GRID], "arms": {}}
    pq_store = {}
    print(f"  검정력: {nf}-fold × {len(seeds)} 시드 = 팔당 {nf * len(seeds)} 회 적합/tier\n")
    print(f"  {'구성':<24}{'fast':<9}{'bal':<9}{'prem':<9}{'종합':<9}"
          f"{'fast호출':<9}{'σ<0':<7}{'λ₀(fast)':<10}")
    for name, se_k, platt in GRID:
        per_tier, calls, sneg, lam0 = {}, {}, [], []
        pq_by_tier = {}
        for tier in TIERS:
            qs, cl, pqs = [], [], []
            for sd in seeds:                      # D58: 다중 시드로 검정력 확보
                for f in range(nf):
                    te = folds[f]
                    tr = np.setdiff1d(np.arange(ds.n), te)
                    ds.verifier, _ = fold_verifier_matrix(F, ds.quality, tr)
                    r = LPBRouter(CFG, 1, seed=1000 * sd + f, use_domain=False,
                                  lam_mults=FINE_MULTS, lam_se_k=se_k,
                                  calibrate_pbar=platt).fit(ds, tr, tier)
                    out = run_tier(ds, te, r.policy(), tier,
                                   tier_budgets(ds, te, CFG)[tier])
                    qs.append(out.mean_quality)
                    cl.append(out.calls_per_query)
                    pqs.append(np.asarray(out.per_query, dtype=float))
                    if tier == "fast":
                        sneg.append(r.diagnostics["sigma_neg_frac"])
                        lam0.append(r.lam0)
            per_tier[tier] = float(np.mean(qs))
            calls[tier] = float(np.mean(cl))
            pq_by_tier[tier] = np.concatenate(pqs)
        comp = sum(W[t] * per_tier[t] for t in TIERS)
        pq_store[name] = pq_by_tier
        res["arms"][name] = {
            "se_k": se_k, "calibrate_pbar": platt,
            "per_tier": {k: round(v, 4) for k, v in per_tier.items()},
            "composite": round(comp, 4),
            "calls": {k: round(v, 3) for k, v in calls.items()},
            "sigma_neg_fast": round(float(np.mean(sneg)), 4),
            "lam0_fast_mean": round(float(np.mean(lam0)), 4),
        }
        print(f"  {name:<24}{per_tier['fast']:<9.4f}{per_tier['balanced']:<9.4f}"
              f"{per_tier['premium']:<9.4f}{comp:<9.4f}{calls['fast']:<9.2f}"
              f"{np.mean(sneg):<7.2f}{np.mean(lam0):<10.4g}", flush=True)

    # ── 기본값 대비 쌍대 비교 ────────────────────────────────────────────────
    # tier 별 쌍대 차이는 같은 fold·같은 쿼리 순서라 유효하다. 종합 차이의 SE 는 tier 를
    # 독립으로 보고 w_t² 로 합성한다 (같은 쿼리를 쓰므로 근사이며, 그 사실을 기록한다).
    base = pq_store[BASE_ARM]
    print(f"\n  {'구성':<24}{'종합Δ':<11}{'쌍대SE':<10}{'요구문턱':<10}{'판정':<8}")
    for name in ARMS:
        if name == BASE_ARM:
            print(f"  {name:<24}{'—':<11}{'—':<10}{'—':<10}{'기준':<8}")
            continue
        d_mean, var = 0.0, 0.0
        per_tier_delta = {}
        for tier in TIERS:
            d = pq_store[name][tier] - base[tier]
            se_t = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
            d_mean += W[tier] * float(d.mean())
            var += (W[tier] ** 2) * (se_t ** 2)
            per_tier_delta[tier] = {"delta": round(float(d.mean()), 4), "se": round(se_t, 4)}
        se = float(np.sqrt(var))
        need = max(ABS_MARGIN, se)
        win = d_mean > need
        res["arms"][name]["vs_base"] = {
            "composite_delta": round(d_mean, 4), "composite_se": round(se, 4),
            "required": round(need, 4), "beats_base": bool(win),
            "per_tier": per_tier_delta,
        }
        print(f"  {name:<24}{d_mean:<+11.4f}{se:<10.4f}{need:<10.4f}"
              f"{'채택' if win else '기각':<8}")

    winners = [n for n in ARMS if n != BASE_ARM
               and res["arms"][n].get("vs_base", {}).get("beats_base")]
    if winners:
        pick = max(winners, key=lambda n: res["arms"][n]["composite"])
    else:
        pick = BASE_ARM
    # ★ D58: 문턱 미달일 때 **"차이가 없다"** 와 **"검정력이 없다"** 를 구분해 기록한다.
    # 최대 관측 격차와 그때의 SE 를 비교해, 이 검정력에서 탐지 가능한 최소 효과(≈1×SE)가
    # 절대 문턱보다 작으면 "충분한 검정력에서 구분되지 않음" 이라고 말할 수 있다.
    gaps = [(n, res["arms"][n]["vs_base"]["composite_delta"],
             res["arms"][n]["vs_base"]["composite_se"]) for n in res["arms"] if n != BASE_ARM]
    max_se = max(s for _, _, s in gaps)
    powered = max_se < ABS_MARGIN            # SE 가 절대 문턱보다 작으면 검정력 충분
    res["verdict"] = {
        "recommended_arm": pick,
        "recommended_se_k": ARMS[pick],
        "recommended_calibrate_pbar": dict((n, p) for n, _, p in GRID)[pick],
        "changed_default": pick != BASE_ARM,
        "n_fits_per_arm_per_tier": nf * len(seeds),
        "max_paired_se": round(max_se, 4),
        "adequately_powered": bool(powered),
        "rule": (f"기본값을 이기려면 종합 쌍대 차이 > max({ABS_MARGIN}, 1×쌍대 SE) — "
                 f"넘지 못하면 기본값 유지 (D21)."),
        "conclusion": (
            "충분한 검정력에서도 어떤 팔도 문턱을 넘지 못했다 → 기본값이 최적과 구분되지 "
            "않는다 (닫힌 결론)." if powered and pick == BASE_ARM else
            "검정력 부족: 최대 쌍대 SE 가 절대 문턱보다 크므로 '미달'을 '차이 없음'으로 "
            "읽으면 안 된다." if not powered and pick == BASE_ARM else
            f"'{pick}' 가 문턱을 넘었다 → 기본값을 바꾼다."),
        "caveat": ("종합 SE 는 tier 를 독립으로 보고 w² 합성한 근사다 (세 tier 가 같은 쿼리를 "
                   "쓰므로 실제로는 상관이 있다). tier 별 Δ·SE 를 함께 기록했다."),
    }
    print(f"\n  → 권고: **{pick}**, 기본값 변경 {'YES' if pick != BASE_ARM else 'NO'}")
    print(f"     최대 쌍대 SE {max_se:.4f} vs 절대 문턱 {ABS_MARGIN} → "
          f"검정력 {'충분' if powered else '부족'}")
    print(f"     {res['verdict']['conclusion']}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "probe_se_k.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"  ({time.time() - t0:.0f}s) → eval/results/probe_se_k.json")
    return res


if __name__ == "__main__":
    nf = int(sys.argv[sys.argv.index("--folds") + 1]) if "--folds" in sys.argv else 5
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 1200
    sd = (tuple(int(x) for x in sys.argv[sys.argv.index("--seeds") + 1].split(","))
          if "--seeds" in sys.argv else (0, 1, 2))
    main(nf, n, sd)
