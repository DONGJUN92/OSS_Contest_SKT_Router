"""동적 교차모델 일치 검증기 측정 (Phase 19) — 남은 유일한 실질 레버.

근거 (`eval/results/probe_predictor_ceiling.json`):
  하한 0.3499 → 현행 0.5051 → **달성 가능 상한 ⓑ 0.5725** (진짜확률 p̄ + 완벽 채점)
  남은 달성 가능 여지 +0.0674 의 분해:  검증기 **+0.0593 (88%)** · 결정구조 +0.0047 · 예측기 −0.0010
  그리고 그 검증기 여지는 tier 별로 fast +0.003 / balanced +0.090 / premium +0.155 —
  **2회 이상 여는 tier 에 전부 몰려 있다.**

정적 검증기 행렬은 일치 신호를 "기준 상자(ref_col) 대비"로 고정해 이 여지에 닿지 못한다.
이 프로브는 `Session.observe_fn` 훅으로 **호출 이력 안에서** 일치율을 재계산하는 경로를 켜고
그 값을 잰다. arm 3종:

  A. 정적 (agree_ref)            — 현행
  B. 정적 (대칭 일치율 agree_frac) — 학습 통계만 바꾼 대조군 (배포 불가, 참조)
  C. **동적** (호출분 일치율)      — 배포 가능한 형태

C > A 이고 쌍대 SE 를 넘으면 채택한다. B 는 "정보가 있는데 정적 계약이 막고 있었나"를 가른다.

사용법: python eval/probe_dynamic_verifier.py [--folds 3] [--n 1200]
"""
import sys, pathlib, json, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from src.textworld import build_textworld, to_dataset
from src.text_encoder import get_encoder
from src.verifier import (augmented_feature_matrix, augmented_feature_names,
                          fold_verifier_matrix, canonical_matrix, agreement_fraction,
                          DynamicAgreementVerifier, ScoringVerifier, TASKS)
from src.harness import run_tier, tier_budgets
from src.router import LPBRouter

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config_ax3.yaml", encoding="utf-8"))
OUT = ROOT / "eval" / "results"
TIERS = ["fast", "balanced", "premium"]
W = {t: CFG["tiers"][t]["weight"] for t in CFG["tiers"]}


def auc(s, y):
    s, y = np.asarray(s).ravel(), (np.asarray(y).ravel() > 0.5).astype(float)
    pos, neg = s[y > 0.5], s[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main(nf=3, n_queries=1200):
    t0 = time.time()
    cfg = dict(CFG, synth=dict(CFG["synth"], n_queries=n_queries))
    tw = build_textworld(cfg, seed=42)
    ds, meta = to_dataset(tw, cfg, Ns=(1,))
    ds.features = get_encoder("hashing").encode(meta["prompts"])
    F = augmented_feature_matrix(meta, text_dim=64, use_prompt=True, use_agreement=True)
    names = augmented_feature_names(TASKS, 64, ds.m, True, True)
    slot = names.index("agree_ref")
    canon = canonical_matrix(meta)
    A_frac = agreement_fraction(canon)
    F_sym = F.copy()
    F_sym[:, :, slot] = A_frac                       # 대칭 일치율로 슬롯 교체
    folds = ds.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])
    print(f"[probe_dynamic_verifier] n={ds.n} M={ds.m} folds={nf}  agree 슬롯={slot}", flush=True)
    print(f"  agree_ref 평균 {F[:, :, slot].mean():.3f} · 대칭 일치율 평균 {A_frac.mean():.3f}",
          flush=True)
    # ★ 사전 점검: 일치 신호가 이 세계에 **존재하는가**. 없으면 이 프로브는 레버를 못 잰다.
    # (textworld 실측: 대칭 일치율 0.003 — 모델들이 서로 같은 답을 내는 일이 거의 없다.
    #  RouterBench 에서는 같은 특성이 검증기 AUC 를 +0.178 올렸다. 세계가 다르면 레버도 다르다.)
    if A_frac.mean() < 0.02:
        print("  ⚠ 이 세계에는 교차모델 일치 신호가 사실상 없다 (평균 <2%) — "
              "동적 일치 레버는 여기서 측정 불가. 판정은 RouterBench 에서 해야 한다.", flush=True)

    res = {}
    base_pq = None
    print(f"\n  {'arm':<34}{'AUC':<8}{'fast':<9}{'bal':<9}{'prem':<9}{'종합':<9}{'Δ(SE)':<14}")
    for arm in ("A 정적 agree_ref (현행)", "B 정적 대칭일치율 (참조)", "C 동적 호출분일치율"):
        per, aucs, pqs_by_tier = {}, [], {}
        for tier in TIERS:
            qs, pqs = [], []
            for f in range(nf):
                te = folds[f]
                tr = np.setdiff1d(np.arange(ds.n), te)
                Fuse = F if arm.startswith("A") else F_sym
                V, sv = fold_verifier_matrix(Fuse, ds.quality, tr)
                ds.verifier = V
                if tier == "fast" and arm.startswith("A"):
                    aucs.append(auc(V[te], ds.quality[te]))
                elif tier == "fast":
                    aucs.append(auc(V[te], ds.quality[te]))
                r = LPBRouter(CFG, 1, seed=f, use_domain=False).fit(ds, tr, tier)
                dyn = None
                if arm.startswith("C"):
                    dv = DynamicAgreementVerifier(Fuse, slot, canon, sv)
                    dyn = dv.row_fn
                out = run_tier(ds, te, r.policy(), tier,
                               tier_budgets(ds, te, CFG)[tier], dyn_observe=dyn)
                qs.append(out.mean_quality)
                pqs.append(np.asarray(out.per_query, dtype=float))
            per[tier] = float(np.mean(qs))
            pqs_by_tier[tier] = np.concatenate(pqs)
        comp = sum(W[t] * per[t] for t in TIERS)
        if base_pq is None:
            base_pq, base_comp, se_txt = pqs_by_tier, comp, "—"
        else:
            d = np.concatenate([W[t] * (pqs_by_tier[t] - base_pq[t]) for t in TIERS])
            se = float(d.std(ddof=1) / np.sqrt(len(d))) * len(TIERS) ** 0.5
            se_txt = f"{comp - base_comp:+.4f}({se:.4f})"
        res[arm] = {"auc_fast": round(float(np.mean(aucs)), 4),
                    "per_tier": {k: round(v, 4) for k, v in per.items()},
                    "composite": round(comp, 4), "delta_se": se_txt}
        print(f"  {arm:<34}{np.mean(aucs):<8.4f}{per['fast']:<9.4f}{per['balanced']:<9.4f}"
              f"{per['premium']:<9.4f}{comp:<9.4f}{se_txt:<14}", flush=True)

    a, c = res["A 정적 agree_ref (현행)"]["composite"], res["C 동적 호출분일치율"]["composite"]
    res["_verdict"] = {
        "static": a, "dynamic": c, "delta": round(c - a, 4),
        "achievable_ceiling": 0.5725, "verifier_headroom": 0.0593,
        "captured_pct": round((c - a) / 0.0593 * 100, 1),
        "note": ("Phase 19. 채택은 쌍대 SE 초과 여부로 판정한다 (D21). "
                 "검증기 여지 +0.0593 중 몇 % 를 회수했는지가 이 프로브의 목적."),
    }
    print(f"\n  동적 − 정적 = {c - a:+.4f}  →  검증기 여지(+0.0593)의 "
          f"{res['_verdict']['captured_pct']}% 회수")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "probe_dynamic_verifier.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"  ({time.time() - t0:.0f}s) → eval/results/probe_dynamic_verifier.json")
    return res


if __name__ == "__main__":
    nf = int(sys.argv[sys.argv.index("--folds") + 1]) if "--folds" in sys.argv else 3
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 1200
    main(nf, n)
