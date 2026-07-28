"""예측기 개편 측정 (Phase 19) — 특성 축과 용량 축을 각각 재고 조합한다.

근거: `probe_predictor_ceiling.json` — 남은 격차 0.1314 중 **82.8% 가 예측(p̄) 몫**이고
결정 구조는 3.6% 다. 그래서 이 프로브가 이 프로젝트의 남은 최대 레버다.

두 축
  ① 특성: `text_encoder.HashingEncoder`(문자 3-gram 96차 + 수치 4) → `features.RichEncoder`
          (문자 3·4-gram + 단어 1·2-gram + idf + 구조 20종)
  ② 용량: 단일 잠재 θ 는 "모델 X 는 이 종류에 강하다"를 **표현할 수 없다**.
          `residual` = 모델별 로짓 잔차 r_m(x) · `hidden` = μ·σ 헤드에 tanh 은닉층
          · `cluster` = 무감독 군집 pseudo-domain (다집단 b_{d,m})

두 지표를 같이 본다 — 예측 품질(held-out log-loss·AUC)이 좋아졌는지, 그리고 그것이 실제
**라우팅 점수**로 이어지는지. Phase 2 는 이 둘이 어긋난 사례를 기록했다(예측이 좋아도 점수가
안 오름). 채택은 항상 라우팅 점수 + 쌍대 표준오차로 판정한다 (D21).

사용법: python eval/probe_predictor.py [--folds 3] [--n 1200] [--quick]
"""
import sys, pathlib, json, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from src.textworld import build_textworld, to_dataset
from src.text_encoder import get_encoder as legacy_encoder
from src.features import RichEncoder
from src.verifier import augmented_feature_matrix, fold_verifier_matrix
from src.harness import run_tier, tier_budgets
from src.router import LPBRouter

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config_ax3.yaml", encoding="utf-8"))
OUT = ROOT / "eval" / "results"
TIERS = ["fast", "balanced", "premium"]
W = {t: CFG["tiers"][t]["weight"] for t in CFG["tiers"]}


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def auc(s, y):
    s, y = np.asarray(s).ravel(), (np.asarray(y).ravel() > 0.5).astype(float)
    pos, neg = s[y > 0.5], s[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# (이름, 특성 백엔드, LPBRouter kwargs)
ARMS = [
    ("기준: hashing + 선형헤드", "hashing", {}),
    ("① rich 특성 + 선형헤드", "rich", {}),
    ("② hashing + 잔차헤드", "hashing", {"enc_residual": True}),
    ("①+② rich + 잔차헤드", "rich", {"enc_residual": True}),
    ("①+② + 은닉층(32)", "rich", {"enc_residual": True, "enc_hidden": 32}),
    ("①+② + 군집 pseudo-domain(6)", "rich", {"enc_residual": True,
                                             "use_domain": "cluster", "n_clusters": 6}),
]


def main(nf=3, n_queries=1200, quick=False):
    t0 = time.time()
    cfg = dict(CFG, synth=dict(CFG["synth"], n_queries=n_queries))
    tw = build_textworld(cfg, seed=42)
    ds, meta = to_dataset(tw, cfg, Ns=(1,))
    F = augmented_feature_matrix(meta, text_dim=64, use_prompt=True, use_agreement=True)
    folds = ds.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])
    prompts = meta["prompts"]
    print(f"[probe_predictor] n={ds.n} M={ds.m} folds={nf}", flush=True)

    feats = {}
    feats["hashing"] = legacy_encoder("hashing").encode(prompts)
    re_ = RichEncoder().fit(prompts)                      # idf 는 비지도 → 누수 없음
    feats["rich"] = re_.encode(prompts)
    print(f"  특성 차원: hashing {feats['hashing'].shape[1]} · rich {feats['rich'].shape[1]}"
          f"  ({time.time() - t0:.0f}s)", flush=True)

    arms = ARMS[:4] if quick else ARMS
    res = {}
    base_pq = {}
    print(f"\n  {'구성':<32}{'logloss':<10}{'AUC':<8}{'fast':<9}{'bal':<9}{'prem':<9}"
          f"{'종합':<9}{'Δ종합(SE)':<14}")
    for name, backend, kw in arms:
        ds.features = feats[backend]
        per, lls, aucs = {}, [], []
        pq_all = {}
        for tier in TIERS:
            qs, pqs = [], []
            for f in range(nf):
                te = folds[f]
                tr = np.setdiff1d(np.arange(ds.n), te)
                ds.verifier, _ = fold_verifier_matrix(F, ds.quality, tr)
                use_dom = kw.get("use_domain", False)
                r = LPBRouter(CFG, 1, seed=f,
                              **{**kw, "use_domain": use_dom}).fit(ds, tr, tier)
                out = run_tier(ds, te, r.policy(), tier, tier_budgets(ds, te, CFG)[tier])
                qs.append(out.mean_quality)
                pqs.append(np.asarray(out.per_query, dtype=float))
                if tier == "fast":                     # 예측 품질은 한 번만 재면 된다
                    dom_te = r._doms[te] if r.clusterer is not None else ds.domains[te]
                    p = r.enc.predict(ds.features[te], dom_te)
                    lls.append(logloss(p, ds.quality[te]))
                    aucs.append(auc(p, ds.quality[te]))
            per[tier] = float(np.mean(qs))
            pq_all[tier] = np.concatenate(pqs)
        comp = sum(W[t] * per[t] for t in TIERS)
        if not base_pq:
            base_pq, se_txt = pq_all, "—"
        else:
            d = np.concatenate([W[t] * (pq_all[t] - base_pq[t]) for t in TIERS])
            se = float(d.std(ddof=1) / np.sqrt(len(d))) * len(TIERS) ** 0.5
            se_txt = f"{comp - res[arms[0][0]]['composite']:+.4f}({se:.4f})"
        res[name] = {"logloss": round(float(np.mean(lls)), 4),
                     "auc": round(float(np.mean(aucs)), 4),
                     "per_tier": {k: round(v, 4) for k, v in per.items()},
                     "composite": round(comp, 4), "delta_se": se_txt,
                     "backend": backend, "kwargs": {k: str(v) for k, v in kw.items()}}
        print(f"  {name:<32}{np.mean(lls):<10.4f}{np.mean(aucs):<8.4f}{per['fast']:<9.4f}"
              f"{per['balanced']:<9.4f}{per['premium']:<9.4f}{comp:<9.4f}{se_txt:<14}",
              flush=True)

    base = res[arms[0][0]]["composite"]
    best = max(res, key=lambda k: res[k]["composite"])
    res["_verdict"] = {"baseline": base, "best_arm": best,
                       "best_composite": res[best]["composite"],
                       "delta": round(res[best]["composite"] - base, 4),
                       "pbar_oracle_reference": 0.6140,
                       "captured_of_prediction_headroom":
                           round((res[best]["composite"] - base) / (0.6140 - base) * 100, 1)}
    print(f"\n  최고: {best}  종합 {res[best]['composite']} (기준 대비 "
          f"{res['_verdict']['delta']:+.4f})")
    print(f"  p̄ 오라클(0.6140) 대비 예측 여지의 "
          f"{res['_verdict']['captured_of_prediction_headroom']}% 회수")
    OUT.mkdir(parents=True, exist_ok=True)
    res["_note"] = ("Phase 19. A.X 3모델·Q2=No·textworld·증강 검증기. 채택은 종합 Δ가 "
                    "쌍대 SE 를 넘을 때만 (D21).")
    json.dump(res, open(OUT / "probe_predictor.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"  ({time.time() - t0:.0f}s) → eval/results/probe_predictor.json")
    return res


if __name__ == "__main__":
    nf = int(sys.argv[sys.argv.index("--folds") + 1]) if "--folds" in sys.argv else 3
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 1200
    main(nf, n, "--quick" in sys.argv)
