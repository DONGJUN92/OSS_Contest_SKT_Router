"""인코더 축 A/B — **쌍대 표준오차까지** (Phase 21, D65 후속).

왜 별도 프로브인가
--------------------
`probe_redteam_closure.py --encoder …` 는 인코더별 종합 점수를 주지만 **쌍대 SE 가 없다.**
그런데 그 격차(+0.0054)가 D21 절대 문턱(0.005) 바로 위라, 점추정만으로 기본값을 바꾸는 것은
이 저장소가 D21·D26·D34 에서 반복해 처벌받은 실수 그대로다.

그래서 **같은 fold·같은 시드에서 인코더만 바꿔** 쿼리별 품질을 기록하고, 쌍대 차이의
표준오차를 낸다. 쌍대로 재는 이유는 같은 쿼리에 대한 두 구성의 차이가 쿼리 난이도 분산을
상쇄해 훨씬 예민하기 때문이다 (게이트 D26 과 같은 논리).

측정 대상은 **제출 대상**(`SelectiveRouter`)과 `LPBRouter` 둘 다다 — D56 의 교훈:
"측정 대상이 제출 대상이 아니면 그 측정은 제출물을 대변하지 못한다." 그리고 둘의 부호가
다를 수 있다(실제로 달랐다)는 것이 이 실험의 핵심 정보다.

사용법: python eval/probe_encoder_axis.py [--folds 3] [--n 1200] [--encoders hashing:96,hashing:512]
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
from baselines.policies import SelectiveRouter

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config_ax3.yaml", encoding="utf-8"))
OUT = ROOT / "eval" / "results"
TIERS = ["fast", "balanced", "premium"]
W = {t: CFG["tiers"][t]["weight"] for t in CFG["tiers"]}


def main(nf=3, n_queries=1200, encoders=("hashing:96", "hashing:512")):
    t0 = time.time()
    cfg = dict(CFG, synth=dict(CFG["synth"], n_queries=n_queries))
    tw = build_textworld(cfg, seed=42)
    ds, meta = to_dataset(tw, cfg, Ns=(1,))
    F = augmented_feature_matrix(meta, text_dim=64, use_prompt=True,
                                 use_agreement=True, ref_col=0)
    folds = ds.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])
    feats = {e: get_encoder(e).encode(meta["prompts"]) for e in encoders}
    base = encoders[0]
    print(f"[probe_encoder_axis] n={ds.n} folds={nf} 인코더={list(encoders)} "
          f"(기준 {base})", flush=True)

    # per_query[enc][policy][tier] = fold 를 이어 붙인 쿼리별 품질
    pq = {e: {p: {t: [] for t in TIERS} for p in ("lpb", "selective3")} for e in encoders}
    for tier in TIERS:
        for f in range(nf):
            te = folds[f]
            tr = np.setdiff1d(np.arange(ds.n), te)
            V, _ = fold_verifier_matrix(F, ds.quality, tr)
            ds.verifier = V
            b_te = tier_budgets(ds, te, CFG)[tier]
            for e in encoders:
                ds.features = feats[e]
                r = LPBRouter(CFG, 1, seed=f, use_domain=False).fit(ds, tr, tier)
                pq[e]["lpb"][tier].append(
                    np.asarray(run_tier(ds, te, r.policy(), tier, b_te).per_query, float))
                sr = SelectiveRouter(CFG, 1, seed=f, use_domain=False).fit(ds, tr, tier)
                pq[e]["selective3"][tier].append(
                    np.asarray(run_tier(ds, te, sr.policy(), tier, b_te).per_query, float))
            print(f"  {tier:<9} fold {f + 1}/{nf}  ({time.time() - t0:.0f}s)", flush=True)

    res = {"base": base, "encoders": list(encoders), "folds": nf, "n_queries": n_queries,
           "policies": {}}
    for pol in ("lpb", "selective3"):
        res["policies"][pol] = {}
        for e in encoders:
            row = {"per_tier": {}, "composite": 0.0}
            for t in TIERS:
                v = np.concatenate(pq[e][pol][t])
                row["per_tier"][t] = round(float(v.mean()), 4)
                row["composite"] += W[t] * float(v.mean())
            row["composite"] = round(row["composite"], 4)
            if e != base:
                # ★ 쌍대: 같은 fold·같은 쿼리 순서이므로 그대로 뺀다. tier 가중 합성의 SE 는
                # tier 를 독립으로 보고 w² 합성한다 (§2.6 과 같은 근사 — 한계도 같이 적는다).
                var, deltas = 0.0, {}
                for t in TIERS:
                    d = np.concatenate(pq[e][pol][t]) - np.concatenate(pq[base][pol][t])
                    se_t = float(d.std(ddof=1) / np.sqrt(len(d)))
                    deltas[t] = {"delta": round(float(d.mean()), 4),
                                 "paired_se": round(se_t, 4), "n": int(len(d))}
                    var += (W[t] * se_t) ** 2
                gap = row["composite"] - res["policies"][pol][base]["composite"]
                se = float(np.sqrt(var))
                need = max(0.005, se)                     # D21 문턱
                row["delta_vs_base"] = round(gap, 4)
                row["composite_paired_se"] = round(se, 4)
                row["required_by_D21"] = round(need, 4)
                row["passes_D21"] = bool(gap > need)
                row["per_tier_delta"] = deltas
            res["policies"][pol][e] = row

    print(f"\n{'정책':<12}{'인코더':<14}{'종합':>9}{'Δ':>9}{'쌍대SE':>9}{'요구':>8}  판정")
    for pol in ("lpb", "selective3"):
        for e in encoders:
            r = res["policies"][pol][e]
            if e == base:
                print(f"  {pol:<10}{e:<14}{r['composite']:>9.4f}{'—':>9}{'—':>9}{'—':>8}  기준")
            else:
                print(f"  {pol:<10}{e:<14}{r['composite']:>9.4f}{r['delta_vs_base']:>+9.4f}"
                      f"{r['composite_paired_se']:>9.4f}{r['required_by_D21']:>8.4f}  "
                      f"{'통과' if r['passes_D21'] else '미달'}")

    res["note"] = ("Phase 21 D65 후속. 같은 fold·시드에서 인코더만 교체한 쌍대 A/B. "
                   "제출 대상(selective3)과 LPB 를 함께 잰다 — 부호가 다를 수 있기 때문(D56). "
                   "종합 SE 는 tier 독립 가정의 w² 합성 근사(§2.6 과 동일 한계).")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "probe_encoder_axis.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n  ({time.time() - t0:.0f}s) → eval/results/probe_encoder_axis.json")
    return res


if __name__ == "__main__":
    def _a(name, d, cast=str):
        return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else d
    main(_a("--folds", 3, int), _a("--n", 1200, int),
         tuple(_a("--encoders", "hashing:96,hashing:512").split(",")))
