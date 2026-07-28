"""Phase 8: 실텍스트 예행 — ScoringVerifier(권고 1) + TextEncoder(권고 2) 통합 검증.

  8a: verifier 단독 품질 — held-out AUC/logloss/ECE + 특성군 ablation
      (자기일관성/구조검사/통계신호의 기여 분해)
  8b: World F(textworld) end-to-end — 실텍스트 verifier·encoder가 꽂힌 최종 라우터
      vs cascade, 손실 분해, tie-break 지연

사용법: python run_phase8.py {8a|8b} [--encoder hashing|st]
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
from src.harness import run_tier, tier_budgets
from src.cost_mirror import cost_matrix
from src.textworld import build_textworld, to_dataset
from src.verifier import feature_matrix, fold_verifier_matrix, FEATURE_NAMES, ScoringVerifier
from src.text_encoder import get_encoder
from src.router import LPBRouter
from run_phase2 import CFG, OUT
from run_diagnostics import diagnose
from baselines.policies import StaticCascade, tune_cascade_tau

TIERS = ["fast", "balanced", "premium"]
W = {t: CFG["tiers"][t]["weight"] for t in CFG["tiers"]}
_CACHE = {}


def build_f(seed=42, encoder_backend="hashing"):
    key = (seed, encoder_backend)
    if key not in _CACHE:
        tw = build_textworld(CFG, seed=seed)
        ds, meta = to_dataset(tw, CFG)
        ds.features = get_encoder(encoder_backend).encode(meta["prompts"])
        F = feature_matrix(meta)
        folds = ds.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])
        _CACHE[key] = (tw, ds, meta, F, folds)
    return _CACHE[key]


def _auc(y, s):
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _ece(p, y, bins=10):
    tot = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (p >= lo) & (p < lo + 1 / bins)
        if m.sum():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def _logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def stage_8a(encoder_backend="hashing"):
    tw, ds, meta, F, folds = build_f(encoder_backend=encoder_backend)
    groups = {
        "full": list(range(len(FEATURE_NAMES))),
        "-vote_share": [i for i, n in enumerate(FEATURE_NAMES) if n != "vote_share"],
        "-struct": [i for i, n in enumerate(FEATURE_NAMES)
                    if n not in ("struct", "has_marker", "parse_ok")],
        "len-only": [i for i, n in enumerate(FEATURE_NAMES)
                     if n in ("out_len", "ans_len")],
    }
    print(f"[8a] ScoringVerifier held-out 품질 (5-fold, 셀 단위 n={ds.n}x{ds.m})")
    print(f"  기저율 P(y=1) = {ds.quality.mean():.4f}")
    print(f"  {'특성군':<14}{'AUC':<9}{'logloss':<10}{'ECE':<8}")
    report = {}
    for gname, idx in groups.items():
        aucs, lls, eces = [], [], []
        for f, te in enumerate(folds):
            tr = np.setdiff1d(np.arange(ds.n), te)
            d = len(idx)
            sv = ScoringVerifier().fit(F[tr][:, :, idx].reshape(-1, d),
                                       ds.quality[tr].ravel())
            p = sv.score(F[te][:, :, idx].reshape(-1, d))
            y = ds.quality[te].ravel()
            aucs.append(_auc(y, p)); lls.append(_logloss(p, y)); eces.append(_ece(p, y))
        report[gname] = {"auc": round(float(np.mean(aucs)), 4),
                         "logloss": round(float(np.mean(lls)), 4),
                         "ece": round(float(np.mean(eces)), 4)}
        r = report[gname]
        print(f"  {gname:<14}{r['auc']:<9}{r['logloss']:<10}{r['ece']:<8}")
    json.dump(report, open(OUT / "phase8a.json", "w", encoding="utf-8"), indent=1)


def stage_8b(encoder_backend="hashing", seed=42, tag="8b", quiet=False,
             prize_mode="calibrated"):
    tw, ds, meta, F, folds = build_f(seed=seed, encoder_backend=encoder_backend)
    rows = {}
    order = list(np.argsort(cost_matrix(ds).mean(axis=0)))
    for tier in TIERS:
        lq, cq, diags = [], [], []
        for f, te in enumerate(folds):
            tr = np.setdiff1d(np.arange(ds.n), te)
            V, _ = fold_verifier_matrix(F, ds.quality, tr)     # verifier: train으로만 적합
            ds.verifier = V
            router = LPBRouter(CFG, len(np.unique(ds.domains)), seed=f,
                               prize_mode=prize_mode).fit(ds, tr, tier)
            b_te = tier_budgets(ds, te, CFG)[tier]
            d = diagnose(ds, te, router, tier, b_te)
            lq.append(d["quality"]); diags.append(d)
            tau = tune_cascade_tau(ds, tr, CFG, tier, tier_budgets(ds, tr, CFG)[tier])
            cq.append(run_tier(ds, te, StaticCascade(order, tau), tier, b_te).mean_quality)
        cats = {k: round(float(np.mean([x["cats_pct"][k] for x in diags])), 2)
                for k in diags[0]["cats_pct"]}
        rows[tier] = {"lpb": round(float(np.mean(lq)), 4),
                      "std": round(float(np.std(lq)), 4),
                      "cascade": round(float(np.mean(cq)), 4),
                      "oracle": round(float(np.mean([x["oracle_any"] for x in diags])), 4),
                      "ms": round(float(np.mean([x["route_ms_mean"] for x in diags])), 3),
                      "cats": cats}
    comb_l = sum(W[t] * rows[t]["lpb"] for t in TIERS)
    comb_c = sum(W[t] * rows[t]["cascade"] for t in TIERS)
    if not quiet:
        print(f"\n[{tag} | World F textworld | encoder={encoder_backend} | seed={seed}]")
        bs = ds.quality[:, [0, 3, 6, 9, 12]].mean(axis=0)   # N=1 열의 모델별 정확도
        print(f"  모델별 단독 정확도(N=1): {np.round(bs, 3).tolist()}")
        print(f"  {'tier':<10}{'LPB(+-std)':<19}{'cascade':<10}{'oracle':<9}"
              f"{'sel.err%':<9}{'regret%':<9}{'unans%':<8}{'ms/q':<7}")
        for t in TIERS:
            r = rows[t]
            print(f"  {t:<10}{r['lpb']:.4f} +-{r['std']:.4f}    {r['cascade']:<10.4f}"
                  f"{r['oracle']:<9.4f}{r['cats']['selection_error']:<9}"
                  f"{r['cats']['early_stop_regret']:<9}{r['cats']['unanswered']:<8}"
                  f"{r['ms']:<7}")
        print(f"  종합점수: LPB={comb_l:.4f}  cascade={comb_c:.4f}")
    json.dump(rows, open(OUT / f"phase{tag}.json", "w", encoding="utf-8"), indent=1)
    return comb_l, comb_c


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "8a"
    backend = sys.argv[sys.argv.index("--encoder") + 1] if "--encoder" in sys.argv \
        else "hashing"
    {"8a": lambda: stage_8a(backend), "8b": lambda: stage_8b(backend)}[stage]()
