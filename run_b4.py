"""B4 실행기 — Q4=연속 품질 분기의 실증 (PROJECT_PLAN §9 P1).

World F-c "textworld-graded": World F와 동일한 실텍스트·실출력이되 품질 라벨만
부분점수(연속)로 바꾼 세계. 신념 모형(Beta 반응모형)이 데이터를 생성하지 않았으므로
자기확증이 차단된다.

  b4a: Beta 반응모형 적합 품질 — held-out 주변 LL, κ 추정, 이진 2PL 대비
  b4b: end-to-end 3자 비교
         ① beta      : 연속 처리 (Beta-IRT + Beta 예약값)   ← B4 본체
         ② binarized : 차선책 (품질을 0.5에서 이진화해 기존 닫힌형 파이프라인)
         ③ cascade   : 정적 threshold 베이스라인
       ②는 "B4 없이도 돌아간다"의 실제 비용을 재는 대조군이다.
  b4c: σ 일반해가 **이진** 세계 점수에 미친 영향 (음수 σ 결함 수정의 순효과)

사용법: python run_b4.py {b4a|b4b|b4c}
"""
import sys, pathlib, json

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

from src.harness import run_tier, tier_budgets
from src.cost_mirror import cost_matrix
from src.textworld import build_textworld, to_dataset
from src.verifier import feature_matrix, fold_verifier_matrix
from src.text_encoder import get_encoder
from src.irt.mml import fit_mml, heldout_marginal_ll
from src.irt.beta_mml import fit_beta_mml, beta_heldout_ll
from src.router import LPBRouter
from run_phase2 import CFG, OUT
from baselines.policies import StaticCascade, tune_cascade_tau

TIERS = ["fast", "balanced", "premium"]
W = {t: CFG["tiers"][t]["weight"] for t in CFG["tiers"]}
_CACHE = {}


def build_fc(seed=42, graded=True, scale="strict"):
    """World F-c (graded=True) 또는 원본 World F (graded=False)."""
    key = (seed, graded, scale)
    if key not in _CACHE:
        tw = build_textworld(CFG, seed=seed)
        ds, meta = to_dataset(tw, CFG, graded=graded, scale=scale)
        ds.features = get_encoder("hashing").encode(meta["prompts"])
        F = feature_matrix(meta)
        folds = ds.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])
        _CACHE[key] = (ds, meta, F, folds)
    return _CACHE[key]


# ---------------- b4a: 신념 모형 적합 품질 ----------------

def stage_b4a(scale="strict"):
    ds, meta, F, folds = build_fc(scale=scale)
    q = ds.quality
    print(f"[b4a] World F-c 부분점수 분포 (n={ds.n} × {ds.m}열, scale={scale})")
    uniq = np.unique(q)
    print(f"  수준 수={len(uniq)}  평균={q.mean():.4f}  "
          f"0인 비율={float((q == 0).mean()):.3f}  1인 비율={float((q == 1).mean()):.3f}")
    print(f"  내부(0<q<1) 비율={float(((q > 0) & (q < 1)).mean()):.3f} "
          f"— 이 부분이 이진화로 소실되는 정보")

    lls_beta, lls_bern, kappas = [], [], []
    for te in folds:
        tr = np.setdiff1d(np.arange(ds.n), te)
        pb = fit_beta_mml(q[tr], ds.domains[tr], CFG["synth"]["n_domains"],
                          per_domain_b=True)
        lls_beta.append(beta_heldout_ll(pb, q[te], ds.domains[te]))
        kappas.append(pb["kappa"])
        # 이진화 대조: 같은 데이터를 0.5에서 자른 뒤 2PL 적합
        yb = (q >= 0.5).astype(float)
        p2 = fit_mml(yb[tr], ds.domains[tr], CFG["synth"]["n_domains"], per_domain_b=True)
        lls_bern.append(heldout_marginal_ll(p2, yb[te], ds.domains[te]))
    kap = np.mean(kappas, axis=0)
    print(f"\n  Beta 반응모형 held-out 주변LL   : {np.mean(lls_beta):+.4f} (연속 밀도)")
    print(f"  이진화 2PL held-out 주변LL      : {np.mean(lls_bern):+.4f} (이산 확률)")
    print(f"  → 척도가 달라 직접 비교 불가. 판정은 b4b의 라우팅 점수로 한다.")
    print(f"  적합된 κ (모델×N표 15열): {np.round(kap, 2).tolist()}")
    print(f"    κ가 작을수록 0/1 양극, 클수록 중간값 집중 (κ→0 이 Bernoulli 극한)")
    json.dump({"ll_beta": float(np.mean(lls_beta)), "ll_binarized": float(np.mean(lls_bern)),
               "kappa": kap.tolist(), "levels": int(len(uniq)),
               "interior_frac": float(((q > 0) & (q < 1)).mean())},
              open(OUT / "b4a.json", "w", encoding="utf-8"), indent=1)


# ---------------- b4b: end-to-end 3자 비교 ----------------

def _run_variant(ds, meta, F, folds, tier, family, binarize, timed=False):
    """family='beta'|'bernoulli', binarize=True면 라우터가 이진 라벨만 본다.

    ★ 채점은 언제나 **연속 품질**로 한다 (실제 챌린지 채점 = 연속). 이진화는
    라우터의 학습·의사결정 정보만 깎는 것이지 점수 척도를 바꾸는 게 아니다.
    """
    import time
    qs, secs = [], []
    q_true = ds.quality.copy()
    for f, te in enumerate(folds):
        tr = np.setdiff1d(np.arange(ds.n), te)
        ds.quality = (q_true >= 0.5).astype(float) if binarize else q_true
        ds.verifier, _ = fold_verifier_matrix(F, ds.quality, tr)
        t0 = time.perf_counter()
        router = LPBRouter(CFG, CFG["synth"]["n_domains"], seed=f,
                           prize_family=family).fit(ds, tr, tier)
        secs.append(time.perf_counter() - t0)
        pol = router.policy()
        ds.quality = q_true                       # 채점은 연속 라벨로
        b_te = tier_budgets(ds, te, CFG)[tier]
        qs.append(run_tier(ds, te, pol, tier, b_te).mean_quality)
    ds.quality = q_true
    if timed:
        return float(np.mean(qs)), float(np.std(qs)), float(np.mean(secs))
    return float(np.mean(qs)), float(np.std(qs))


def stage_b4b(scale="strict"):
    ds, meta, F, folds = build_fc(scale=scale)
    q_true = ds.quality.copy()
    order = list(np.argsort(cost_matrix(ds).mean(axis=0)))
    rows = {}
    print(f"[b4b] World F-c end-to-end (5-fold, 채점=연속 품질, scale={scale})\n")
    print(f"  {'tier':<10}{'① beta':<18}{'② binarized':<18}{'③ cascade':<12}"
          f"{'oracle':<10}{'①−②':<9}")
    for tier in TIERS:
        m_beta, s_beta = _run_variant(ds, meta, F, folds, tier, "beta", False)
        m_bin, s_bin = _run_variant(ds, meta, F, folds, tier, "bernoulli", True)
        cq = []
        for f, te in enumerate(folds):
            tr = np.setdiff1d(np.arange(ds.n), te)
            ds.verifier, _ = fold_verifier_matrix(F, q_true, tr)
            tau = tune_cascade_tau(ds, tr, CFG, tier, tier_budgets(ds, tr, CFG)[tier])
            cq.append(run_tier(ds, te, StaticCascade(order, tau), tier,
                               tier_budgets(ds, te, CFG)[tier]).mean_quality)
        orc = float(np.mean([q_true[te].max(axis=1).mean() for te in folds]))
        rows[tier] = {"beta": round(m_beta, 4), "beta_std": round(s_beta, 4),
                      "binarized": round(m_bin, 4), "bin_std": round(s_bin, 4),
                      "cascade": round(float(np.mean(cq)), 4), "oracle": round(orc, 4)}
        r = rows[tier]
        print(f"  {tier:<10}{r['beta']:.4f} ±{s_beta:.4f}   {r['binarized']:.4f} "
              f"±{s_bin:.4f}   {r['cascade']:<12.4f}{r['oracle']:<10.4f}"
              f"{r['beta'] - r['binarized']:+.4f}")
    comb = {k: sum(W[t] * rows[t][k] for t in TIERS)
            for k in ("beta", "binarized", "cascade")}
    print(f"\n  종합점수  ① beta={comb['beta']:.4f}  ② binarized={comb['binarized']:.4f}"
          f"  ③ cascade={comb['cascade']:.4f}")
    print(f"  B4의 순가치 (①−②) = {comb['beta'] - comb['binarized']:+.4f}")
    rows["combined"] = {k: round(v, 4) for k, v in comb.items()}
    rows["scale"] = scale
    json.dump(rows, open(OUT / f"b4b-{scale}.json", "w", encoding="utf-8"), indent=1)


# ---------------- b4c: σ 일반해가 이진 세계에 미친 순효과 ----------------

def stage_b4c():
    """음수 σ 결함 수정의 효과를 원본 World F(이진)에서 측정."""
    ds, meta, F, folds = build_fc(graded=False)
    print("[b4c] 원본 World F (이진) — σ 일반해 적용 후 점수")
    print(f"  {'tier':<10}{'LPB':<12}{'std':<10}")
    rows = {}
    for tier in TIERS:
        qs = []
        for f, te in enumerate(folds):
            tr = np.setdiff1d(np.arange(ds.n), te)
            ds.verifier, _ = fold_verifier_matrix(F, ds.quality, tr)
            router = LPBRouter(CFG, CFG["synth"]["n_domains"], seed=f).fit(ds, tr, tier)
            qs.append(run_tier(ds, te, router.policy(), tier,
                               tier_budgets(ds, te, CFG)[tier]).mean_quality)
        rows[tier] = {"lpb": round(float(np.mean(qs)), 4),
                      "std": round(float(np.std(qs)), 4)}
        print(f"  {tier:<10}{rows[tier]['lpb']:<12.4f}{rows[tier]['std']:<10.4f}")
    comb = sum(W[t] * rows[t]["lpb"] for t in TIERS)
    print(f"  종합점수={comb:.4f}   (Phase 8 기록: 0.8192)")
    rows["combined"] = round(comb, 4)
    json.dump(rows, open(OUT / "b4c.json", "w", encoding="utf-8"), indent=1)


def stage_b4d(scale="strict"):
    """★ GRM 팔 추가 — B4가 남긴 오설정(점질량)이 실제로 해소되는가.

    World F-c는 품질의 65.8%가 정확히 1.0인 점질량 지배 분포다. Beta는 개구간 밀도라
    이를 표현하지 못하고, 그래서 b4b에서 연속 처리의 이득이 +0.0026(무차이)에 그쳤다.
    GRM은 등급확률을 직접 모델링하므로 점질량이 자연스럽다 — 그 예측을 검증한다.
    """
    ds, meta, F, folds = build_fc(scale=scale)
    rows = {}
    print(f"[b4d] World F-c — GRM vs Beta vs 이진화 (5-fold, 채점=연속, scale={scale})\n")
    print(f"  {'tier':<10}{'① GRM':<20}{'② Beta':<20}{'③ 이진화':<20}"
          f"{'①−③':<9}{'①−②':<9}")
    for tier in TIERS:
        g, gs, gt = _run_variant(ds, meta, F, folds, tier, "grm", False, timed=True)
        b, bs, bt = _run_variant(ds, meta, F, folds, tier, "beta", False, timed=True)
        z, zs, zt = _run_variant(ds, meta, F, folds, tier, "bernoulli", True, timed=True)
        rows[tier] = {"grm": round(g, 4), "grm_std": round(gs, 4), "grm_fit_s": round(gt, 1),
                      "beta": round(b, 4), "beta_std": round(bs, 4), "beta_fit_s": round(bt, 1),
                      "binarized": round(z, 4), "bin_std": round(zs, 4),
                      "bin_fit_s": round(zt, 1)}
        print(f"  {tier:<10}{g:.4f} ±{gs:.4f} {gt:>5.0f}s  {b:.4f} ±{bs:.4f} {bt:>5.0f}s  "
              f"{z:.4f} ±{zs:.4f} {zt:>5.0f}s  {g - z:+.4f}  {g - b:+.4f}")
    comb = {k: sum(W[t] * rows[t][k] for t in TIERS) for k in ("grm", "beta", "binarized")}
    print(f"\n  종합점수  ① GRM={comb['grm']:.4f}  ② Beta={comb['beta']:.4f}  "
          f"③ 이진화={comb['binarized']:.4f}")
    print(f"  연속 처리의 순가치 (GRM−이진화) = {comb['grm'] - comb['binarized']:+.4f}"
          f"   (Beta로는 {comb['beta'] - comb['binarized']:+.4f} 였다)")
    rows["combined"] = {k: round(v, 4) for k, v in comb.items()}
    json.dump(rows, open(OUT / f"b4d-{scale}.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "b4a"
    sc = sys.argv[sys.argv.index("--scale") + 1] if "--scale" in sys.argv else "strict"
    {"b4a": lambda: stage_b4a(sc), "b4b": lambda: stage_b4b(sc),
     "b4c": stage_b4c, "b4d": lambda: stage_b4d(sc)}[stage]()
