"""검증기 예리함 스윕 — §3 "검증기 예리함 법칙"의 교차점을 **재현 가능하게** 만든다.

★ 왜 이 파일이 새로 쓰였는가 (D50 → Phase 20 해소)
--------------------------------------------------
`SUBMISSION.md` §3 의 법칙 표에는 인위 열화 3행(AUC 0.507 / 0.725 / 0.845)이 있고,
`eval/probe_verifier_real.py:32` 는 그 출처를 `probe_verifier_sensitivity.py` 라 적어 뒀다.
그런데 **그 파일이 저장소에 없었다.** 즉 이 프로젝트의 핵심 명제("순차 관측형 라우팅의
순가치는 검증기 예리함이 지배한다")의 교차점 상수 0.845 가 **재현 불가능**했고,
그 값이 `src/gate.py:BREAK_EVEN_AUC` 에 하드코딩돼 있었다. 외부 감사가 지적한 대로
"산문만 있는 수치는 쓰지 않는다"는 Phase 17 의 규칙이 코드 상수에는 적용되지 않았던 것이다.

이 스크립트가 그 3행을 실제로 만든다.

방법
----
검증기 점수 V(0~1)를 **연속적으로 열화**시켜 목표 AUC 를 맞춘다:

    V(α) = (1−α)·V_full + α·U,     U ~ Uniform(0,1) (고정 시드)

α=0 이면 원래 증강 검증기, α=1 이면 완전 무정보(AUC≈0.5). α 를 이분탐색해 목표 AUC 를
±0.005 안에서 맞춘 뒤, 그 관측 채널로 **LPB 와 학습형 단일호출을 같은 fold 에서** 돌려
`LPB − learned` 를 잰다. 학습형은 검증기를 쓰지 않으므로 α 에 **불변**이어야 하고
(그 자체가 이 실험의 내부 정합성 검사다), 따라서 격차의 움직임은 전부 LPB 쪽이다.

교차점은 격차가 0 을 지나는 AUC 를 선형 보간으로 추정한다.

사용법: python eval/probe_verifier_sensitivity.py [--folds 3] [--n 1200]
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
from src.encoder import DiscLR
from baselines.policies import LearnedRouter, tune_learned_lambda

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config_ax3.yaml", encoding="utf-8"))
OUT = ROOT / "eval" / "results"
TIERS = ["fast", "balanced", "premium"]
W = {t: CFG["tiers"][t]["weight"] for t in CFG["tiers"]}

#: 목표 AUC — §3 표의 인위 열화 3행 + 원본(열화 없음)
TARGETS = [0.507, 0.725, 0.845]


def auc(scores, labels) -> float:
    s = np.asarray(scores, dtype=float).ravel()
    y = (np.asarray(labels, dtype=float).ravel() > 0.5).astype(float)
    pos, neg = s[y > 0.5], s[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def degrade(V, U, alpha):
    return (1.0 - alpha) * V + alpha * U


def solve_alpha(V, U, quality, target, tol=0.005, iters=40):
    """목표 AUC 를 만드는 혼합계수 α 를 이분탐색. AUC 는 α 에 대해 단조 감소."""
    lo, hi = 0.0, 1.0
    a_lo = auc(degrade(V, U, lo), quality)
    if target >= a_lo:                       # 원본보다 높은 목표는 만들 수 없다
        return 0.0, a_lo
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        a = auc(degrade(V, U, mid), quality)
        if abs(a - target) <= tol:
            return mid, a
        if a > target:
            lo = mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    return mid, auc(degrade(V, U, mid), quality)


def main(nf=3, n_queries=1200):
    t0 = time.time()
    cfg = dict(CFG, synth=dict(CFG["synth"], n_queries=n_queries))
    tw = build_textworld(cfg, seed=42)
    ds, meta = to_dataset(tw, cfg, Ns=(1,))                     # 실전 조건 (Q2=No)
    ds.features = get_encoder("hashing").encode(meta["prompts"])
    ds.text_encoder = get_encoder("hashing")
    F = augmented_feature_matrix(meta, text_dim=64, use_prompt=True, use_agreement=True)
    folds = ds.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])
    rng = np.random.default_rng(2026)
    U = rng.random(ds.quality.shape)                            # 고정 잡음 채널
    print(f"[probe_verifier_sensitivity] n={ds.n} M={ds.m} folds={nf}", flush=True)

    # 원본(증강) 검증기 — α 탐색의 기준이 되는 out-of-sample 점수 행렬.
    # ★ 주의: **전체 fold** 를 돌아야 모든 행이 채워진다. 초판은 `range(nf)` 만 돌아
    # nf<k_folds 일 때 40% 행이 0 으로 남았고, 그 0 들이 AUC 를 0.914 → 0.651 로 끌어내려
    # 목표 0.725·0.845 가 "기준보다 높음"으로 판정돼 α=0 으로 붕괴했다 (자체 발견).
    V_full = np.zeros_like(ds.quality, dtype=float)
    for te in folds:                                            # ← 전체 fold
        tr = np.setdiff1d(np.arange(ds.n), te)
        Vf, _ = fold_verifier_matrix(F, ds.quality, tr)
        V_full[te] = Vf[te]                                     # 각 행은 자기 fold 의 out-of-sample 점수
    assert (V_full != 0).any(axis=1).all(), "일부 행이 채워지지 않았다 (fold 커버리지 확인)"
    base_auc = auc(V_full, ds.quality)
    print(f"  원본(증강) 검증기 AUC = {base_auc:.4f}")

    arms = []
    for target in TARGETS:
        alpha, got = solve_alpha(V_full, U, ds.quality, target)
        arms.append({"target_auc": target, "alpha": round(alpha, 5), "auc": round(got, 4)})
        print(f"  목표 AUC {target:.3f} → α={alpha:.4f} (실측 {got:.4f})", flush=True)
    arms.append({"target_auc": None, "alpha": 0.0, "auc": round(base_auc, 4)})   # 열화 없음

    print(f"\n  {'검증기 AUC':<12}{'LPB':<9}{'learned':<9}{'LPB−learned':<14}{'σ<0':<7}")
    rows = []
    for arm in arms:
        alpha = arm["alpha"]
        per_tier = {"lpb": {}, "learned": {}}
        sneg = []
        for tier in TIERS:
            ql, qn = [], []
            for f in range(nf):
                te = folds[f]
                tr = np.setdiff1d(np.arange(ds.n), te)
                Vf, _ = fold_verifier_matrix(F, ds.quality, tr)
                ds.verifier = degrade(Vf, U, alpha)             # ← 열화된 관측 채널
                b_te = tier_budgets(ds, te, CFG)[tier]
                b_tr = tier_budgets(ds, tr, CFG)[tier]
                r = LPBRouter(CFG, 1, seed=f, use_domain=False).fit(ds, tr, tier)
                ql.append(run_tier(ds, te, r.policy(), tier, b_te).mean_quality)
                if tier == "fast":
                    sneg.append(r.diagnostics["sigma_neg_frac"])
                # 학습형은 검증기를 쓰지 않는다 → α 에 불변이어야 한다 (정합성 검사)
                disc = DiscLR().fit(ds.features[tr], ds.domains[tr], ds.quality[tr])
                lam = tune_learned_lambda(disc, ds, tr, b_tr)
                qn.append(run_tier(ds, te, LearnedRouter(disc, lam), tier, b_te).mean_quality)
            per_tier["lpb"][tier] = float(np.mean(ql))
            per_tier["learned"][tier] = float(np.mean(qn))
        c_l = sum(W[t] * per_tier["lpb"][t] for t in TIERS)
        c_n = sum(W[t] * per_tier["learned"][t] for t in TIERS)
        row = {**arm, "lpb": round(c_l, 4), "learned": round(c_n, 4),
               "delta": round(c_l - c_n, 4),
               "sigma_neg_fast": round(float(np.mean(sneg)), 4),
               "per_tier": {k: {t: round(v, 4) for t, v in d.items()}
                            for k, d in per_tier.items()}}
        rows.append(row)
        print(f"  {arm['auc']:<12.4f}{c_l:<9.4f}{c_n:<9.4f}{c_l - c_n:<+14.4f}"
              f"{np.mean(sneg):<7.2f}", flush=True)

    # 정합성 검사: 학습형은 검증기를 쓰지 않으므로 전 arm 에서 같아야 한다
    lset = {r["learned"] for r in rows}
    consistent = len(lset) == 1
    print(f"\n  정합성 검사 (학습형은 α 에 불변이어야 함): "
          f"{'PASS' if consistent else 'FAIL'} — 값 {sorted(lset)}")

    # 교차점: delta 가 0 을 지나는 AUC 를 선형 보간
    pts = sorted([(r["auc"], r["delta"]) for r in rows])
    cross = None
    for (a1, d1), (a2, d2) in zip(pts, pts[1:]):
        if d1 <= 0.0 <= d2 and d2 != d1:
            cross = a1 + (a2 - a1) * (0.0 - d1) / (d2 - d1)
            break
    print(f"  → 교차점(LPB=learned) 추정 AUC ≈ "
          f"{'%.4f' % cross if cross is not None else '구간 내 교차 없음'}")

    res = {"n": ds.n, "n_folds": nf, "base_auc": round(base_auc, 4),
           "rows": rows, "crossing_auc": None if cross is None else round(cross, 4),
           "learned_invariance_check": {"passed": consistent, "values": sorted(lset)},
           "note": ("§3 법칙 표의 인위 열화 행을 만드는 스크립트 (Phase 20 신설, D50 해소). "
                    "열화는 V(α)=(1−α)V+αU 혼합이며 α 는 목표 AUC 로 이분탐색한다.")}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "probe_verifier_sensitivity.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"  ({time.time() - t0:.0f}s) → eval/results/probe_verifier_sensitivity.json")
    return res


if __name__ == "__main__":
    nf = int(sys.argv[sys.argv.index("--folds") + 1]) if "--folds" in sys.argv else 3
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 1200
    main(nf, n)
