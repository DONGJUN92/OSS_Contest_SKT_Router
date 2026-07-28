"""Phase 2b/2c: Amortized 인코더 학습 + 다운스트림 라우팅 가치 검증.

2b 완료 기준(PROJECT_PLAN): IRT 인코더의 held-out log-loss가 판별식 베이스라인 대비
열세가 아니고, 캘리브레이션(ECE)이 양호할 것.
2c: 신호 품질이 실제 라우팅 점수로 이어지는지 — Phase 1 하네스로 mini 검증.

**Phase 17 이관**: `IRTEncoder`·`DiscLR`·`C_PROBIT` 는 배포 경로(`src/router.py`)가
쓰는 라이브러리 부품이므로 `src/encoder.py` 로 옮겼다. 이 모듈은 재수출 shim 이며
기존 실행기·프로브·테스트는 무변경 동작한다 (근거: `src/encoder.py` 문서 문자열).
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import yaml
from src.irt.mml import fit_mml, Adam, _sigmoid
from src.encoder import C_PROBIT, IRTEncoder, DiscLR      # 하위호환 재수출
from src.harness import run_tier, tier_budgets
from src.cost_mirror import cost_matrix
from run_phase2 import load_worlds_and_folds, true_p, CFG, OUT


# ---------------- 지표 ----------------

def cell_logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def ece(p, y, bins=10):
    p, y = p.ravel(), y.ravel()
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi)
        if mask.sum() > 0:
            total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


# ---------------- Stage 2b ----------------

def stage_2b():
    worlds, folds = load_worlds_and_folds()
    n_dom = CFG["synth"]["n_domains"]
    report = {}
    for wname, ds in worlds.items():
        tp = true_p(ds)
        rows = []
        for f, test_idx in enumerate(folds[wname]):
            tr = np.setdiff1d(np.arange(ds.n), test_idx)
            irt_fit = fit_mml(ds.quality[tr], ds.domains[tr], n_dom, per_domain_b=True)
            preds = {
                "irt-enc": IRTEncoder(irt_fit["a"], irt_fit["b"]).fit(
                    ds.features[tr], ds.domains[tr], ds.quality[tr]),
                "disc-lr": DiscLR().fit(ds.features[tr], ds.domains[tr], ds.quality[tr]),
            }
            for name, model in preds.items():
                p = model.predict(ds.features[test_idx], ds.domains[test_idx])
                rows.append({
                    "fold": f, "variant": name,
                    "logloss": round(cell_logloss(p, ds.quality[test_idx]), 4),
                    "ece": round(ece(p, ds.quality[test_idx]), 4),
                    "corr_truep": round(float(np.corrcoef(p.ravel(), tp[test_idx].ravel())[0, 1]), 4),
                })
        report[wname] = rows
    json.dump(report, open(OUT / "phase2b.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    for wname, rows in report.items():
        print(f"\n[2b | World: {wname}]  (held-out, 5-fold 평균)")
        print(f"  {'variant':<10} {'logloss↓':<12} {'ECE↓':<10} {'corr(true_p)↑':<12}")
        for v in ["irt-enc", "disc-lr"]:
            sub = [r for r in rows if r["variant"] == v]
            ll = np.mean([r["logloss"] for r in sub])
            ee = np.mean([r["ece"] for r in sub])
            cc = np.mean([r["corr_truep"] for r in sub])
            print(f"  {v:<10} {ll:<12.4f} {ee:<10.4f} {cc:<12.4f}")


# ---------------- Stage 2b-low: 저데이터 효율 (2b reflection에서 추가) ----------------

def stage_2b_low():
    """train을 n_sub개로 제한 — 파라미터 수가 적은 IRT의 표본 효율 가설 검증."""
    worlds, folds = load_worlds_and_folds()
    n_dom = CFG["synth"]["n_domains"]
    rng = np.random.default_rng(CFG["seed"])
    print("\n[2b-low] 저데이터 held-out logloss (5-fold 평균, 낮을수록 좋음)")
    for wname, ds in worlds.items():
        print(f"  World: {wname}")
        for n_sub in [100, 200, 400]:
            res = {"irt-enc": [], "disc-lr": []}
            for f, test_idx in enumerate(folds[wname]):
                tr_full = np.setdiff1d(np.arange(ds.n), test_idx)
                tr = rng.choice(tr_full, size=n_sub, replace=False)
                irt_fit = fit_mml(ds.quality[tr], ds.domains[tr], n_dom, per_domain_b=True)
                preds = {
                    "irt-enc": IRTEncoder(irt_fit["a"], irt_fit["b"]).fit(
                        ds.features[tr], ds.domains[tr], ds.quality[tr]),
                    "disc-lr": DiscLR().fit(ds.features[tr], ds.domains[tr], ds.quality[tr]),
                }
                for name, model in preds.items():
                    p = model.predict(ds.features[test_idx], ds.domains[test_idx])
                    res[name].append(cell_logloss(p, ds.quality[test_idx]))
            line = f"    n_train={n_sub:<5}"
            for name in ["irt-enc", "disc-lr"]:
                line += f" {name}={np.mean(res[name]):.4f}+-{np.std(res[name]):.4f} "
            print(line)


# ---------------- Stage 2c: mini 라우팅 검증 (Phase 1 하네스 재사용) ----------------

class SingleShotIndex:
    """p̄ 기반 단일 호출 정책: argmax(p̄ − λ·cost). Phase 3 M2의 축소판 프리뷰."""

    def __init__(self, predictor, lam, name):
        self.pred, self.lam, self.name = predictor, lam, name

    def route(self, sess, features, domain) -> int:
        pbar = self.pred.predict_row(features, domain)
        order = np.argsort(-(pbar - self.lam * sess.costs))
        for m in order:                                # 최고 지수부터, 감당 가능한 것 호출
            if sess.call(int(m)) is not None:
                return int(m)
        return 0


class OracleP:
    """진짜 생성확률을 아는 오라클 예측기 (상한 참조 — 인덱스 카운터 방식)."""

    def __init__(self, tp, idx):
        self.tp, self.rows, self.k = tp, idx, 0

    def predict_row(self, x, d):
        p = self.tp[self.rows[self.k % len(self.rows)]]
        self.k += 1
        return p


def _tune_lambda(pred, ds, idx, budget):
    """train fold에서 총지출 ≤ budget이 되는 최소 λ를 이분 탐색."""
    cmat = cost_matrix(ds)
    pbar = pred.predict_row  # row-wise
    P = np.stack([pbar(ds.features[i], int(ds.domains[i])) for i in idx])
    lo, hi = 0.0, 1e5
    for _ in range(40):
        mid = (lo + hi) / 2
        chosen = np.argmax(P - mid * cmat[idx], axis=1)
        spend = cmat[idx, chosen].sum()
        if spend > budget:
            lo = mid
        else:
            hi = mid
    return hi


def stage_2c():
    worlds, folds = load_worlds_and_folds()
    n_dom = CFG["synth"]["n_domains"]
    report = {}
    for wname, ds in worlds.items():
        tp = true_p(ds)
        rows = []
        for f, test_idx in enumerate(folds[wname]):
            tr = np.setdiff1d(np.arange(ds.n), test_idx)
            irt_fit = fit_mml(ds.quality[tr], ds.domains[tr], n_dom, per_domain_b=True)
            preds = {
                "irt-enc": IRTEncoder(irt_fit["a"], irt_fit["b"]).fit(
                    ds.features[tr], ds.domains[tr], ds.quality[tr]),
                "disc-lr": DiscLR().fit(ds.features[tr], ds.domains[tr], ds.quality[tr]),
                "oracle-p": OracleP(tp, test_idx),
            }
            budgets_te = tier_budgets(ds, test_idx, CFG)
            budgets_tr = tier_budgets(ds, tr, CFG)
            for tier in ["fast", "balanced"]:          # 승부처 저예산 tier만
                for name, pred in preds.items():
                    # D4 수정: λ는 반드시 각 예측기 자신의 train 예측으로 튜닝 (공정성)
                    lam_pred = OracleP(tp, tr) if name == "oracle-p" else pred
                    lam = _tune_lambda(lam_pred, ds, tr, budgets_tr[tier])
                    if name == "oracle-p":
                        pred = OracleP(tp, test_idx)   # 평가용 카운터 재생성
                    pol = SingleShotIndex(pred, lam, name)
                    r = run_tier(ds, test_idx, pol, tier, budgets_te[tier])
                    rows.append({"fold": f, "tier": tier, "policy": name,
                                 "quality": round(r.mean_quality, 4),
                                 "cost_pct": round(100 * r.total_cost / r.budget, 1),
                                 "unanswered": r.unanswered})
        report[wname] = rows
    json.dump(report, open(OUT / "phase2c.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    phase1 = json.load(open(OUT / "phase1.json", encoding="utf-8"))
    for wname, rows in report.items():
        print(f"\n[2c | World: {wname}]  단일호출 지수 정책 (λ는 train에서 이분탐색)")
        print(f"  {'tier':<10}{'policy':<12}{'quality(+-std)':<20}{'budget%':<10}{'unans':<6}"
              f"   vs Phase1 static-cascade")
        for tier in ["fast", "balanced"]:
            p1 = np.mean([r["quality"] for r in phase1[wname]["rows"]
                          if r["tier"] == tier and r["policy"] == "static-cascade"])
            for pol in ["irt-enc", "disc-lr", "oracle-p"]:
                sub = [r for r in rows if r["tier"] == tier and r["policy"] == pol]
                q = np.array([r["quality"] for r in sub])
                c = np.mean([r["cost_pct"] for r in sub])
                u = sum(r["unanswered"] for r in sub)
                print(f"  {tier:<10}{pol:<12}{q.mean():.4f} +-{q.std():.4f}    "
                      f"{c:>6.1f}%   {u:<6}   (cascade={p1:.4f})")
