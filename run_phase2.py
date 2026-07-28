"""Phase 2 실행기 (단계별): 2a MML 적합 → 2b Amortized 인코더 → 2c 라우팅 가치 검증.

Phase 1과 동일한 데이터 사용: seed 42의 두 세계 + 동일 층화 5-fold (연속성 요건).
사용법: python run_phase2.py {2a|2b|2c}
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import yaml
from src.synth import make_world
from src.irt.mml import fit_mml, heldout_marginal_ll

ROOT = pathlib.Path(__file__).resolve().parent
CFG = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
OUT = ROOT / "eval" / "results"


def load_worlds_and_folds():
    """Phase 1과 완전히 동일한 데이터·fold (seed 고정 재생성 → phase1.json과 정합)."""
    worlds = {w: make_world(CFG, w, seed=CFG["seed"]) for w in ["irt", "specialist"]}
    folds = {w: ds.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])
             for w, ds in worlds.items()}
    return worlds, folds


def true_p(ds) -> np.ndarray:
    """세계 생성기의 진짜 정답확률 (평가 전용 오라클 — 정책은 접근 불가)."""
    if ds.world == "irt":
        th, a, b = ds.extras["theta"], ds.extras["a"], ds.extras["b"]
        return 1 / (1 + np.exp(-a[None, :] * (th[:, None] - b[None, :])))
    skill, diff = ds.extras["skill"], ds.extras["difficulty"]
    slope = CFG["synth"]["specialist"]["difficulty_slope"]
    return np.clip(skill[ds.domains] - slope * (diff[:, None] - 0.5), 0.02, 0.98)


def stage_2a():
    worlds, folds = load_worlds_and_folds()
    n_dom = CFG["synth"]["n_domains"]
    report = {}
    for wname, ds in worlds.items():
        rows = []
        fitted_b = {"1d": [], "dom": []}
        for f, test_idx in enumerate(folds[wname]):
            train_idx = np.setdiff1d(np.arange(ds.n), test_idx)
            for tag, pdb in [("irt-1d", False), ("irt-domain", True)]:
                fit = fit_mml(ds.quality[train_idx], ds.domains[train_idx], n_dom,
                              per_domain_b=pdb)
                ho = heldout_marginal_ll(fit, ds.quality[test_idx], ds.domains[test_idx])
                rows.append({"fold": f, "variant": tag, "train_ll": round(fit["final_ll"], 4),
                             "heldout_ll": round(ho, 4),
                             "a": np.round(fit["a"], 3).tolist(),
                             "b_mean": np.round(fit["b"].mean(axis=0), 3).tolist()})
                fitted_b["dom" if pdb else "1d"].append(fit["b"])
                if wname == "irt" and not pdb:
                    ca = np.corrcoef(fit["a"], ds.extras["a"])[0, 1]
                    cb = np.corrcoef(fit["b"].ravel(), ds.extras["b"])[0, 1]
                    rows[-1]["recover_corr_a"] = round(float(ca), 4)
                    rows[-1]["recover_corr_b"] = round(float(cb), 4)
        # fold 안정성: 적합된 b의 fold 간 표준편차
        for tag in ["1d", "dom"]:
            stack = np.stack(fitted_b[tag])
            report[f"{wname}/b_fold_std/{tag}"] = round(float(stack.std(axis=0).mean()), 4)
        report[wname] = rows
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT / "phase2a.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    for wname in worlds:
        print(f"\n[2a | World: {wname}]  (held-out 주변 로그우도, 쿼리당 — 높을수록 좋음)")
        for tag in ["irt-1d", "irt-domain"]:
            sub = [r for r in report[wname] if r["variant"] == tag]
            ho = np.array([r["heldout_ll"] for r in sub])
            line = f"  {tag:<12} heldout_ll = {ho.mean():.4f} +-{ho.std():.4f}"
            if tag == "irt-1d" and wname == "irt":
                ca = np.mean([r["recover_corr_a"] for r in sub])
                cb = np.mean([r["recover_corr_b"] for r in sub])
                line += f"   | 파라미터 복원 corr(a)={ca:.3f} corr(b)={cb:.3f}"
            print(line)
        print(f"  b fold-안정성(std): 1d={report[f'{wname}/b_fold_std/1d']}"
              f"  domain={report[f'{wname}/b_fold_std/dom']}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "2a"
    if stage == "2a":
        stage_2a()
    elif stage == "2b":
        from phase2_stages import stage_2b
        stage_2b()
    elif stage == "2c":
        from phase2_stages import stage_2c
        stage_2c()
