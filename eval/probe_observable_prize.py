"""관측 상금 예약값의 값 측정 (Phase 17 — 레드팀 CRITICAL #2).

결함: Weitzman 예약값은 "상자를 열어 **실제로 얻는 상금**"의 분포로 풀어야 하는데, 엔진은
잠재 품질 Q 로 풀고 있었다. 이 라우터가 상자를 열어 얻는 것은 X=E[Q|v] (정지 규칙과 최종
선택이 쓰는 값)이고, X 는 Q 의 mean-preserving contraction 이라 σ 가 체계적으로 과대해진다.
상자별 검증기 예리함이 다르면(PerModelNoise) 과대 정도가 달라 **상자 순서까지 뒤집힌다**.

여기서 재는 것
  ① 해석적 확인: 같은 (p̄, λc)에서 latent σ vs observable σ, 그리고 순서 역전 사례
  ② 종합 점수: prize_reservation="latent" vs "observable" (동일 fold·동일 예산)
  ③ open_order A/B (Phase 17에서 인자 연결이 고쳐졌으므로 **처음으로 실제 실행**된다)

사용법: python eval/probe_observable_prize.py [--folds N] [--worlds a,b,...]
"""
import sys, pathlib, json, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from src.engine.pandora import (NoiseModel, observable_prize_grid, V_GRID)
from src.engine.prize import bernoulli_reservation, perbox_reservation
from src.router import LPBRouter
from src.harness import run_tier, tier_budgets
from src.synth import make_world
from src.synth_ext import extend_with_votes
from src.synth2 import make_world2
from src.textworld import build_textworld, to_dataset
from src.text_encoder import get_encoder
from src.verifier import feature_matrix, fold_verifier_matrix

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config_ax3.yaml", encoding="utf-8"))
OUT = ROOT / "eval" / "results"
TIERS = ["fast", "balanced", "premium"]
W = {t: CFG["tiers"][t]["weight"] for t in CFG["tiers"]}


class _Chan:
    """수동 가우시안 채널 — 상자별 예리함(sd)을 직접 지정해 순서 역전을 구성한다."""

    def __init__(self, sds):
        self.sds = list(sds)

    def _sd(self, m):
        return self.sds[0] if m is None or m >= len(self.sds) else self.sds[m]

    def liks_grid(self, v, m=None):
        s = self._sd(m)
        v = np.asarray(v, dtype=float)
        l1 = np.exp(-0.5 * ((v - 1.0) / s) ** 2) / s
        l0 = np.exp(-0.5 * ((v - 0.0) / s) ** 2) / s
        return l1, np.maximum(l0, 1e-12)

    def p_correct_grid(self, v, m=None):
        l1, l0 = self.liks_grid(v, m)
        return l1 / np.maximum(l1 + l0, 1e-12)          # 사전 0.5 기준 보정확률


def analytic():
    """① 해석적 확인 — σ 과대와 순서 역전."""
    print("① 해석 확인 (가우시안 검증기 채널)")
    for sd in (0.2, 0.6, 2.0):
        ch = _Chan([sd])
        p = np.array([0.60])
        t = np.array([0.05])
        s_lat = float(bernoulli_reservation(p, t)[0])
        vals, prob = observable_prize_grid(p, ch)
        s_obs = float(perbox_reservation(t, vals, prob)[0])
        print(f"   verifier sd={sd:<4} p̄=0.60 λc=0.05 →  σ_latent={s_lat:.4f}   "
              f"σ_observable={s_obs:.4f}   (과대 {s_lat - s_obs:+.4f})")
    # 순서 역전: 값이 비슷하나 예리함이 크게 다른 두 상자
    ch = _Chan([0.30, 1.50])
    p = np.array([0.55, 0.58])
    t = np.array([0.12, 0.12])
    s_lat = bernoulli_reservation(p, t)
    vals, prob = observable_prize_grid(p, ch)
    s_obs = perbox_reservation(t, vals, prob)
    print(f"   순서: p̄=[0.55,0.58] sd=[0.30,1.50] λc=0.12")
    print(f"     latent     σ={np.round(s_lat, 4).tolist()} → 개봉 box{int(np.argmax(s_lat))}")
    print(f"     observable σ={np.round(s_obs, 4).tolist()} → 개봉 box{int(np.argmax(s_obs))}")
    flipped = int(np.argmax(s_lat)) != int(np.argmax(s_obs))
    print(f"     순서 역전: {'YES — 잠재 Q 로 풀면 덜 예리한 상자를 고른다' if flipped else 'no'}")
    return {"order_flip_demo": bool(flipped),
            "sigma_gap_sd2.0": round(float(bernoulli_reservation(np.array([0.6]), np.array([0.05]))[0])
                                     - float(perbox_reservation(
                                         np.array([0.05]),
                                         *observable_prize_grid(np.array([0.6]), _Chan([2.0])))[0]), 4)}


def _world(name, ns):
    if name == "textworld":
        tw = build_textworld(CFG, seed=42)
        ds, meta = to_dataset(tw, CFG, Ns=ns)
        _enc = get_encoder("hashing")
        ds.features = _enc.encode(meta["prompts"])
        ds.text_encoder = _enc          # D70: 배포 텍스트 경로 계약
        return ds, feature_matrix(meta)
    base = make_world(CFG, name, seed=CFG["seed"]) if name in ("irt", "specialist") \
        else make_world2(CFG, name, seed=CFG["seed"])
    return extend_with_votes(base, CFG, Ns=ns, conc=0.6 if name == "crossing" else None), None


def score(ds, F, nf, **kw):
    folds = ds.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])
    total = 0.0
    for tier in TIERS:
        qs = []
        for f in range(nf):
            te = folds[f]
            tr = np.setdiff1d(np.arange(ds.n), te)
            if F is not None:
                ds.verifier = fold_verifier_matrix(F, ds.quality, tr)[0]
            r = LPBRouter(CFG, len(np.unique(ds.domains)), seed=f,
                          use_domain=False, **kw).fit(ds, tr, tier)
            qs.append(run_tier(ds, te, r.policy(), tier,
                               tier_budgets(ds, te, CFG)[tier]).mean_quality)
        total += W[tier] * float(np.mean(qs))
    return round(total, 4)


def main(nf=2, worlds=("irt", "textworld"), ns=(1,)):
    t0 = time.time()
    res = {"analytic": analytic(), "ns": list(ns), "n_folds": nf, "worlds": {}}
    print("\n② 종합 점수: latent vs observable 예약값   ③ open_order A/B (인자 연결 후 최초)")
    print(f"   {'world':<12}{'latent':<10}{'observable':<12}{'obs−lat':<10}"
          f"{'obs+value':<11}{'value−sigma':<12}")
    for w in worlds:
        ds, F = _world(w, ns)
        lat = score(ds, F, nf, prize_reservation="latent")
        obs = score(ds, F, nf, prize_reservation="observable")
        val = score(ds, F, nf, prize_reservation="observable", open_order="value")
        res["worlds"][w] = {"latent": lat, "observable": obs, "obs_minus_lat": round(obs - lat, 4),
                            "observable_value_order": val, "value_minus_sigma": round(val - obs, 4)}
        print(f"   {w:<12}{lat:<10.4f}{obs:<12.4f}{obs - lat:<+10.4f}{val:<11.4f}{val - obs:<+12.4f}")
    OUT.mkdir(parents=True, exist_ok=True)
    res["note"] = ("Phase 17. analytic=σ 과대·순서역전 재현. observable=E[Q|v] 분포로 푼 예약값. "
                   "value_minus_sigma 는 open_order 인자 연결(결함 수정) 후 처음 실행된 A/B.")
    json.dump(res, open(OUT / "probe_observable_prize.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n   ({time.time() - t0:.0f}s) → eval/results/probe_observable_prize.json")
    return res


if __name__ == "__main__":
    nf = int(sys.argv[sys.argv.index("--folds") + 1]) if "--folds" in sys.argv else 2
    ws = sys.argv[sys.argv.index("--worlds") + 1].split(",") if "--worlds" in sys.argv \
        else ("irt", "textworld")
    ns = (1, 3, 5) if "--votes" in sys.argv else (1,)
    main(nf, ws, ns)
