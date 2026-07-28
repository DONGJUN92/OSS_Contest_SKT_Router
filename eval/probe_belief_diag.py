"""한계 진단 (Phase 15) — LPB가 실텍스트 3모델서 학습형에 진 원인이 belief 예측력인가?

Weitzman 엔진은 belief-agnostic이다(GridBelief=IRT / FixedBelief=판별식 모두 같은 엔진에 꽂힘).
가설: LPB가 진 건 결정 구조가 아니라 **IRT 신념의 예측력이 실텍스트 특성에서 DiscLR보다 약해서**.
그렇다면 **Weitzman + 판별식 신념(Disc-LPB) = 학습형의 예측력 + LPB의 최적 순차 정지**가
learned(단일호출)와 IRT-LPB를 둘 다 이겨야 한다.

비교(3모델 config_ax3, Q2=Yes 투표상자 — learned가 textworld서 이긴 체제):
  · IRT-LPB   : Weitzman + IRT GridBelief (현행 LPB, 단일집단)
  · Disc-LPB  : Weitzman + DiscLR FixedBelief (belief만 교체)
  · learned   : DiscLR 단일호출 (baselines.LearnedRouter)
  + held-out 예측 log-loss: IRT-enc vs DiscLR (예측력 직접 비교)

사용법: python eval/probe_belief_diag.py [--folds N] [--worlds textworld,irt,specialist]
"""
import sys, pathlib, json, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from src.synth import make_world
from src.synth_ext import extend_with_votes
from src.textworld import build_textworld, to_dataset
from src.text_encoder import get_encoder
from src.verifier import feature_matrix, fold_verifier_matrix
from src.harness import run_tier, tier_budgets
from src.cost_mirror import cost_matrix
from src.irt.mml import fit_mml
from src.engine.pandora import NoiseModel, GridBelief, FixedBelief, PandoraPolicy, tune_lambda_replay
from phase2_stages import IRTEncoder, DiscLR
from baselines.policies import LearnedRouter, tune_learned_lambda

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config_ax3.yaml", encoding="utf-8"))
OUT = ROOT / "eval" / "results"
TIERS = ["fast", "balanced", "premium"]
W = {t: CFG["tiers"][t]["weight"] for t in CFG["tiers"]}


def _logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _fit(ds, tr):
    """단일집단 IRT(배포 구성) + 판별식 + 잡음모델."""
    irt = fit_mml(ds.quality[tr], ds.domains[tr], 1, per_domain_b=False)      # 단일집단
    enc = IRTEncoder(irt["a"], irt["b"]).fit(ds.features[tr], ds.domains[tr], ds.quality[tr])
    disc = DiscLR().fit(ds.features[tr], ds.domains[tr], ds.quality[tr])
    noise = NoiseModel().fit(ds.verifier[tr].ravel(), ds.quality[tr].ravel())
    return irt, enc, disc, noise


def _irt_lpb(irt, enc, noise, lam):
    def make_belief(x, dom):
        mu, s = enc.belief_row(x)
        return GridBelief(mu, s, irt["a"], irt["b"][0], noise)               # 단일집단 b[0]
    return PandoraPolicy(make_belief, lam, name="irt-lpb")


def _disc_lpb(disc, noise, lam):
    def make_belief(x, dom):
        return FixedBelief(disc.predict_row(x, dom), noise)
    return PandoraPolicy(make_belief, lam, name="disc-lpb")


class ValueOrderPolicy:
    """개봉 순서를 Weitzman σ가 아니라 **즉시 순가치 p̄−λc**로 하는 변형 (가설: 빠듯한
    예산에서 σ의 옵션가치가 무의미해 단일호출-최적 순가치 순서가 낫다 — knapsack 관점).
    정지 규칙은 Weitzman 그대로(관측 최고 상금 ≥ 잔여 최대 σ)."""

    def __init__(self, make_belief, lam, name="value-lpb"):
        self.make_belief, self.lam, self.name = make_belief, lam, name

    def route(self, sess, features, domain) -> int:
        bel = self.make_belief(features, domain)
        M = len(sess.costs)
        unopened = set(range(M))
        obs = {}
        costs = np.asarray(sess.costs, dtype=float)
        while unopened:
            afford = [m for m in unopened if sess.can_afford(m)]
            if not afford:
                break
            pbar = bel.pbar()
            m_star = max(afford, key=lambda m: pbar[m] - self.lam * costs[m])   # 즉시 순가치
            res = bel.reservation(self.lam * costs)
            if obs and max(bel.prize(v, m) for m, v in obs.items()) >= max(res[m] for m in afford):
                break
            v = sess.call(m_star)
            if v is None:
                break
            unopened.discard(m_star)
            obs[m_star] = v
            bel.update(m_star, v)
        if not obs:
            m = int(np.argmin(costs))
            v = sess.call(m)
            if v is not None:
                obs[m] = v
        if not obs:
            return 0
        return max(obs, key=lambda m: bel.prize(obs[m], m))


def _value_lpb(irt, enc, noise, lam):
    def make_belief(x, dom):
        mu, s = enc.belief_row(x)
        return GridBelief(mu, s, irt["a"], irt["b"][0], noise)
    return ValueOrderPolicy(make_belief, lam, name="value-lpb")


def _build(wname):
    if wname == "textworld":
        tw = build_textworld(CFG, seed=42)
        ds, meta = to_dataset(tw, CFG)
        ds.features = get_encoder("hashing").encode(meta["prompts"])
        F = feature_matrix(meta)
        folds = ds.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])
        return ds, F, folds
    base = make_world(CFG, wname, seed=CFG["seed"])
    eds = extend_with_votes(base, CFG)
    folds = base.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])
    return eds, None, folds


def _run(wname, nf):
    ds, F, folds = _build(wname)
    res = {t: {"irt-lpb": [], "disc-lpb": [], "value-lpb": [], "learned": []} for t in TIERS}
    ll_irt, ll_disc = [], []
    for f in range(nf):
        te = folds[f]
        tr = np.setdiff1d(np.arange(ds.n), te)
        if F is not None:                                    # textworld: verifier per-fold
            V, _ = fold_verifier_matrix(F, ds.quality, tr)
            ds.verifier = V
        irt, enc, disc, noise = _fit(ds, tr)
        # 예측력 직접 비교 (held-out log-loss, 전 열 평균)
        ll_irt.append(_logloss(enc.predict(ds.features[te], ds.domains[te]), ds.quality[te]))
        ll_disc.append(_logloss(disc.predict(ds.features[te], ds.domains[te]), ds.quality[te]))
        for tier in TIERS:
            b_te = tier_budgets(ds, te, CFG)[tier]
            b_tr = tier_budgets(ds, tr, CFG)[tier]
            lam_i = tune_lambda_replay(lambda l: _irt_lpb(irt, enc, noise, l), ds, tr, b_tr)
            lam_d = tune_lambda_replay(lambda l: _disc_lpb(disc, noise, l), ds, tr, b_tr)
            lam_v = tune_lambda_replay(lambda l: _value_lpb(irt, enc, noise, l), ds, tr, b_tr)
            lam_l = tune_learned_lambda(disc, ds, tr, b_tr)
            res[tier]["irt-lpb"].append(run_tier(ds, te, _irt_lpb(irt, enc, noise, lam_i), tier, b_te).mean_quality)
            res[tier]["disc-lpb"].append(run_tier(ds, te, _disc_lpb(disc, noise, lam_d), tier, b_te).mean_quality)
            res[tier]["value-lpb"].append(run_tier(ds, te, _value_lpb(irt, enc, noise, lam_v), tier, b_te).mean_quality)
            res[tier]["learned"].append(run_tier(ds, te, LearnedRouter(disc, lam_l), tier, b_te).mean_quality)
    comp = {p: sum(W[t] * float(np.mean(res[t][p])) for t in TIERS)
            for p in ("irt-lpb", "disc-lpb", "value-lpb", "learned")}
    return comp, float(np.mean(ll_irt)), float(np.mean(ll_disc))


def main(nf=5, which=None):
    which = which or ["textworld", "irt", "specialist"]
    t0 = time.time()
    print(f"\n[probe_belief_diag] 3모델 — belief 예측력이 원인인가 (nf={nf})")
    print(f"  {'world':<11}{'IRT-LPB':<9}{'Disc-LPB':<9}{'value-LPB':<10}{'learned':<9}"
          f"{'val−learn':<11}{'val−irt':<10}{'ll_irt/disc':<14}")
    summary = {}
    for wname in which:
        comp, lli, lld = _run(wname, nf)
        summary[wname] = {**{k: round(v, 4) for k, v in comp.items()},
                          "value_minus_learned": round(comp["value-lpb"] - comp["learned"], 4),
                          "value_minus_irt": round(comp["value-lpb"] - comp["irt-lpb"], 4),
                          "disc_minus_irt": round(comp["disc-lpb"] - comp["irt-lpb"], 4),
                          "logloss_irt": round(lli, 4), "logloss_disc": round(lld, 4)}
        print(f"  {wname:<11}{comp['irt-lpb']:<9.4f}{comp['disc-lpb']:<9.4f}{comp['value-lpb']:<10.4f}"
              f"{comp['learned']:<9.4f}{comp['value-lpb'] - comp['learned']:<+11.4f}"
              f"{comp['value-lpb'] - comp['irt-lpb']:<+10.4f}{lli:.3f}/{lld:.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"n_folds": nf, "summary": summary,
               "note": "Disc-LPB>learned ∧ Disc-LPB>IRT-LPB 이면 원인=belief 예측력 → 개선=적응 신념"},
              open(OUT / "probe_belief_diag.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("  → eval/results/probe_belief_diag.json")
    return summary


if __name__ == "__main__":
    nf = 5
    which = None
    if "--folds" in sys.argv:
        nf = int(sys.argv[sys.argv.index("--folds") + 1])
    if "--worlds" in sys.argv:
        which = sys.argv[sys.argv.index("--worlds") + 1].split(",")
    main(nf, which)
