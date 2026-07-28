"""3분 시연용 데모 — 라우터가 무엇을 하는지 한 화면에 보여준다 (대회 필수 산출물 대응).

`python demo.py` 만 실행하면 외부 데이터·네트워크 없이 다음을 순서대로 출력한다.
  1) 후보 모델 3종(A.X 사다리 대리값)과 tier 예산
  2) 쿼리 하나에 대한 **의사결정 추적**: 예약지수 σ, 어떤 상자를 왜 열었는지, 언제 멈췄는지
  3) tier별 종합 성적과 **호출/쿼리** — fast tier 에서 1회 호출로 퇴화하는 사실을 그대로 보여줌
  4) 비교군(학습형 단일호출 · 예산인지 cascade · cascade-routing) 대비 위치
  5) M4 인증서와 σ<0 체제 진단

시연 스크립트(내레이션)는 docs/demo_script.md 참조.
"""
import sys, pathlib, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import yaml

from src.textworld import build_textworld, to_dataset
from src.text_encoder import get_encoder
from src.verifier import augmented_feature_matrix, fold_verifier_matrix
from src.harness import run_tier, tier_budgets, Session
from src.cost_mirror import cost_matrix
from src.router import LPBRouter
from src.encoder import DiscLR
from baselines.policies import (LearnedRouter, tune_learned_lambda, BudgetCascade,
                                tune_budget_cascade_tau, CascadeRouting)
from src.engine.pandora import tune_lambda_quality, FINE_MULTS

ROOT = pathlib.Path(__file__).resolve().parent
CFG = yaml.safe_load(open(ROOT / "config_ax3.yaml", encoding="utf-8"))
CFG["synth"]["n_queries"] = 600
TIERS = ["fast", "balanced", "premium"]
W = {t: CFG["tiers"][t]["weight"] for t in CFG["tiers"]}
BAR = "─" * 78


def main():
    t0 = time.time()
    print(f"\n{BAR}\n LPB-Router 데모 — 쉬운 문제는 싸게, 어려운 문제만 비싸게\n{BAR}")
    print(" 후보 모델 (A.X 사다리 대리값, 1k 토큰당 가격):")
    for mid, m in CFG["models"].items():
        print(f"   {mid:<16} prefill {m['prefill_price']:<6} decode {m['decode_price']:<6}"
              f" 평균출력 {m['avg_out_tokens']}토큰")
    print(" 예산 tier:", ", ".join(
        f"{t} = 최대모델 전량호출의 {v['budget_frac']:.0%} (가중 {v['weight']})"
        for t, v in CFG["tiers"].items()))

    tw = build_textworld(CFG, seed=42)
    ds, meta = to_dataset(tw, CFG, Ns=(1,))            # 실전 조건: 모델당 출력 1개
    ds.features = get_encoder("hashing").encode(meta["prompts"])
    F = augmented_feature_matrix(meta, text_dim=64, use_prompt=True, use_agreement=True)
    folds = ds.stratified_folds(5, CFG["seed"])
    te = folds[0]
    tr = np.setdiff1d(np.arange(ds.n), te)
    ds.verifier, _ = fold_verifier_matrix(F, ds.quality, tr)
    cmat = cost_matrix(ds)
    order = list(np.argsort(cmat.mean(axis=0)))

    print(f"\n{BAR}\n [1] 단일 쿼리 의사결정 추적 (balanced tier)\n{BAR}")
    r = LPBRouter(CFG, 1, seed=0, use_domain=False).fit(ds, tr, "balanced")
    b = tier_budgets(ds, te, CFG)["balanced"]
    i = int(te[0])
    print(f" 프롬프트: {meta['prompts'][i][:96]}...")
    pol = r._factory(r.lam0)
    bel = pol.make_belief(ds.features[i], 0)
    sig = bel.reservation(r.lam0 * cmat[i])
    print(f" λ₀ = {r.lam0:.4g}")
    for m, mid in enumerate(ds.model_ids):
        print(f"   {mid:<16} 비용 {cmat[i, m]:.5f}  예상정답률 p̄ {bel.pbar()[m]:.3f}"
              f"  예약지수 σ {sig[m]:+.4f}")
    sess = Session(costs=cmat[i], verifier_row=ds.verifier[i], remaining_budget=b)
    choice = pol.route(sess, ds.features[i], 0)
    print(f" → 실제로 연 상자: {[ds.model_ids[m] for m in sess.called]}")
    print(f" → 최종 답: {ds.model_ids[choice]}  (진짜 품질 {ds.quality[i, choice]:.0f}, "
          f"지출 {sess.spent:.5f})")
    print(f" 해석: σ 가 모두 음수면 정지 조건이 첫 관측 직후 성립 → 1회 호출로 퇴화한다.")

    print(f"\n{BAR}\n [2] tier별 성적과 호출 수\n{BAR}")
    print(f" {'tier':<10}{'LPB 품질':<11}{'호출/쿼리':<11}{'σ<0 비율':<11}{'무응답':<8}")
    routers = {}
    for tier in TIERS:
        rr = LPBRouter(CFG, 1, seed=0, use_domain=False).fit(ds, tr, tier)
        routers[tier] = rr
        out = run_tier(ds, te, rr.policy(), tier, tier_budgets(ds, te, CFG)[tier])
        print(f" {tier:<10}{out.mean_quality:<11.4f}{out.calls_per_query:<11.2f}"
              f"{rr.diagnostics['sigma_neg_frac']:<11.3f}{out.unanswered:<8}")

    print(f"\n{BAR}\n [3] 비교군 대비 (tier 가중 종합)\n{BAR}")
    comp = {}
    for name in ("LPB", "learned(RouteLLM류)", "cascade(예산인지)", "cascade-routing"):
        comp[name] = 0.0
    for tier in TIERS:
        b_te = tier_budgets(ds, te, CFG)[tier]
        b_tr = tier_budgets(ds, tr, CFG)[tier]
        comp["LPB"] += W[tier] * run_tier(ds, te, routers[tier].policy(), tier, b_te).mean_quality
        disc = DiscLR().fit(ds.features[tr], ds.domains[tr], ds.quality[tr])
        comp["learned(RouteLLM류)"] += W[tier] * run_tier(
            ds, te, LearnedRouter(disc, tune_learned_lambda(disc, ds, tr, b_tr)), tier, b_te).mean_quality
        comp["cascade(예산인지)"] += W[tier] * run_tier(
            ds, te, BudgetCascade(order, tune_budget_cascade_tau(ds, tr, CFG, tier, b_tr)),
            tier, b_te).mean_quality
        mb = routers[tier]._factory(1.0).make_belief
        # ★ D48 대등 튜닝: 초판 데모는 이 비교군에만 약한 `tune_lambda_replay`(예산을 채우는
        # 최소 λ)를 주고 LPB 에는 `tune_lambda_quality`(세밀격자 + SE 문턱)를 줬다. Phase 19 가
        # 헤드라인 프로브·게이트·SelectiveRouter 세 곳은 공정화했지만 **데모는 빠뜨렸고**,
        # 하필 데모가 심사위원이 보는 시연영상의 촬영 대상이다. 같은 튜너를 준다.
        lam = tune_lambda_quality(lambda l: CascadeRouting(mb, l), ds, tr, b_tr,
                                  mults=FINE_MULTS, se_k=1.0)
        comp["cascade-routing"] += W[tier] * run_tier(
            ds, te, CascadeRouting(mb, lam), tier, b_te).mean_quality
    for k, v in sorted(comp.items(), key=lambda kv: -kv[1]):
        print(f" {k:<24}{v:.4f}")

    print(f"\n{BAR}\n [4] M4 인증서 (인증 전용 split, λ 선택과 분리)\n{BAR}")
    c = routers["fast"].certificate
    print(f" 조기정지 후회율 ≤ {c['risk_upper']:.4f} @ {c['confidence']:.0%} (n={c['n_cal']})")
    print(f" {c['note']}")
    print(f"\n ({time.time() - t0:.0f}s) 데모 종료. 전체 재현: python reproduce.py\n")


if __name__ == "__main__":
    main()
