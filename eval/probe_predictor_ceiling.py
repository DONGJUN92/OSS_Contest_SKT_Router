"""예측 상한 진단 (Phase 19) — 남은 46% 를 **예측 / 관측 / 결정** 몫으로 분해한다.

왜 이걸 먼저 재는가
--------------------
`eval/probe_redteam_closure.py` 기준 LPB 종합은 0.5051 이고, 같은 세계의 달성 구간은
하한(always-cheapest) 0.3499 → 상한(예산제약 오라클) 0.6365 다. 즉 달성폭 0.2866 중 54% 만
가져갔고 **46% 가 남아 있다.** 그런데 그 46% 가 어디에 있는지 모르고 예측기에 투자하면
Phase 17~18 이 반복한 배분 실수를 또 저지른다 (결정 구조에 노력의 대부분을 쓰고 달성폭의
1.4% 를 다퉜다).

그래서 세 축을 하나씩 **오라클로 치환**해 기여를 분리한다.

  ⓐ **진짜 확률 p̄**  : 신념을 생성식의 참 확률 p(d, cap) 로 치환
                       → **어떤 예측기도 넘을 수 없는 학습 가능 상한** (알레아토릭 잡음 제외)
  ① 실현 정오 p̄     : 신념을 실제 정오(0/1)로 치환
  ② 검증기 오라클    : 관측 v 를 실제 정오로 치환 → "출력을 보고 완벽히 판정"의 상한
  ③ 둘 다 오라클    : 남은 격차 = **결정 구조 자체의 한계**
  ④ 예산제약 오라클  : 이론 상한 (정책이 사후적으로 최적 배정)

★★ **초판의 측정 오류 (반드시 읽을 것)**: 초판은 ⓐ 없이 ①만 재고 그것을 "완벽한 예측기의
상한"이라 불렀다. **①은 답을 아는 것(clairvoyance)이지 확률을 아는 것이 아니다** — 품질이
Bernoulli(p) 로 표집되는 세계에서 어떤 예측기도 ①에 도달할 수 없다. 그 결과 "남은 격차의
82.8% 가 예측 몫"이라는 **잘못된 결론**을 냈고 그에 기반해 투자 권고까지 했다. ⓐ 로 재면
현행 예측기는 이미 학습 가능 상한의 AUC 대비 96% 다.

★ 비대칭 주의: **②(검증기 오라클)는 clairvoyance 가 아니다.** 정오는 출력 텍스트의
결정적 함수이므로 완벽한 채점기는 원리적으로 달성 가능하다. 그래서 ②는 정당한 목표치다.

읽는 법: **ⓐ−현행**이 예측기의 실제 여지, **②−현행**이 검증기의 여지, ③−④ 가 결정 구조다.

사용법: python eval/probe_predictor_ceiling.py [--folds 3] [--n 1200]
"""
import sys, pathlib, json, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from src.textworld import build_textworld, to_dataset
from src.text_encoder import get_encoder
from src.verifier import augmented_feature_matrix, fold_verifier_matrix
from src.harness import run_tier, tier_budgets
from src.cost_mirror import cost_matrix
from src.router import LPBRouter
from src.engine.pandora import (NoiseModel, ExactNoise, FixedBelief, PandoraPolicy,
                                tune_lambda_quality, FINE_MULTS)
from src.engine.pacing import PacedPandora
from baselines.policies import AllModel

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config_ax3.yaml", encoding="utf-8"))
OUT = ROOT / "eval" / "results"
TIERS = ["fast", "balanced", "premium"]
W = {t: CFG["tiers"][t]["weight"] for t in CFG["tiers"]}


class _RowBelief:
    """행별로 미리 정해진 p̄ 를 그대로 내는 신념 (오라클 치환용).

    갱신 불가(FixedBelief 와 동일 성격) — 완벽히 아는 사전에는 갱신할 것이 없다.
    """

    def __init__(self, pbar_row, noise):
        self._p = np.asarray(pbar_row, dtype=float)
        self.noise = noise

    def pbar(self):
        return self._p

    def reservation(self, target):
        from src.engine.prize import bernoulli_reservation
        return bernoulli_reservation(self._p, target)

    def update(self, m, v):
        pass

    def prize(self, v, m=None):
        return self.noise.p_correct(v, m)


def _oracle_policy(ds, lam, noise, rows_lookup):
    """p̄ 오라클 정책 — features 대신 행 인덱스를 신념 생성에 쓴다.

    하네스는 `route(sess, features, domain)` 로 features 만 넘기므로, features 를 키로
    행을 되찾을 수 있게 미리 만든 사전을 쓴다 (평가 전용 — 정책은 이 정보를 배포에서 못 쓴다).

    ★ D35 (짝맞춤 결함 수정): 이 팩토리는 **맨 PandoraPolicy** 를 돌려준다. 초판은 이것을
    그대로 `run_tier` 에 넣어 평가했는데, 비교 대상인 '현행' arm 은 `LPBRouter.policy()`
    = **PacedPandora**(페이싱 ON)였다. 즉 오라클 arm 만 페이싱이 빠진 **짝 안 맞는 대조군**
    이었고, 페이싱은 이득이므로(Phase 6 ablation −0.036) 상한이 과소평가되어 "달성 구간의
    69.7%" 가 실제보다 좋게 나왔다. 또 '예측기 여지 = 0%' 결론에도 페이싱 효과가 섞여 있었다
    (D30 이 고치려던 혼입이 다른 축에 남아 있었다).
    수정: 호출부가 `PacedPandora(fac, lam)` 로 감싼다 — LPBRouter 와 동일하게 **λ 탐색은
    맨 정책으로, 평가는 페이싱으로** 하여 전 arm 이 같은 기구를 쓴다.
    """
    def mb(x, dom):
        return _RowBelief(rows_lookup[x.tobytes()], noise)
    return PandoraPolicy(mb, lam, "oracle-pbar")


def _budget_oracle(ds, cmat, idx, budget):
    """예산 제약 하 배정의 그리디 상한 (probe 설명용 — 이론 상한)."""
    q, c = ds.quality[idx], cmat[idx]
    chosen = np.argmin(c, axis=1)
    spent = c[np.arange(len(idx)), chosen].sum()
    cands = []
    for r in range(len(idx)):
        for m in range(ds.m):
            dq, dc = q[r, m] - q[r, chosen[r]], c[r, m] - c[r, chosen[r]]
            if dq > 0 and dc > 0:
                cands.append((dq / dc, r, m))
    cands.sort(reverse=True)
    for _, r, m in cands:
        dc = c[r, m] - c[r, chosen[r]]
        if dc > 0 and q[r, m] > q[r, chosen[r]] and spent + dc <= budget:
            spent += dc
            chosen[r] = m
    return float(q[np.arange(len(idx)), chosen].mean())


def main(nf=3, n_queries=1200):
    t0 = time.time()
    cfg = dict(CFG, synth=dict(CFG["synth"], n_queries=n_queries))
    tw = build_textworld(cfg, seed=42)
    ds, meta = to_dataset(tw, cfg, Ns=(1,))
    _enc = get_encoder("hashing")
    ds.features = _enc.encode(meta["prompts"])
    ds.text_encoder = _enc          # D70: 배포 텍스트 경로 계약
    F = augmented_feature_matrix(meta, text_dim=64, use_prompt=True, use_agreement=True)
    cmat = cost_matrix(ds)
    folds = ds.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])
    lookup = {ds.features[i].tobytes(): ds.quality[i] for i in range(ds.n)}
    # ⓐ 생성식의 참 확률 — textworld: p_ok = 0.03 + 0.94·σ(7·(cap_m − d_i))
    caps = np.asarray(cfg["synth"]["textworld_caps"], dtype=float)
    dvec = tw.difficulty
    P_true = 0.03 + 0.94 / (1.0 + np.exp(-7.0 * (caps[None, :] - dvec[:, None])))
    lookup_true = {ds.features[i].tobytes(): P_true[i] for i in range(ds.n)}
    print(f"[probe_predictor_ceiling] n={ds.n} M={ds.m} folds={nf}", flush=True)

    arms = ["현행 (실측 p̄ · 실측 검증기)", "ⓐ 진짜확률 p̄ (학습가능 상한)",
            "① 실현정오 p̄ (clairvoyant)", "② 검증기 오라클",
            "★ⓑ 진짜확률 p̄ + 검증기 오라클 (달성 가능 상한)",
            "③ 실현정오+검증기 오라클", "④ 예산제약 오라클", "하한 always-cheapest"]
    res = {a: {} for a in arms}
    for tier in TIERS:
        acc = {a: [] for a in arms}
        for f in range(nf):
            te = folds[f]
            tr = np.setdiff1d(np.arange(ds.n), te)
            V, _ = fold_verifier_matrix(F, ds.quality, tr)
            b_te = tier_budgets(ds, te, CFG)[tier]
            b_tr = tier_budgets(ds, tr, CFG)[tier]

            # 현행
            ds.verifier = V
            r = LPBRouter(CFG, 1, seed=f, use_domain=False).fit(ds, tr, tier)
            acc[arms[0]].append(run_tier(ds, te, r.policy(), tier, b_te).mean_quality)

            noise = NoiseModel().fit(V[tr].ravel(), ds.quality[tr].ravel())
            # ⓐ 진짜 확률 p̄ — 학습 가능 상한 (알레아토릭 잡음은 넘을 수 없다)
            facA = lambda l: _oracle_policy(ds, l, noise, lookup_true)
            lamA = tune_lambda_quality(facA, ds, tr, b_tr, mults=FINE_MULTS, se_k=1.0)
            acc[arms[1]].append(run_tier(ds, te, PacedPandora(facA, lamA), tier,
                                         b_te).mean_quality)

            # ① 실현 정오 p̄ (clairvoyant — 달성 불가, 참조용)
            fac = lambda l: _oracle_policy(ds, l, noise, lookup)
            lam = tune_lambda_quality(fac, ds, tr, b_tr, mults=FINE_MULTS, se_k=1.0)
            acc[arms[2]].append(run_tier(ds, te, PacedPandora(fac, lam), tier,
                                         b_te).mean_quality)

            # ② 검증기 오라클 (신념은 현행 그대로) — clairvoyance 가 아니라 달성 가능 목표
            ds.verifier = ds.quality.astype(float)
            r2 = LPBRouter(CFG, 1, seed=f, use_domain=False,
                           per_model_verifier=False).fit(ds, tr, tier)
            acc[arms[3]].append(run_tier(ds, te, r2.policy(), tier, b_te).mean_quality)

            # ★ⓑ 진짜확률 p̄ + 완벽 채점 = **실제로 달성 가능한 상한**
            # (예측은 확률까지만 알 수 있고, 채점은 출력의 함수라 완벽해질 수 있다)
            facB = lambda l: _oracle_policy(ds, l, ExactNoise(), lookup_true)
            lamB = tune_lambda_quality(facB, ds, tr, b_tr, mults=FINE_MULTS, se_k=1.0)
            acc[arms[4]].append(run_tier(ds, te, PacedPandora(facB, lamB), tier,
                                         b_te).mean_quality)

            # ③ 둘 다 오라클 (clairvoyant 포함 — 참조용)
            fac3 = lambda l: _oracle_policy(ds, l, ExactNoise(), lookup)
            lam3 = tune_lambda_quality(fac3, ds, tr, b_tr, mults=FINE_MULTS, se_k=1.0)
            acc[arms[5]].append(run_tier(ds, te, PacedPandora(fac3, lam3), tier,
                                         b_te).mean_quality)
            ds.verifier = V

            acc[arms[6]].append(_budget_oracle(ds, cmat, te, b_te))
            cheapest = int(np.argmin(cmat[te].mean(axis=0)))
            acc[arms[7]].append(run_tier(ds, te, AllModel(cheapest, "cheap"), tier,
                                         b_te).mean_quality)
        for a in arms:
            res[a][tier] = float(np.mean(acc[a]))
        print(f"  {tier} 완료", flush=True)

    comp = {a: sum(W[t] * res[a][t] for t in TIERS) for a in arms}
    print(f"\n  {'구성':<32}{'fast':<9}{'bal':<9}{'prem':<9}{'종합':<9}{'구간%':<8}")
    lo, hi = comp[arms[7]], comp[arms[6]]
    for a in arms:
        pct = (comp[a] - lo) / max(hi - lo, 1e-9) * 100
        print(f"  {a:<32}{res[a]['fast']:<9.4f}{res[a]['balanced']:<9.4f}"
              f"{res[a]['premium']:<9.4f}{comp[a]:<9.4f}{pct:<8.1f}")

    cur, oA, o1, o2, oB, o3 = (comp[arms[0]], comp[arms[1]], comp[arms[2]],
                               comp[arms[3]], comp[arms[4]], comp[arms[5]])
    span, rem = hi - lo, hi - cur
    # ★ 진짜 기준선: 달성 가능 상한 ⓑ 대비 우리 위치
    ach_span, ach_rem = oB - lo, oB - cur
    print(f"\n  ★★ 달성 가능 구간 (하한 {lo:.4f} → ⓑ {oB:.4f}) 기준 우리 위치 = "
          f"{(cur - lo) / max(ach_span, 1e-9) * 100:.1f}%   남은 여지 {ach_rem:+.4f}")
    print(f"     (명목 구간 {span:.4f} 중 {(oB - lo) / span * 100:.1f}% 만 달성 가능 — "
          f"나머지는 알레아토릭 잡음)")
    print(f"\n  ── 명목 남은 격차 {rem:.4f} (명목 달성폭의 {rem / span * 100:.1f}%) 분해 ──")
    print(f"   ★ 예측기 실제 여지 (진짜확률 상한) : {oA - cur:+.4f}  "
          f"(남은 격차의 {(oA - cur) / rem * 100:5.1f}%)")
    print(f"   ★ 검증기 여지 (달성 가능)          : {o2 - cur:+.4f}  "
          f"({(o2 - cur) / rem * 100:5.1f}%)")
    print(f"     결정 구조 잔차                  : {hi - o3:+.4f}  "
          f"({(hi - o3) / rem * 100:5.1f}%)")
    print(f"     [참조] 실현정오 p̄ (달성 불가)     : {o1 - cur:+.4f}  "
          f"({(o1 - cur) / rem * 100:5.1f}%)  ← 초판이 예측 여지로 오독한 값")
    # ★★ D67 (외부 레드팀 2026-08-01) — **가중 잔여 여지의 tier별 분해.**
    #
    # 지적: 이 저장소는 여지를 *축별*(예측기·검증기·결정구조)로만 쪼갰고 **tier별로는
    # 한 번도 쪼개지 않았다.** 그런데 tier 가중이 0.5/0.3/0.2 이므로 "어디에 투자할 것인가"는
    # 축이 아니라 tier 가 먼저 결정한다. 쪼개 보면 결론이 뒤집힌다:
    #
    #   fast 는 가중치가 절반인데 **가중 잔여 여지는 8% 뿐**이다. 이유는 §2.6 이 이미
    #   보고한 것과 같다 — 빠듯한 예산에서는 "항상 최저가"(하한)와 "참 확률+완벽 채점"
    #   (달성 가능 상한) 사이의 폭 자체가 좁다. 나머지는 알레아토릭이다.
    #
    # 함의: §2.5·§2.6 이 fast tier 퇴화에 쏟은 노력은 **여지 8% 구간에 대한 투자**였다.
    # 그 절들은 "완화해도 점수가 안 오른다"까지는 스스로 측정했지만, **"그러므로
    # balanced·premium 으로 옮긴다"** 는 다음 문장을 쓰지 않았다. 이 표가 그 문장이다.
    tier_gap = {t: (res[arms[4]][t] - res[arms[0]][t]) * W[t] for t in TIERS}
    tot_gap = sum(tier_gap.values())
    print(f"\n  ── ★ 가중 잔여 여지의 tier 분해 (합 {tot_gap:+.4f}) ──")
    for t in TIERS:
        band = (res[arms[4]][t] - res[arms[7]][t]) * W[t]   # 하한→달성상한 가중 폭
        print(f"   {t:<9} 가중 {W[t]:.1f}  현행 {res[arms[0]][t]:.4f} → 상한 "
              f"{res[arms[4]][t]:.4f}  가중여지 {tier_gap[t]:+.4f} "
              f"({tier_gap[t] / max(tot_gap, 1e-9) * 100:5.1f}%)  "
              f"[가중 달성폭 전체 {band:.4f}]")

    out = {"per_tier": {a: {t: round(res[a][t], 4) for t in TIERS} for a in arms},
           "composite": {a: round(comp[a], 4) for a in arms},
           "weighted_headroom_by_tier": {
               t: {"weight": W[t],
                   "current": round(res[arms[0]][t], 4),
                   "achievable_ceiling": round(res[arms[4]][t], 4),
                   "weighted_headroom": round(tier_gap[t], 4),
                   "share_of_total_headroom_pct": round(
                       tier_gap[t] / max(tot_gap, 1e-9) * 100, 1),
                   "weighted_total_band_over_floor": round(
                       (res[arms[4]][t] - res[arms[7]][t]) * W[t], 4)}
               for t in TIERS},
           "span_nominal": round(span, 4), "remaining_nominal": round(rem, 4),
           "achievable_ceiling": round(oB, 4),
           "achievable_span": round(ach_span, 4),
           "achievable_position_pct": round((cur - lo) / max(ach_span, 1e-9) * 100, 1),
           "achievable_remaining": round(ach_rem, 4),
           "aleatoric_share_of_nominal_span": round(1 - (oB - lo) / span, 4),
           "decomposition": {"predictor_headroom_learnable": round(oA - cur, 4),
                             "verifier_headroom_achievable": round(o2 - cur, 4),
                             "decision_structure_residual": round(hi - o3, 4),
                             "clairvoyant_pbar_reference_NOT_achievable": round(o1 - cur, 4)},
           "note": ("Phase 19. A.X 3모델·Q2=No·textworld. ⓐ진짜확률=학습가능 상한, "
                    "①실현정오=clairvoyance(달성 불가). 초판은 ①을 예측 여지로 오독했다.")}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "probe_predictor_ceiling.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n  ({time.time() - t0:.0f}s) → eval/results/probe_predictor_ceiling.json")
    return out


if __name__ == "__main__":
    nf = int(sys.argv[sys.argv.index("--folds") + 1]) if "--folds" in sys.argv else 3
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 1200
    main(nf, n)
