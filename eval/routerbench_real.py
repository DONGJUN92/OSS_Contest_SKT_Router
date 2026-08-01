"""실물 RouterBench 실벤치 (Phase 16) — LPB vs cascade vs learned vs oracle/단일모델.

실물 RouterBench 0-shot(HF withmartian/routerbench, 36,497 샘플 × 11 실모델, MMLU/HellaSwag/
GSM8K/ARC/Winogrande/MBPP/MT-Bench). 각 (샘플,모델)에 실제 성능점수·실제 $비용·응답. SKT 과제와
동형(사전계산 출력 선택). LPB 스키마로 적재해 **품질-비용 frontier**로 비교한다.

비용 인코딩: RouterBench의 실제 $비용을 out_tokens에 담아 cost_matrix가 그대로 반환하게 한다
(prefill=0, decode=1e-6, out_tokens=round($×1e9) → cost = out_tokens×1e-9 = $).

★ **관측 조건 2종 (Phase 17 — 레드팀 CRITICAL)**
  `--exact`  (구 기본값): 관측 = 실품질. RouterBench 가 채점된 참 품질을 주므로 **정확 관측
             Pandora** 가 되어 Weitzman 최적 조건이 성립한다. 그러나 **챌린지는 이 조건이
             아니다** — 런타임은 출력 텍스트만 주고 품질은 주지 않는다. 이 모드에서 얻은
             "LPB ≫ learned" 격차는 상당 부분 설정의 산물이며, 단일호출 학습형은 검증기를
             쓰지 않으므로 이 특혜를 받지 못한다.
  `--noisy`  (Phase 17 기본): RouterBench 가 함께 제공하는 **실제 응답 텍스트**
             (`<model>|model_response`)를 자작 `ScoringVerifier` 로 채점해 관측을 만든다.
             챌린지 조건과 동형이며, 이것이 정직한 실벤치 수치다.

또한 이 스크립트의 "LPB" 는 원래 `_irt_lpb` — 페이싱·quality-first λ·PerModelNoise·인증서가
없는 **수제 PandoraPolicy** 였다(측정 대상 ≠ 제출 대상). `--router` 로 실제 `LPBRouter` 를
함께 측정한다.

사용법: python eval/routerbench_real.py [--subset N] [--exact|--noisy] [--router] [--per-bench]
"""
import sys, pathlib, json, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from src.schema import Dataset, ModelMeta
from src.cost_mirror import cost_matrix
from src.harness import run_tier
from src.irt.mml import fit_mml
from src.engine.pandora import (NoiseModel, GridBelief, PandoraPolicy,
                                tune_lambda_replay, tune_lambda_quality, FINE_MULTS)
from src.text_encoder import get_encoder
from phase2_stages import IRTEncoder, DiscLR
from baselines.policies import (StaticCascade, BudgetCascade, CascadeRouting,
                                tune_cascade_tau, LearnedRouter, tune_learned_lambda)

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "results"
MODELS = ["gpt-4-1106-preview", "gpt-3.5-turbo-1106", "claude-v1", "claude-v2",
          "claude-instant-v1", "meta/llama-2-70b-chat", "mistralai/mixtral-8x7b-chat",
          "mistralai/mistral-7b-chat", "zero-one-ai/Yi-34B-Chat",
          "meta/code-llama-instruct-34b-chat", "WizardLM/WizardLM-13B-V1.2"]
SHORT = {m: m.split("/")[-1].replace("-1106-preview", "").replace("-1106", "")
         .replace("-chat", "").replace("-instruct", "")[:16] for m in MODELS}
FRACS = [0.03, 0.06, 0.12, 0.25, 0.50, 1.0]      # budget = frac × (always-max-model total)


def _bench(e):
    e = str(e).lower()
    for k in ["mmlu", "hellaswag", "gsm8k", "arc", "winogrande", "mbpp", "mt-bench", "chinese"]:
        if k in e:
            return k
    if "grade-school" in e:
        return "gsm8k"
    if "mtbench" in e:
        return "mt-bench"
    return "other"


def _rb_meta(df, cols):
    """RouterBench 실제 응답 → verifier 특성용 meta (textworld 의 meta 계약과 동형)."""
    n = len(df)
    rep = [[str(df[f"{m}|model_response"].iloc[i]) for m in cols] for i in range(n)]
    return {"prompts": [str(p) for p in df["prompt"]],
            "tasks": [str(b) for b in df["bench"]],
            "rep_text": rep,
            "vote_share": np.ones((n, len(cols)))}      # 단일 응답 = 자기일관성 신호 없음


def load_rb(subset=None):
    df = pd.read_pickle(hf_hub_download("withmartian/routerbench", "routerbench_0shot.pkl",
                                        repo_type="dataset"))
    df["bench"] = df["eval_name"].map(_bench)
    if subset:
        df = df.sample(subset, random_state=42).reset_index(drop=True)
    N = len(df)
    Q = np.clip(df[MODELS].to_numpy(dtype=float), 0.0, 1.0)
    C = df[[m + "|total_cost" for m in MODELS]].to_numpy(dtype=float)
    out_tokens = np.rint(C * 1e9).astype(np.int64)             # 실제 $비용 인코딩
    bench = df["bench"].to_numpy()
    dnames = sorted(set(bench))
    dom = np.array([dnames.index(b) for b in bench], dtype=np.int64)
    models = {m: ModelMeta(m, 0.0, 1e-6) for m in MODELS}      # prefill 0, decode 1e-6
    ds = Dataset(model_ids=list(MODELS), models=models, domains=dom,
                 features=np.zeros((N, 1)), in_tokens=np.ones(N, dtype=np.int64),
                 out_tokens=out_tokens, quality=Q, verifier=Q.copy(), world="routerbench",
                 extras={"bench": bench})
    meta = {"prompts": [str(p) for p in df["prompt"].tolist()], "bench": bench}
    if all(f"{m}|model_response" in df.columns for m in MODELS):
        meta["verifier_meta"] = _rb_meta(df, list(MODELS))       # --noisy 경로용
    return ds, meta


def _irt_lpb(irt, enc, noise, lam):
    def mb(x, dom):
        mu, s = enc.belief_row(x)
        return GridBelief(mu, s, irt["a"], irt["b"][0], noise)   # 단일집단 b[0]
    return PandoraPolicy(mb, lam, "lpb")


def _irt_lpb_mg(irt, enc, noise, lam):
    """다집단 LPB — 벤치마크별 난이도 b_{d,m} 사용 (RouterBench는 벤치 라벨 제공).
    SKT 런타임엔 도메인이 없어 단일집단이 배포용이나, RouterBench에선 벤치가 있어 공정."""
    def mb(x, dom):
        mu, s = enc.belief_row(x)
        return GridBelief(mu, s, irt["a"], irt["b"][dom], noise)
    return PandoraPolicy(mb, lam, "lpb-mg")


def _submitted_router(ds, tr, tier_frac, cfg_like):
    """★ D56: **제출 대상 그 자체**(`src.router.LPBRouter`)를 이 벤치에 적합한다.

    왜 필요한가 — 이 파일의 `_irt_lpb` 는 IRT+인코더+엔진을 손으로 조립한 **대용 정책**이라
    페이싱(M3)·Platt 재보정·비용 추정·λ quality-first 튜닝·인증서가 빠져 있다. 즉 지금까지의
    실벤치 수치는 **측정 대상 ≠ 제출 대상**이었다. docstring 은 이 사실을 인정하면서
    `--router` 로 실제 라우터를 쓸 수 있다고 적어 뒀는데, 그 플래그는 **본문에서 한 번도
    읽히지 않는 dead code** 였다 (외부 감사 지적). 여기서 실제로 구현한다.

    `LPBRouter` 는 `cfg["tiers"][tier]["budget_frac"]` 로 예산을 잡으므로, 이 벤치의 tier
    정의를 그대로 담은 임시 cfg 를 만들어 넘긴다. 도메인은 단일집단(배포 구성)으로 고정한다.
    """
    from src.router import LPBRouter
    return LPBRouter(cfg_like, 1, seed=0, use_domain=False).fit(ds, tr, tier_frac)


def main(subset=None, per_bench=False, observation="noisy", with_router=False):
    t0 = time.time()
    ds, meta = load_rb(subset)
    _enc = get_encoder("hashing")
    ds.features = _enc.encode(meta["prompts"])
    ds.text_encoder = _enc          # D70: 배포 텍스트 경로 계약
    cmat = cost_matrix(ds)
    N = ds.n
    rng = np.random.default_rng(42)
    perm = rng.permutation(N)
    n_tr = int(0.7 * N)
    tr, te = perm[:n_tr], perm[n_tr:]

    # ★ 관측 채널 (Phase 17): 기본은 챌린지와 동형인 **잡음 관측**.
    v_auc = None
    if observation == "noisy":
        if "verifier_meta" not in meta:
            raise SystemExit("응답 텍스트 컬럼이 없어 --noisy 를 쓸 수 없습니다 (--exact 사용)")
        from src.verifier import augmented_feature_matrix, fold_verifier_matrix
        F = augmented_feature_matrix(meta["verifier_meta"], text_dim=64,
                                     use_prompt=True, use_agreement=True, ref_col=0)
        V, _ = fold_verifier_matrix(F, ds.quality, tr)       # train 라벨만으로 적합
        ds.verifier = V
        s, y = V[te].ravel(), (ds.quality[te] > 0.5).astype(float).ravel()
        pos, neg = s[y > 0.5], s[y <= 0.5]
        r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
        v_auc = float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
        print(f"  [관측=잡음] 실응답 채점 ScoringVerifier held-out AUC = {v_auc:.3f}")
    else:
        print("  [관측=정확] verifier = 실품질 (챌린지 조건 아님 — 상한 참조용)")
    print(f"[routerbench_real] n={N} (train {len(tr)}/test {len(te)}), 11 모델, "
          f"{ds.extras['bench'].__class__.__name__}")

    # 한 번 적합 (예산 무관)
    irt = fit_mml(ds.quality[tr], ds.domains[tr], 1, per_domain_b=False, steps=300)
    enc = IRTEncoder(irt["a"], irt["b"]).fit(ds.features[tr], ds.domains[tr], ds.quality[tr])
    n_dom = len(np.unique(ds.domains))
    irt_mg = fit_mml(ds.quality[tr], ds.domains[tr], n_dom, per_domain_b=True, steps=300)
    enc_mg = IRTEncoder(irt_mg["a"], irt_mg["b"]).fit(ds.features[tr], ds.domains[tr], ds.quality[tr])
    disc = DiscLR().fit(ds.features[tr], ds.domains[tr], ds.quality[tr])
    noise = NoiseModel().fit(ds.verifier[tr].ravel(), ds.quality[tr].ravel())
    order = list(np.argsort(cmat[tr].mean(axis=0)))
    maxtot_te = cmat[te].max(axis=1).sum()

    # 기준점: 단일 모델 · oracle · random
    refs = {}
    for i, m in enumerate(MODELS):
        refs[SHORT[m]] = (float(cmat[te, i].mean()), float(ds.quality[te, i].mean()))
    oracle_q = float(ds.quality[te].max(axis=1).mean())
    oracle_c = float(cmat[te, ds.quality[te].argmax(axis=1)].mean())   # max품질 선택시 비용
    print(f"\n  기준점 (cost$/query, quality):")
    print(f"    {'oracle(max품질)':<20} {oracle_c:.5f}  {oracle_q:.4f}   ← 상한")
    for m in ["gpt-4", "gpt-3.5-turbo", "claude-v2", "mixtral-8x7b", "mistral-7b"]:
        k = [s for s in refs if m in s][0]
        print(f"    {k:<20} {refs[k][0]:.5f}  {refs[k][1]:.4f}")

    # 품질-비용 frontier (예산 sweep)
    print(f"\n  품질-비용 frontier (예산 sweep, cost$=실지출/query):")
    print(f"  {'budget_frac':<12}{'LPB q/cost$':<18}{'cascade q/cost$':<18}{'learned q/cost$':<18}")
    rows = {}
    for frac in FRACS:
        b_te = frac * maxtot_te
        sub = rng.choice(tr, min(5000, len(tr)), replace=False)
        b_sub = frac * cmat[sub].max(axis=1).sum()
        # D49: frontier 도 tier 표와 같은 튜너를 쓴다 (한 스크립트 안에서 두 강도를 섞지 않는다)
        lam = tune_lambda_quality(lambda l: _irt_lpb(irt, enc, noise, l), ds, sub, b_sub,
                                  mults=FINE_MULTS, se_k=1.0)
        r_lpb = run_tier(ds, te, _irt_lpb(irt, enc, noise, lam), "x", b_te)
        # cascade τ 튜닝 (자체 그리드, train sub)
        best_tau, best_q = 0.5, -1
        for tau_c in [0.3, 0.5, 0.7, 0.9]:
            q = run_tier(ds, sub, StaticCascade(order, tau_c), "x", b_sub).mean_quality
            if q > best_q:
                best_q, best_tau = q, tau_c
        r_cas = run_tier(ds, te, StaticCascade(order, best_tau), "x", b_te)
        lam_l = tune_learned_lambda(disc, ds, sub, b_sub)
        r_lrn = run_tier(ds, te, LearnedRouter(disc, lam_l), "x", b_te)
        rows[frac] = {"lpb": (round(r_lpb.mean_quality, 4), round(r_lpb.total_cost / len(te), 6)),
                      "cascade": (round(r_cas.mean_quality, 4), round(r_cas.total_cost / len(te), 6)),
                      "learned": (round(r_lrn.mean_quality, 4), round(r_lrn.total_cost / len(te), 6))}
        print(f"  {frac:<12}{r_lpb.mean_quality:.4f}/{r_lpb.total_cost/len(te):.5f}     "
              f"{r_cas.mean_quality:.4f}/{r_cas.total_cost/len(te):.5f}     "
              f"{r_lrn.mean_quality:.4f}/{r_lrn.total_cost/len(te):.5f}")

    # 챌린지 tier-가중 종합 (fast/balanced/premium = 0.08/0.25/0.60, 가중 0.5/0.3/0.2)
    print(f"\n  챌린지 tier-가중 종합 (fast 0.08 / balanced 0.25 / premium 0.60):")
    tier_fracs = {"fast": 0.08, "balanced": 0.25, "premium": 0.60}
    tw = {"fast": 0.5, "balanced": 0.3, "premium": 0.2}
    # ★ D49 (대등화): 초판은 이 표에서 **LPB 에만 약한 튜너**(`tune_lambda_replay`)를 주고,
    # cascade 자리에는 Phase 17 이 strawman 이라 폐기 선언한 `StaticCascade` 를 세웠으며,
    # LPB 를 앞선다고 보고된 `CascadeRouting` 은 아예 빠져 있었다. 즉 이 저장소의 **유일한
    # 실데이터 실험**이 자기 기준(대등 튜닝 + 예산인지 베이스라인)을 지키지 않았다.
    # 세 가지를 모두 맞춘다: ① 전 정책 동일 튜너 ② BudgetCascade 추가 ③ CascadeRouting 추가.
    keys = ["lpb", "lpb-mg", "cascade", "cascade_budget", "cascade_routing", "learned", "meta"]
    if with_router:
        keys.append("lpb-submitted")
    comp = {k: 0.0 for k in keys}
    # D56: `--router` 는 이제 실제로 동작한다 — 제출 대상 `LPBRouter` 를 tier 마다 적합해
    # 대용 정책과 **나란히** 보고한다 (대체가 아니라 추가 — 두 값의 차이가 곧 "대용 정책이
    # 제출 대상을 얼마나 잘 대변했는가"이므로 그 자체가 정보다).
    cfg_like = {"tiers": {t: {"budget_frac": f, "weight": tw[t]}
                          for t, f in tier_fracs.items()}}
    pols_bal = {}
    meta_picks = {}
    for tname, frac in tier_fracs.items():
        b_te = frac * maxtot_te
        sub = rng.choice(tr, min(5000, len(tr)), replace=False)
        b_sub = frac * cmat[sub].max(axis=1).sum()
        p_lpb = _irt_lpb(irt, enc, noise, tune_lambda_quality(
            lambda l: _irt_lpb(irt, enc, noise, l), ds, sub, b_sub,
            mults=FINE_MULTS, se_k=1.0))
        p_mg = _irt_lpb_mg(irt_mg, enc_mg, noise, tune_lambda_quality(
            lambda l: _irt_lpb_mg(irt_mg, enc_mg, noise, l), ds, sub, b_sub,
            mults=FINE_MULTS, se_k=1.0))
        bt, bq = 0.5, -1
        for tc in [0.3, 0.5, 0.7, 0.9]:
            q = run_tier(ds, sub, StaticCascade(order, tc), "x", b_sub).mean_quality
            if q > bq:
                bq, bt = q, tc
        p_cas = StaticCascade(order, bt)
        bbt, bbq = 0.5, -1
        for tc in [0.3, 0.5, 0.7, 0.9]:
            q = run_tier(ds, sub, BudgetCascade(order, tc), "x", b_sub).mean_quality
            if q > bbq:
                bbq, bbt = q, tc
        p_cbud = BudgetCascade(order, bbt)
        mb_cr = _irt_lpb(irt, enc, noise, 1.0).make_belief          # LPB 와 **같은 신념**
        p_cr = CascadeRouting(mb_cr, tune_lambda_quality(
            lambda l: CascadeRouting(mb_cr, l), ds, sub, b_sub, mults=FINE_MULTS, se_k=1.0))
        p_lrn = LearnedRouter(disc, tune_learned_lambda(disc, ds, sub, b_sub))
        q_lpb = run_tier(ds, te, p_lpb, "x", b_te).mean_quality
        q_mg = run_tier(ds, te, p_mg, "x", b_te).mean_quality
        q_cas = run_tier(ds, te, p_cas, "x", b_te).mean_quality
        q_cbud = run_tier(ds, te, p_cbud, "x", b_te).mean_quality
        q_cr = run_tier(ds, te, p_cr, "x", b_te).mean_quality
        q_lrn = run_tier(ds, te, p_lrn, "x", b_te).mean_quality
        comp["lpb"] += tw[tname] * q_lpb
        comp["lpb-mg"] += tw[tname] * q_mg
        comp["cascade"] += tw[tname] * q_cas
        comp["cascade_budget"] += tw[tname] * q_cbud
        comp["cascade_routing"] += tw[tname] * q_cr
        comp["learned"] += tw[tname] * q_lrn
        q_sub = q_sub_sub = None
        if with_router:                                   # D56: 제출 대상 그 자체
            r_sub = _submitted_router(ds, tr, tname, cfg_like)
            p_sub = r_sub.policy()
            q_sub = run_tier(ds, te, p_sub, tname, b_te).mean_quality
            comp["lpb-submitted"] += tw[tname] * q_sub
            # ★ D59: 메타 선택기 후보에 **제출 대상을 넣는다.** D56 을 고친 뒤에도 메타는
            # 후보 목록에 제출 대상이 없어 learned 를 3/3 골랐다 — 즉 "무엇이 이기는지 데이터가
            # 고르게 한다"는 구조가 **정작 이기는 후보를 볼 수 없었다.** 선택은 train sub 로만
            # 하므로 테스트 미접촉 원칙은 유지된다.
            q_sub_sub = run_tier(ds, sub, p_sub, tname, b_sub).mean_quality
        # per-tier 메타 선택 (train sub 품질로 후보 중 선택, 테스트 미접촉)
        subq = {"lpb-mg": run_tier(ds, sub, p_mg, "x", b_sub).mean_quality, "cascade": bq,
                "cascade_budget": bbq,
                "cascade_routing": run_tier(ds, sub, p_cr, "x", b_sub).mean_quality,
                "learned": run_tier(ds, sub, p_lrn, "x", b_sub).mean_quality}
        test_q = {"lpb-mg": q_mg, "cascade": q_cas, "cascade_budget": q_cbud,
                  "cascade_routing": q_cr, "learned": q_lrn}
        if q_sub_sub is not None:                         # D59: 제출 대상도 후보다
            subq["lpb-submitted"] = q_sub_sub
            test_q["lpb-submitted"] = q_sub
        pick = max(subq, key=subq.get)
        comp["meta"] += tw[tname] * test_q[pick]
        meta_picks[tname] = pick
        print(f"    {tname:<10} LPB {q_lpb:.4f}  LPB-mg {q_mg:.4f}  cascade {q_cas:.4f}  "
              f"cascade-bud {q_cbud:.4f}  cascade-rout {q_cr:.4f}  "
              f"learned {q_lrn:.4f}"
              + (f"  ★제출라우터 {q_sub:.4f}" if q_sub is not None else "")
              + f"   메타선택={pick}")
        if tname == "balanced":
            pols_bal = {"lpb": p_lpb, "lpb-mg": p_mg, "cascade": p_cas, "learned": p_lrn}
    comp = {k: round(v, 4) for k, v in comp.items()}
    print(f"    {'종합':<10} " + "  ".join(f"{k} {v}" for k, v in comp.items()) +
          f"   메타선택={meta_picks}")
    if with_router:
        gap = comp["lpb-submitted"] - comp["lpb"]
        print(f"    ★ 제출 대상 vs 대용 정책: {comp['lpb-submitted']} vs {comp['lpb']} "
              f"({gap:+.4f}) — 이 차이가 곧 '대용 정책이 제출 대상을 대변한 정확도'다")

    # 벤치마크별 품질 (balanced 예산) — 원인 분석용
    print(f"\n  벤치마크별 품질 (balanced 예산): LPB / LPB-mg / cascade / learned / best-single")
    bench_te = ds.extras["bench"][te]
    per_bench = {}
    for b in sorted(set(bench_te)):
        idx = te[bench_te == b]
        if len(idx) < 30:
            continue
        bb = 0.25 * cmat[idx].max(axis=1).sum()
        ql = run_tier(ds, idx, pols_bal["lpb"], "x", bb).mean_quality
        qm = run_tier(ds, idx, pols_bal["lpb-mg"], "x", bb).mean_quality
        qc = run_tier(ds, idx, pols_bal["cascade"], "x", bb).mean_quality
        qr = run_tier(ds, idx, pols_bal["learned"], "x", bb).mean_quality
        bs = float(ds.quality[idx].mean(axis=0).max())
        per_bench[b] = [round(ql, 3), round(qm, 3), round(qc, 3), round(qr, 3), round(bs, 3), len(idx)]
        print(f"    {b:<12} {ql:.3f} / {qm:.3f} / {qc:.3f} / {qr:.3f} / {bs:.3f}  (n={len(idx)})")

    result = {"n": N, "observation": observation, "verifier_auc": v_auc,
              "oracle": [round(oracle_c, 6), oracle_q], "single_models": refs,
              "frontier": rows, "tier_composite": comp, "meta_picks": meta_picks,
              "per_bench": per_bench,
              "note": f"실물 RouterBench 0-shot. 관측={observation} "
                      f"(noisy=실응답을 ScoringVerifier로 채점 = 챌린지 동형 / "
                      f"exact=실품질 관측 = 상한 참조). cost=실제$/query. "
                      f"per_bench=[lpb,lpb-mg,cascade,learned,best-single,n]"}
    OUT.mkdir(parents=True, exist_ok=True)
    # 관측 조건별로 **다른 파일**에 쓴다 — 한 파일을 덮어써서 산문과 아티팩트가 어긋났던
    # Phase 16 의 실수(36k 산문 vs n=8000 JSON)를 구조적으로 막는다.
    fname = f"routerbench_real_{observation}_n{N}.json"
    json.dump(result, open(OUT / fname, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n  ({time.time() - t0:.0f}s) → eval/results/{fname}")
    return result


if __name__ == "__main__":
    subset = int(sys.argv[sys.argv.index("--subset") + 1]) if "--subset" in sys.argv else None
    obs = "exact" if "--exact" in sys.argv else "noisy"
    main(subset, "--per-bench" in sys.argv, obs, "--router" in sys.argv)
