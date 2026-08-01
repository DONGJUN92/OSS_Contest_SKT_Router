"""라우터 결정 지연 측정 (Phase 21) — **tie-break 축인데 수치가 없었다**.

챌린지 평가 방식: "동일 점수일 경우 latency 를 tie-break 로 활용." 즉 점수에는 안 들어가지만
**동점 시 순위를 가르는 유일한 기준**이다. 그런데 이 저장소는 20 Phase 동안 지연을 한 번도
측정해 보고하지 않았다 — 외부 레드팀(2026-08-01) 지적.

재는 것은 **라우터의 결정 시간**이다 (모델 호출 시간이 아니다 — 그것은 호스트 몫이고
후보 모델의 성질이지 라우터의 성질이 아니다). 배포 규약대로 `SubmissionRouter.step()` 을
호스트처럼 밀어서 잰다.

  · step 지연  : 액션 하나를 결정하는 시간 (호스트가 매 스텝 기다리는 시간)
  · 문항 지연  : 한 문항을 끝내는 데 든 라우터 시간 총합 (= step × 스텝 수)
  · 적합 시간  : 오프라인 1회 (채점 중에는 발생하지 않는다 — 구분해 보고)

비교군도 같은 방식으로 잰다. 순차 개봉은 정의상 스텝이 더 많으므로 **문항당 지연이
단일호출보다 크다** — 그것을 숨기지 않고 그대로 싣는 것이 이 프로브의 목적이다.

사용법: python eval/probe_latency.py [--n 600] [--reps 3]
"""
import sys, pathlib, json, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from src.textworld import build_textworld, to_dataset
from src.text_encoder import get_encoder
from src.verifier import augmented_feature_matrix, fold_verifier_matrix
from src.harness import tier_budgets
from src.cost_mirror import cost_matrix
from src.router import LPBRouter
from src.submission import SubmissionRouter

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config_ax3.yaml", encoding="utf-8"))
OUT = ROOT / "eval" / "results"
TIERS = ["fast", "balanced", "premium"]


def _drive(sub, tier, ds, idx, cmat, budget, meta_fn, prompts):
    """호스트 루프를 돌리며 **step 호출 시간만** 누적한다 (검증기·과금 시간은 제외)."""
    steps, per_step, per_query, remaining = 0, [], [], budget
    for i in idx:
        i = int(i)
        hist, spent, t_q = [], 0.0, 0.0
        for _ in range(12):
            t0 = time.perf_counter()
            act = sub.step(prompts[i], tier, hist, meta_fn(i),
                           remaining_budget=remaining - spent)
            dt = time.perf_counter() - t0
            per_step.append(dt)
            t_q += dt
            steps += 1
            if act.kind != "call":
                break
            m = sub.index[act.model_id]
            spent += float(cmat[i, m])
            hist.append({"model_id": act.model_id, "v": float(ds.verifier[i, m])})
        per_query.append(t_q)
        remaining -= spent
        sub.end_query(spent, tier)
    return np.asarray(per_step), np.asarray(per_query), steps


def main(n_queries=600, reps=3):
    t0 = time.time()
    cfg = dict(CFG, synth=dict(CFG["synth"], n_queries=n_queries))
    tw = build_textworld(cfg, seed=42)
    ds, meta = to_dataset(tw, cfg, Ns=(1,))
    enc = get_encoder("hashing")
    ds.features = enc.encode(meta["prompts"])
    ds.text_encoder = enc
    F = augmented_feature_matrix(meta, text_dim=64, use_prompt=True, use_agreement=True)
    folds = ds.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])
    te = folds[0]
    tr = np.setdiff1d(np.arange(ds.n), te)
    ds.verifier, _ = fold_verifier_matrix(F, ds.quality, tr)
    cmat = cost_matrix(ds)
    print(f"[probe_latency] n={ds.n} M={ds.m} 평가 {len(te)}쿼리 × {reps}회", flush=True)

    res = {}
    for tier in TIERS:
        t_fit = time.perf_counter()
        r = LPBRouter(CFG, 1, seed=0, use_domain=False).fit(ds, tr, tier)
        fit_s = time.perf_counter() - t_fit
        b = tier_budgets(ds, te, CFG)[tier]
        runs = []
        for _ in range(reps):
            sub = SubmissionRouter({tier: r}, list(ds.model_ids),
                                   observe=lambda p, mid, rec: float(rec["v"]))
            sub.begin_tier(tier, budget=b, n_queries=len(te))
            ps, pq, steps = _drive(sub, tier, ds, te, cmat, b,
                                   lambda i: cmat[i], meta["prompts"])
            runs.append((ps, pq, steps))
        ps = np.concatenate([x[0] for x in runs])
        pq = np.concatenate([x[1] for x in runs])
        res[tier] = {
            "step_ms_mean": round(float(ps.mean() * 1e3), 4),
            "step_ms_p50": round(float(np.percentile(ps, 50) * 1e3), 4),
            "step_ms_p99": round(float(np.percentile(ps, 99) * 1e3), 4),
            "query_ms_mean": round(float(pq.mean() * 1e3), 4),
            "query_ms_p99": round(float(np.percentile(pq, 99) * 1e3), 4),
            "steps_per_query": round(runs[0][2] / len(te), 3),
            "offline_fit_s": round(fit_s, 2),
        }
        print(f"  {tier:<9} step {res[tier]['step_ms_mean']:.3f} ms "
              f"(p99 {res[tier]['step_ms_p99']:.3f})  문항 "
              f"{res[tier]['query_ms_mean']:.3f} ms  스텝/문항 "
              f"{res[tier]['steps_per_query']:.2f}  [오프라인 적합 {fit_s:.1f}s]")

    w = {t: CFG["tiers"][t]["weight"] for t in CFG["tiers"]}
    res["weighted_query_ms"] = round(sum(w[t] * res[t]["query_ms_mean"] for t in TIERS), 4)
    res["note"] = (
        "라우터의 **결정** 시간만 잰다 (모델 호출·채점 시간 제외 — 호스트 몫). "
        "오프라인 적합은 채점 중에 발생하지 않으므로 분리해 보고한다. "
        "순차 개봉은 스텝이 많아 문항당 지연이 단일호출보다 크다 — 그 대가를 그대로 싣는다. "
        "측정 환경 의존이므로 절대값이 아니라 규모(µs~ms)로 읽을 것.")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "probe_latency.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"  가중 문항 지연 {res['weighted_query_ms']:.3f} ms "
          f"({time.time() - t0:.0f}s) → eval/results/probe_latency.json")
    return res


if __name__ == "__main__":
    def _arg(name, d):
        return int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else d
    main(_arg("--n", 600), _arg("--reps", 3))
