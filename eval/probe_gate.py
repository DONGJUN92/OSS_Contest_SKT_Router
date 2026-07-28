"""배포 게이트를 **양쪽 분기로 실제 통과**시키는 예행연 (Phase 18).

실데이터(SKT)는 아직 없다. 그래서 게이트를 "문서에 적힌 계획"이 아니라 실행되는 코드로
만들었는지 확인하려면, 임계값을 **양쪽으로 걸치는 두 실데이터**로 돌려 보는 것이 최선이다.

  A. textworld + 증강 검증기   — AUC ≈ 0.91  → 예측 PASS(lpb) 쪽
  B. 실물 RouterBench 실응답   — AUC ≈ 0.80  → 예측 FAIL(alternative) 쪽

각 분기에서 확인하는 것
  · 게이트가 끝까지 돌고 배포 라우터 객체를 실제로 만들어 내는가
  · 그 라우터가 **제출 어댑터로 배포 가능**한가 (push 규약으로 유효 액션 열)
  · 예측층과 결정층이 일치하는가 (불일치면 기록 — 임계값 교정 근거)
  · 게이트가 고른 정책이 test 에서 실제로 가장 좋은가 (사후 검증: 선택오류 측정)

사용법: python eval/probe_gate.py [--skip-bench]
"""
import sys, pathlib, json, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from src.gate import run_gate, print_decision
from src.harness import run_tier, tier_budgets
from src.cost_mirror import cost_matrix
from src.submission import SubmissionRouter

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "results"
TIERS = ["fast", "balanced", "premium"]


def _post_hoc(ds, cfg, te, dec, routers):
    """사후 검증 — 게이트가 고른 정책이 test 에서 정말 최선이었나 (선택오류)."""
    from src.gate import _candidates
    w = {t: cfg["tiers"][t]["weight"] for t in TIERS}
    got = regret = 0.0
    detail = {}
    for tier in TIERS:
        b_te = tier_budgets(ds, te, cfg)[tier]
        q_deployed = run_tier(ds, te, routers[tier].policy(), tier, b_te).mean_quality
        cands, _ = _candidates(ds, cfg, np.arange(ds.n)[:0] if False else
                               np.setdiff1d(np.arange(ds.n), te), tier, 0, {"use_domain": False})
        q_all = {n: run_tier(ds, te, p, tier, b_te).mean_quality for n, p in cands.items()}
        best = max(q_all.values())
        detail[tier] = {"deployed": round(q_deployed, 4),
                        "all": {k: round(v, 4) for k, v in q_all.items()},
                        "regret_vs_best": round(best - q_deployed, 4)}
        got += w[tier] * q_deployed
        regret += w[tier] * (best - q_deployed)
    return {"composite_deployed": round(got, 4),
            "composite_regret_vs_oracle_choice": round(regret, 4), "per_tier": detail}


def _adapter_smoke(ds, routers, te, cfg, n=15):
    """배포 라우터가 push 규약으로 유효 액션 열을 내는가 (규칙 불변식 포함)."""
    cmat = cost_matrix(ds)
    tier = "fast"
    budget = tier_budgets(ds, te, cfg)[tier]
    sub = SubmissionRouter({tier: routers[tier]}, ds.model_ids,
                           observe=lambda p, mid, rec: ds.verifier[rec["_row"], rec["_m"]])
    sub.begin_tier(tier, budget=budget, n_queries=len(te))
    kinds, answers = set(), 0
    for i in te[:n]:
        hist, called, spent = [], [], 0.0
        for _ in range(ds.m + 2):
            act = sub.step(ds.features[i], tier, hist, cmat[i], budget - spent, domain=0)
            if act.kind in ("answer", "abstain"):
                if act.kind == "answer":
                    assert act.model_id in called, "규칙 위반: 미호출 모델을 답으로 지명"
                    answers += 1
                break
            m = sub.index[act.model_id]
            assert cmat[i, m] <= budget - spent + 1e-9
            spent += cmat[i, m]
            called.append(act.model_id)
            hist.append({"model_id": act.model_id, "output": None, "_row": i, "_m": m})
        else:
            raise AssertionError("어댑터가 종료하지 않음")
        sub.end_query(spent, tier)
        kinds.add(sub._kind[tier])
    return {"adapter_kind": sorted(kinds), "answered": answers, "queries": min(n, len(te))}


def branch_textworld(nf_seed=0):
    from src.textworld import build_textworld, to_dataset
    from src.text_encoder import get_encoder
    cfg = yaml.safe_load(open(ROOT / "config_ax3.yaml", encoding="utf-8"))
    cfg["synth"]["n_queries"] = 1000
    tw = build_textworld(cfg, seed=42)
    ds, meta = to_dataset(tw, cfg, Ns=(1,))
    ds.features = get_encoder("hashing").encode(meta["prompts"])
    folds = ds.stratified_folds(5, cfg["eval"]["k_folds"] and cfg["seed"])
    te = folds[0]
    tr = np.setdiff1d(np.arange(ds.n), te)
    dec, routers = run_gate(ds, cfg, tr, tiers=TIERS, meta=meta, seed=nf_seed,
                            lpb_kwargs={"use_domain": False})
    return cfg, ds, tr, te, dec, routers


def branch_routerbench(subset=2500):
    sys.path.insert(0, str(ROOT / "eval"))
    from routerbench_real import load_rb
    cfg = yaml.safe_load(open(ROOT / "config_ax3.yaml", encoding="utf-8"))
    ds, meta = load_rb(subset)
    from src.text_encoder import get_encoder
    ds.features = get_encoder("hashing").encode(meta["prompts"])
    # tier 예산 구조는 config 에서 오되 모델 집합은 데이터에서 온다
    cfg = dict(cfg)
    cfg["models"] = {m: {"prefill_price": 0.0, "decode_price": 1e-6} for m in ds.model_ids}
    rng = np.random.default_rng(42)
    perm = rng.permutation(ds.n)
    k = int(0.7 * ds.n)
    tr, te = perm[:k], perm[k:]
    vm = meta.get("verifier_meta")
    dec, routers = run_gate(ds, cfg, tr, tiers=TIERS, meta=vm, seed=0,
                            lpb_kwargs={"use_domain": False})
    return cfg, ds, tr, te, dec, routers


def main(skip_bench=False):
    t0 = time.time()
    result = {}
    for name, fn in (("A_textworld_augmented", branch_textworld),
                     ("B_routerbench_real", branch_routerbench)):
        if name.startswith("B") and skip_bench:
            continue
        print(f"\n{'#' * 72}\n# 분기 {name}\n{'#' * 72}", flush=True)
        cfg, ds, tr, te, dec, routers = fn()
        print_decision(dec)
        post = _post_hoc(ds, cfg, te, dec, routers)
        smoke = _adapter_smoke(ds, routers, te, cfg)
        print(f"  사후 검증: 배포 종합 {post['composite_deployed']}  "
              f"선택 후회(최선 대비) {post['composite_regret_vs_oracle_choice']:+.4f}")
        print(f"  어댑터 스모크: {smoke}")
        result[name] = {"decision": dec.record, "post_hoc": post, "adapter": smoke}
    OUT.mkdir(parents=True, exist_ok=True)
    # ★ D46: `--skip-bench` 로 돌리면 분기 B 가 없는데, 초판은 그 결과로 파일을 **통째로
    # 덮어써서** SUBMISSION §4.5 가 인용하는 표의 절반(실물 RouterBench 분기)을 지웠다.
    # 그리고 `reproduce.py` 의 기본 경로가 정확히 `--skip-bench` 다 — 즉 "원커맨드 재현"이
    # 자기 아티팩트를 파괴했다. 이제 기존 파일의 분기를 **병합**해 보존한다.
    dst = OUT / "probe_gate.json"
    if skip_bench and dst.exists():
        try:
            prev = json.load(open(dst, encoding="utf-8"))
            for k, v in prev.items():
                result.setdefault(k, v)          # 이번에 안 돈 분기는 이전 값을 유지
            result["note_skipped"] = ("--skip-bench 실행: 분기 B 는 이전 실행 값을 보존했다 "
                                      "(재계산하려면 --bench 없이 eval/probe_gate.py 실행).")
        except Exception:                        # 손상된 파일이면 그냥 새로 쓴다
            pass
    result["note"] = ("Phase 18. 실데이터 미수령 상태에서 게이트를 임계값 양쪽 실데이터로 "
                      "실제 통과시킨 예행연. 예측층=AUC, 결정층=train 내 직접 비교(권위).")
    json.dump(result, open(OUT / "probe_gate.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n({time.time() - t0:.0f}s) → eval/results/probe_gate.json")
    return result


if __name__ == "__main__":
    main("--skip-bench" in sys.argv)
