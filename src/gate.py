"""배포 게이트 — **데이터가 배포 정책을 결정한다** (Phase 18).

왜 이 모듈이 필요한가
----------------------
Phase 17 이 확인한 사실: 이 라우터의 순차 개봉·정지 구조가 순가치를 내는지는 **검증기 예리함**
하나로 갈리고, 교차점은 대략 AUC 0.845 다. 그런데 그 상수는 textworld 에서 인위적으로 검증기를
열화시켜 얻은 값이고, 실물 RouterBench(AUC 0.804 → LPB 열세)에서 예측이 맞았을 뿐이다.
**전이된 상수를 결정 근거로 쓰는 것은 Phase 17 이 스스로 비판한 짓과 같다.**

그래서 게이트는 두 층으로 만든다.

  ① 예측층 (값싸다): 받은 데이터로 검증기를 적합해 held-out AUC 를 재고, 임계값과 비교해
     어느 분기가 될지 **예상**한다. 보고·조기경보용이며 결정권은 없다.
  ② 결정층 (권위): 받은 데이터의 **train 안에서** LPB / 학습형 단일호출 / cascade-routing 을
     직접 리플레이 비교해 tier 별로 고른다. 테스트셋은 건드리지 않는다.
     이것이 `baselines.SelectiveRouter` 의 프로토콜과 같은 논리이고, 실제 배포 정책이 된다.

그리고 ①과 ②가 **어긋나면 그 사실을 기록**한다. 어긋남은 버그가 아니라 정보다 — 전이된
임계값이 이 데이터에서 안 맞는다는 증거이고, 다음 번 임계값을 교정할 근거가 된다.

수령 당일 사용법
----------------
    python -m src.gate <데이터경로> --map quality=<컬럼> --config config_ax3.yaml

또는 라이브러리로:

    from src.gate import run_gate
    decision, routers = run_gate(ds, cfg, train_idx, meta=meta)
    # routers = {tier: 배포할 라우터}  ·  decision.record 를 그대로 제출 문서에 싣는다
"""
from dataclasses import dataclass, field, asdict
import json
import pathlib

import numpy as np

# Phase 17 이 보고한 교차점. **결정 근거가 아니라 보고용**이다 (결정층만 권한을 갖는다).
#
# ⚠ D50 (Phase 20 정직성 정정): 이 상수를 만든 스크립트가 **저장소에 없다.**
# `eval/probe_verifier_real.py` 는 출처를 `probe_verifier_sensitivity.py` 라 적었지만 그
# 파일은 존재하지 않고, SUBMISSION §3 표의 인위 열화 3행(AUC 0.507/0.725/0.845)에 대응하는
# 아티팩트도 없다. 즉 **재현 불가능한 상수**다. Phase 17 이 스스로 세운 규칙("산문만 있는
# 수치는 쓰지 않는다")을 코드 상수에도 적용해, 이 값은 로그에 찍는 참고선으로만 남긴다.
# 실데이터에서 배포 정책을 고르는 것은 오직 결정층(직접 리플레이 비교)이다.
BREAK_EVEN_AUC = 0.845
#: 예측층 표기에 쓸 경고 문구 — 임계값의 지위를 코드가 스스로 말하게 한다.
BREAK_EVEN_PROVENANCE = "재현 스크립트 부재 (참고용 · 결정 권한 없음)"
# 대안 정책이 LPB 를 이겼다고 판정하는 **절대** 문턱.
SELECT_MARGIN = 0.01
# ★ 그리고 절대 문턱만으로는 부족하다 (Phase 18 자체 발견). 첫 구현은 margin 0.01 만 봤는데,
# 선택셋 n=240 에서 품질 0.35 의 표준오차는 ≈0.031 이라 **0.0125 차이로 정책을 바꿨다** —
# 정확히 D21(잡음을 승리로 오인)의 재발이었다. 이제 쌍대 차이의 표준오차를 함께 요구한다:
#     대안 채택 조건:  mean(q_alt − q_lpb) > max(SELECT_MARGIN, SE_K × SE(쌍대차이))
# 쌍대(paired)로 재는 이유는 같은 쿼리에 대한 두 정책의 차이가 쿼리 난이도 분산을 상쇄해
# 훨씬 예민하기 때문이다 (같은 표본으로 더 작은 진짜 차이를 잡아낼 수 있다).
SE_K = 1.0
#: 결정층 교차검증 fold 수 (D57). 초판은 단일 70/30 분할이라 유효 표본이 train 의 21% 뿐이었고,
#: 쌍대 SE(0.02~0.045)가 판정 대상 격차(0.003~0.008)보다 3~7배 컸다 — 즉 **분해능이 신호보다
#: 큰 결정 절차**였다. k-fold 로 모든 train 쿼리를 정확히 한 번씩 held-out 으로 쓰면 유효
#: 표본이 100% 가 되어 SE 가 약 √(1/0.21) ≈ 2.2배 줄어든다.
N_SPLITS = 5
#: 교차검증 반복 횟수 (D61). k-fold 는 표본을 키우지만 **분할의 운**은 남는다 —
#: 한 번의 불운한 셔플이 배포 정책을 뒤집을 수 있다. 서로 다른 셔플로 반복하고
#: '반복 과반에서 같은 대안이 이길 것'을 채택 요건에 추가해 그 위험을 없앤다.
N_REPEATS = 3


@dataclass
class GateDecision:
    verifier_auc: float | None
    threshold: float
    predicted: str                       # "lpb" | "alternative"
    per_tier_scores: dict = field(default_factory=dict)
    chosen: dict = field(default_factory=dict)
    agreement: bool | None = None
    notes: list = field(default_factory=list)

    @property
    def deploy(self) -> str:
        """전 tier 가 lpb 면 "lpb", 아니면 "selective" (tier 별로 다른 정책을 쓴다는 뜻)."""
        vals = set(self.chosen.values())
        return "lpb" if vals == {"lpb"} else "selective"

    @property
    def record(self) -> dict:
        d = asdict(self)
        d["deploy"] = self.deploy
        return d


def verifier_auc(scores, labels) -> float:
    s = np.asarray(scores, dtype=float).ravel()
    y = (np.asarray(labels, dtype=float).ravel() > 0.5).astype(float)
    pos, neg = s[y > 0.5], s[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def measure_verifier(ds, meta, train_idx, seed: int = 0, text_dim: int = 64):
    """① 예측층 — 받은 데이터로 검증기를 적합해 held-out AUC 를 잰다.

    meta 가 있으면(출력 텍스트 제공) 증강 특성으로 검증기를 적합한다. meta 가 없으면
    주최측이 준 `ds.verifier` 를 그대로 평가한다 (이미 점수가 오는 경우).
    train 안에서 다시 쪼개므로 테스트 누수가 없다.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(np.asarray(train_idx))
    k = int(0.7 * len(perm))
    fit_idx, ev_idx = perm[:k], perm[k:]
    if meta is None:
        return verifier_auc(ds.verifier[ev_idx], ds.quality[ev_idx]), None, None
    from .verifier import augmented_feature_matrix, fold_verifier_matrix
    F = augmented_feature_matrix(meta, text_dim=text_dim, use_prompt=True,
                                 use_agreement=True, ref_col=0)
    V, sv = fold_verifier_matrix(F, ds.quality, fit_idx)
    # ★ D45: 적합된 `ScoringVerifier` 를 **돌려준다**. 초판은 `sv` 를 버리고 점수 행렬 V 만
    # 반환했다 — 그래서 운영자는 배포 라우터는 받지만 런타임에 출력 텍스트를 점수화할
    # `observe` 콜백을 만들 도구가 없었고, 배포된 신념의 잡음 모델은 **재현 불가능한 관측
    # 채널**에 캘리브레이션돼 있었다. 이제 호출자가 sv 로 observe 를 구성할 수 있다.
    return verifier_auc(V[ev_idx], ds.quality[ev_idx]), V, sv


def _candidates(ds, cfg, idx, tier, seed, lpb_kwargs):
    """tier 별 후보 정책 3종을 idx 로 적합해 돌려준다."""
    from .router import LPBRouter
    from .harness import tier_budgets
    from .encoder import DiscLR
    from .engine.pandora import tune_lambda_quality, FINE_MULTS
    from baselines.policies import (LearnedRouter, tune_learned_lambda, CascadeRouting)
    b = tier_budgets(ds, idx, cfg)[tier]
    n_dom = int(len(np.unique(ds.domains)))
    lpb = LPBRouter(cfg, n_dom, seed=seed, **lpb_kwargs).fit(ds, idx, tier)
    disc = DiscLR().fit(ds.features[idx], ds.domains[idx], ds.quality[idx])
    mb = lpb._factory(1.0).make_belief
    # 대등 튜닝 (Phase 19): 후보 간 λ 탐색을 같은 강도로 — 게이트의 공정성 요건
    lam_cr = tune_lambda_quality(lambda l: CascadeRouting(mb, l), ds, idx, b,
                                 mults=FINE_MULTS, se_k=1.0)
    return {
        "lpb": lpb.policy(),
        "learned": LearnedRouter(disc, tune_learned_lambda(disc, ds, idx, b)),
        "cascade_routing": CascadeRouting(mb, lam_cr),
    }, lpb


def _null_progress(stage: str, info: dict) -> None:                # pragma: no cover
    pass


def make_stderr_progress(total_units: int):
    """진행 표시기 — 수령 당일 운영자가 **끝날지 여부를 알 수 있게** 한다 (D64).

    ★ D64 (외부 레드팀 2026-08-01): "수령 당일 첫 명령"인 `python -m src.gate` 를
    1,800쿼리(9,000행) 데이터에 돌렸더니 **900초가 지나도록 표준출력 한 줄이 없었고
    부분 아티팩트도 없었다.** 구조상 당연하다 — tier × 반복 × fold × 후보 = 수십~수백 회
    라우터 적합인데 `print_decision` 전까지 아무것도 찍지 않고 `out_path` 는 맨 끝에
    한 번만 쓴다. 즉 중간에 끊기면 **전부 잃는다.** 출품 마감(8/27)과 데이터 수령 시점을
    생각하면 이것은 편의가 아니라 리스크 항목이다.
    """
    import sys
    import time
    t0 = time.time()
    state = {"done": 0}

    def progress(stage: str, info: dict) -> None:
        if stage == "fold":
            state["done"] += 1
            el = time.time() - t0
            eta = el / max(state["done"], 1) * max(total_units - state["done"], 0)
            sys.stderr.write(
                f"\r  [{state['done']:>3}/{total_units}] {info.get('tier', ''):<9} "
                f"rep {info.get('rep', 0) + 1} fold {info.get('fold', 0) + 1}  "
                f"경과 {el:6.0f}s  잔여 ~{eta:6.0f}s   ")
            sys.stderr.flush()
        elif stage == "tier_done":
            sys.stderr.write(f"\n  ✓ {info['tier']} → {info['pick']}\n")
            sys.stderr.flush()
        elif stage == "verifier":
            sys.stderr.write(f"  예측층 검증기 AUC = {info['auc']}\n")
            sys.stderr.flush()
    return progress


def run_gate(ds, cfg, train_idx, tiers=None, meta=None, seed: int = 0,
             threshold: float = BREAK_EVEN_AUC, margin: float = SELECT_MARGIN,
             out_path=None, lpb_kwargs=None, apply_verifier: bool = True,
             se_k: float = SE_K, n_splits: int = N_SPLITS,
             n_repeats: int = N_REPEATS, progress=None):
    """게이트 실행 → (GateDecision, {tier: 배포 라우터}).

    apply_verifier=True 이고 meta 가 있으면, 측정 과정에서 적합한 검증기 행렬을 `ds.verifier`
    에 반영한다 (배포 시 실제로 쓸 관측 채널과 평가 채널을 일치시킨다).

    n_splits (D57): 결정층 교차검증 fold 수. 단일 70/30 분할 대비 유효 표본이 k 배가 되어
    쌍대 SE 가 약 √k 배 줄어든다. 비용도 k 배지만 배포 정책을 정하는 단 한 번의 결정이다.

    n_repeats (D61): 서로 다른 셔플로 교차검증을 반복하는 횟수. k-fold 는 표본 크기를
    키우지만 **분할의 운**은 남는다. 대안 채택에 "반복 과반에서 승리" 요건을 추가해
    한 번의 불운한 셔플이 배포 정책을 뒤집는 일을 막는다.
    """
    from .harness import run_tier, tier_budgets
    from .router import LPBRouter
    from baselines.policies import SelectiveRouter
    tiers = list(tiers or cfg["tiers"].keys())
    lpb_kwargs = dict(lpb_kwargs or {})
    progress = progress or _null_progress
    # ★ D44: 배포 가능한 라우터만 만든다. 초판은 `use_domain` 기본값(True)으로 다집단 2PL 을
    # 적합했는데, 챌린지 런타임은 task/도메인 라벨을 주지 않으므로 `SubmissionRouter` 의 D18
    # 가드가 그 라우터를 **거부**한다 — 즉 "수령 당일 첫 명령"이 배포 불가한 산출물을 냈다.
    # 단일집단 퇴화가 배포 구성이므로 그것을 기본값으로 둔다 (호출자가 덮어쓸 수 있다).
    lpb_kwargs.setdefault("use_domain", False)
    train_idx = np.asarray(train_idx)
    notes = []

    # ① 예측층
    auc, V, sv = measure_verifier(ds, meta, train_idx, seed=seed)
    if V is not None and apply_verifier:
        ds.verifier = V
        notes.append("증강 검증기를 ds.verifier 에 반영 (관측 채널 = 배포 채널)")
    predicted = "lpb" if (auc is not None and not np.isnan(auc) and auc >= threshold) \
        else "alternative"
    notes.append(f"예측층: AUC {auc:.4f} vs 임계 {threshold} → {predicted} "
                 f"(결정권 없음; 임계값 출처 = {BREAK_EVEN_PROVENANCE})")
    progress("verifier", {"auc": None if auc is None else round(float(auc), 4)})

    # ② 결정층 — **k-fold 교차검증 쌍대 비교** (D57, Phase 20)
    #
    # 초판은 train 을 70/30 으로 한 번 쪼개 30% 에서만 비교했다. 그 선택셋의 쌍대 SE 는
    # n_sel = 0.21·n 에서 0.02~0.045 인데, 판정해야 할 실제 정책 격차는 0.003~0.008 이다.
    # **분해능이 신호보다 3~7배 컸다** — 외부 감사가 지적하고 `probe_gate.json` 의 사후
    # 선택후회(+0.0050 / +0.0030)로 확인된 결함이다. 특히 RouterBench 분기는 예측층과
    # 결정층이 **일치했는데도 틀렸다**.
    #
    # 수정: 모든 train 쿼리가 정확히 한 번 held-out 이 되도록 k-fold 로 돌린다.
    #   · 유효 표본이 0.21·n → **1.00·n** 으로 늘어 SE 가 약 √(1/0.21) ≈ 2.2배 줄어든다.
    #   · 각 fold 에서 후보를 **그 fold 를 뺀 데이터로 적합**하므로 여전히 out-of-sample 이다.
    #   · 쌍대 차이는 같은 쿼리에 대한 정책 간 차이를 fold 를 가로질러 이어 붙여 만든다.
    # 비용은 k 배지만, 배포 정책을 고르는 단 한 번의 결정이므로 정당하다.
    # ★ D61: **반복 교차검증 + 반복 간 일치 요건.** k-fold 는 표본을 키워 SE 를 줄이지만
    # **분할 자체의 운**은 남는다 — 한 번의 불운한 셔플이 배포 정책을 뒤집을 수 있다.
    # 그래서 서로 다른 셔플로 R 번 반복하고, 대안을 채택하려면 두 조건을 **모두** 넘게 한다:
    #   ① 풀링한 전체 표본에서 격차 > max(margin, se_k·SE)        (크기 요건)
    #   ② 반복 과반에서 같은 대안이 이김                           (안정성 요건)
    # ②가 없으면 "우연히 이긴 한 번"이 정책을 바꾼다 — D21 의 교훈을 분할 축에 적용한 것이다.
    kf = max(2, int(n_splits))
    reps = max(1, int(n_repeats))
    per_tier, chosen = {}, {}
    for tier in tiers:
        pooled: dict[str, list] = {}
        rep_winners = []
        for rep in range(reps):
            rng = np.random.default_rng(1000 + seed + 7919 * rep)   # 반복마다 다른 셔플
            parts = np.array_split(rng.permutation(train_idx), kf)
            acc: dict[str, list] = {}
            for f in range(kf):
                sel = parts[f]
                fit = np.concatenate([parts[g] for g in range(kf) if g != f])
                cands, _ = _candidates(ds, cfg, fit, tier, seed, lpb_kwargs)
                b_sel = tier_budgets(ds, sel, cfg)[tier]
                for name, pol in cands.items():
                    r = run_tier(ds, sel, pol, tier, b_sel)
                    acc.setdefault(name, []).append(np.asarray(r.per_query, dtype=float))
                progress("fold", {"tier": tier, "rep": rep, "fold": f})
            rq = {n: float(np.concatenate(v).mean()) for n, v in acc.items()}
            rep_winners.append(max(rq, key=rq.get))                 # 이 반복의 승자
            for n, v in acc.items():
                pooled.setdefault(n, []).extend(v)
        pq = {n: np.concatenate(v) for n, v in pooled.items()}
        q = {n: float(v.mean()) for n, v in pq.items()}
        best_alt = max((n for n in q if n != "lpb"), key=lambda n: q[n])
        diff = pq[best_alt] - pq["lpb"]
        se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("inf")
        need = max(margin, se_k * se)
        wins = sum(1 for w in rep_winners if w == best_alt)
        stable = wins * 2 > reps                                    # 과반 반복에서 승리
        pick = best_alt if ((q[best_alt] - q["lpb"]) > need and stable) else "lpb"
        per_tier[tier] = {n: round(v, 4) for n, v in q.items()}
        per_tier[tier]["_best_alt"] = best_alt
        per_tier[tier]["_paired_se"] = round(se, 4)
        per_tier[tier]["_required_margin"] = round(need, 4)
        per_tier[tier]["_observed_gap"] = round(q[best_alt] - q["lpb"], 4)
        per_tier[tier]["_n_eval"] = int(len(diff))
        per_tier[tier]["_n_splits"] = kf
        per_tier[tier]["_n_repeats"] = reps
        per_tier[tier]["_rep_winners"] = rep_winners
        per_tier[tier]["_alt_win_repeats"] = f"{wins}/{reps}"
        per_tier[tier]["_stable"] = bool(stable)
        chosen[tier] = pick
        progress("tier_done", {"tier": tier, "pick": pick})
        # ★ D64: tier 하나가 끝날 때마다 중간 판정을 디스크에 남긴다. 끊겨도 이미 끝난
        # tier 의 비교 결과는 살아 있고, 무엇이 남았는지 알 수 있다.
        if out_path:
            _dump({"status": "partial", "chosen_so_far": dict(chosen),
                   "per_tier_scores": per_tier, "verifier_auc": auc,
                   "tiers_remaining": [t for t in tiers if t not in chosen]}, out_path)
    notes.append(f"결정층: {kf}-fold × {reps}회 반복 교차검증. 대안 채택 = 크기(>max(margin,"
                 f"{se_k}×SE)) **및** 안정성(반복 과반 승리) 동시 충족 (D57·D61)")
    dec = GateDecision(verifier_auc=None if auc is None else round(float(auc), 4),
                       threshold=threshold, predicted=predicted,
                       per_tier_scores=per_tier, chosen=chosen, notes=notes)
    dec.agreement = (dec.predicted == "lpb") == (dec.deploy == "lpb")
    if not dec.agreement:
        dec.notes.append(
            "⚠ 예측층과 결정층 불일치 — 전이된 임계값이 이 데이터에서 맞지 않는다. "
            "결정층을 따르고, 이 사건을 임계값 교정 근거로 기록한다 (Phase 17 §2 법칙 검증).")

    # 배포 라우터 구성 — full train 으로 재적합
    routers = {}
    n_dom = int(len(np.unique(ds.domains)))
    for tier in tiers:
        if chosen[tier] == "lpb":
            routers[tier] = LPBRouter(cfg, n_dom, seed=seed, **lpb_kwargs).fit(
                ds, train_idx, tier)
        else:
            routers[tier] = SelectiveRouter(cfg, n_dom, seed=seed, margin=margin,
                                            **lpb_kwargs).fit(ds, train_idx, tier)
        # D45: 런타임 관측 채널을 배포 산출물에 실어 보낸다. 이 검증기가 없으면 운영자는
        # 출력 텍스트를 점수화할 수 없고, 신념의 잡음 모델이 가정한 채널과 실제 채널이 갈린다.
        setattr(routers[tier], "scoring_verifier", sv)
    dec.scoring_verifier = sv
    if out_path:
        _dump(dec.record, out_path)
    return dec, routers


def _dump(obj: dict, path) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    def _plain(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return str(o)
    json.dump(obj, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=_plain)


def print_decision(dec: GateDecision):
    print(f"\n{'=' * 72}\n 배포 게이트 판정\n{'=' * 72}")
    print(f"  검증기 held-out AUC : {dec.verifier_auc}  (임계 {dec.threshold})")
    print(f"  예측층              : {dec.predicted}")
    for tier, q in dec.per_tier_scores.items():
        pol = {k: v for k, v in q.items() if not k.startswith("_")}
        line = "  ".join(f"{k}={v:.4f}" for k, v in pol.items())
        gap, need = q.get("_observed_gap", 0.0), q.get("_required_margin", 0.0)
        verdict = dec.chosen[tier]
        why = "" if verdict != "lpb" or gap <= 0 else \
            f" (최고는 {q.get('_best_alt')}, 격차 {gap:+.4f} ≤ 요구 {need:.4f}=max(절대,SE))"
        print(f"  [{tier:<9}] {line}   → 선택 {verdict}{why}")
    print(f"  결정층 (권위)       : deploy = {dec.deploy}")
    print(f"  예측-결정 일치      : {dec.agreement}")
    for n in dec.notes:
        print(f"    · {n}")


USAGE = """사용법
  python -m src.gate <데이터경로> [옵션]

옵션
  --config <path>       config yaml (기본 config.yaml)
  --map k=v             컬럼명 매핑 (예: --map quality=score). 여러 번 사용 가능
  --quick               3-fold × 1반복 정찰용 (기본 5×3). ★ 먼저 이것으로 파이프라인을
                        확인하고, 시간이 허락하면 기본 설정으로 다시 돌린다
  --splits N            결정층 fold 수 (기본 5)
  --repeats R           반복 교차검증 횟수 (기본 3)
  --encoder <name>      프롬프트 인코더: hashing | hashing:<dim> | st | st:<model>
  --scan-encoders       ★ 인코더 후보를 예측기 품질로 먼저 고른다 (D65, §N5/N6)
  --out <path>          판정 아티팩트 경로 (기본 eval/results/gate_decision.json)
  --quiet               진행 표시 끄기
"""


def _cli():                                                    # pragma: no cover
    import sys
    import yaml
    from .loader import load_dataset, FieldSpec, validate_dataset
    from .text_encoder import get_encoder
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        print(USAGE)
        return

    def opt(name, default=None):
        return args[args.index(name) + 1] if name in args else default

    path = args[0]
    cfg = yaml.safe_load(open(opt("--config", "config.yaml"), encoding="utf-8"))
    spec = FieldSpec()
    for i, a in enumerate(args):
        if a == "--map":
            key, val = args[i + 1].split("=", 1)
            setattr(spec, key, val)
    quick = "--quick" in args
    kf = int(opt("--splits", 3 if quick else N_SPLITS))
    reps = int(opt("--repeats", 1 if quick else N_REPEATS))
    out_path = opt("--out", "eval/results/gate_decision.json")

    ds, meta = load_dataset(path, cfg, spec)
    validate_dataset(ds)
    if getattr(ds, "features", None) is None or ds.features.shape[1] <= 1:
        # D36: 특성을 만든 **그 인코더**를 데이터셋에 실어 둔다 → LPBRouter.fit() 이 물려받고
        # → SubmissionRouter 가 런타임 텍스트 프롬프트를 같은 공간으로 인코딩할 수 있다.
        if "--scan-encoders" in args:
            from .encoder_scan import scan_encoders, default_candidates, apply_encoder
            rep = scan_encoders(meta["prompts"], ds, np.arange(int(0.7 * ds.n)),
                                default_candidates(), verbose="--quiet" not in args)
            apply_encoder(ds, meta["prompts"], rep.best_encoder)
            print(rep.table())
        else:
            # ★ D65: **배포 기본값만** hashing:512 다. `get_encoder("hashing")`(=96) 은
            # 그대로 두어 커밋된 아티팩트 전체가 재현된다 — 전역 기본값을 바꾸면 20 Phase 치
            # 수치가 한꺼번에 무효가 되고, 그 대가로 얻는 것은 합성 세계의 +0.0054 다.
            # 근거(`probe_encoder_axis.json`, 쌍대 3-fold): 제출 대상 selective3 가
            # +0.0054 (요구 0.0050) 로 D21 문턱을 **통과**, 세 tier 전부 같은 부호.
            # 단 LPB 단독은 −0.0013 (fast −2.0×SE) 로 **회귀**한다 — 그래서 이 값은
            # "정답"이 아니라 **기본 후보**이고, `--scan-encoders` 가 실데이터에서 재판정한다.
            enc = get_encoder(opt("--encoder", "hashing:512"))
            ds.features = enc.encode(meta["prompts"])
            ds.text_encoder = enc

    tr = np.arange(int(0.7 * ds.n))
    n_units = len(cfg["tiers"]) * kf * reps
    prog = None if "--quiet" in args else make_stderr_progress(n_units)
    print(f"  결정층 구성: {kf}-fold × {reps}반복 × {len(cfg['tiers'])} tier "
          f"= 라우터 적합 {n_units * 3}회" + ("  (--quick)" if quick else ""))
    dec, routers = run_gate(ds, cfg, tr, meta=meta, out_path=out_path,
                            n_splits=kf, n_repeats=reps, progress=prog)
    print_decision(dec)
    print(f"\n  배포 라우터: {{tier: {type(list(routers.values())[0]).__name__}}} "
          f"→ {out_path}")


if __name__ == "__main__":                                     # pragma: no cover
    _cli()
