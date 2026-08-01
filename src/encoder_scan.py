"""인코더 선택층 — **예측기 축도 데이터가 고르게 한다** (D65).

왜 이 모듈이 필요한가
----------------------
`src/gate.py` 는 이 저장소의 가장 좋은 아이디어다: "정책은 사람이 아니라 데이터가 고른다."
그런데 외부 레드팀(2026-08-01)이 지적한 대로, **그 원칙이 정작 여지가 큰 축에는 적용되지
않았다.** Phase 20 이 재산출한 잔여 여지 분해는

    검증기 45.8%  ·  결정 구조 13.9%  ·  예측기(p̄) 9.2%

인데, 게이트가 고르는 것은 **결정 구조뿐**이고 인코더는 CLI 에 `hashing` 하드코딩이었다
(`gate.py` 구판 `_cli`). 그리고 Phase 20 은 Phase 19 의 "예측기 축은 닫혔다(0%)" 결론을
**부호까지 뒤집어 철회**(−0.0032 → +0.0123)했으면서, 그 철회에 뒤따르는 코드 경로를
만들지 않았다.

외부 근거도 같은 방향이다 — arXiv:2606.07587 *The Routing Plateau* 는 21개 라우팅 기법이
oracle 에 한참 못 미치는 좁은 대역으로 수렴한다고 보고하고, 그 대역을 넘는 레버로
**① 더 큰 학습 데이터 ② 더 강한 인코더 ③ end-to-end 파인튜닝**을 지목한다. 이 저장소는
셋 중 어느 것도 하지 않았고, ②는 `text_encoder.py` 백엔드 교체로 접근 가능하다.

왜 리플레이가 아니라 예측력으로 재는가
----------------------------------------
인코더는 `p̄` 를 통해서만 정책에 들어간다. 그래서 전 정책 리플레이(게이트 결정층, 후보
3종 × fold × 반복)를 인코더 수만큼 곱하는 대신, **같은 fold 에서 인코더만 바꿔 `p̄` 의
held-out 예측력을 쌍대로** 잰다. 비용이 수십 배 싸고, 축이 하나뿐이라 혼입도 없다.

**판정 지표는 log-loss** 다 (AUC 는 보고용). 이유: `p̄` 는 예약값 `σ` 와 `λ` 비교에 **확률
값 그대로** 들어가므로 순위만 맞으면 되는 AUC 가 아니라 **보정까지 보는 적정 채점 규칙**
(proper scoring rule)이 정책 손실에 가깝다. AUC 만 보고 고르면 잘 정렬됐지만 과신하는
인코더를 뽑을 수 있고, 그것은 `σ = 1 − λc/p̄` 를 직접 왜곡한다.

D21 규칙은 그대로 적용한다 — 기본값(hashing)을 바꾸려면 쌍대 SE 문턱과 절대 문턱을
**둘 다** 넘어야 한다. "인코더를 바꿀 수 있다"가 아니라 "데이터가 바꾸라고 말했을 때만
바꾼다"가 이 모듈의 계약이다.
"""
from dataclasses import dataclass, field

import numpy as np

#: 대안 인코더 채택의 절대 문턱 (평균 log-loss 개선). 게이트의 `SELECT_MARGIN` 과 같은 역할.
LOSS_MARGIN = 0.005
#: 쌍대 표준오차 배수 (게이트 `SE_K` 와 동일 규약).
SE_K = 1.0
#: 교차검증 fold 수.
N_SPLITS = 3


@dataclass
class EncoderScan:
    rows: list = field(default_factory=list)      # [{name, logloss, auc, delta, se, ...}]
    best: str = "hashing"
    baseline: str = "hashing"
    changed: bool = False
    notes: list = field(default_factory=list)
    _encoders: dict = field(default_factory=dict, repr=False)

    @property
    def best_encoder(self):
        return self._encoders[self.best]

    @property
    def record(self) -> dict:
        return {"rows": self.rows, "baseline": self.baseline, "best": self.best,
                "changed": self.changed, "notes": self.notes,
                "loss_margin": LOSS_MARGIN, "se_k": SE_K}

    def table(self) -> str:
        w = max([len(r["name"]) for r in self.rows] + [8])
        out = [f"\n{'=' * 72}\n 인코더 선택층 (p̄ 예측력, {N_SPLITS}-fold 쌍대)\n{'=' * 72}",
               f"  {'인코더':<{w}}  {'log-loss':>9}  {'AUC':>7}  {'Δloss':>9}  {'쌍대SE':>8}  판정"]
        for r in self.rows:
            verdict = ("기준" if r["name"] == self.baseline else
                       ("채택" if r["name"] == self.best and self.changed else "미달"))
            d = "—" if r["delta"] is None else f"{r['delta']:+.4f}"
            s = "—" if r["se"] is None else f"{r['se']:.4f}"
            out.append(f"  {r['name']:<{w}}  {r['logloss']:>9.4f}  {r['auc']:>7.4f}  "
                       f"{d:>9}  {s:>8}  {verdict}")
        for n in self.notes:
            out.append(f"    · {n}")
        return "\n".join(out)


def default_candidates() -> dict:
    """기본 후보군. ST 백엔드는 미설치·미캐시면 `get_encoder` 가 해싱으로 폴백하므로
    그 경우 중복 팔이 생긴다 — `scan_encoders` 가 특성 동일성으로 감지해 제외한다."""
    from .text_encoder import get_encoder
    return {"hashing96": get_encoder("hashing:96"),
            "hashing256": get_encoder("hashing:256"),
            "st-multilingual": get_encoder("st:multilingual")}


def _logloss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _auc(scores, labels) -> float:
    from .gate import verifier_auc
    return verifier_auc(scores, labels)


def _pbar_cv(features, ds, train_idx, n_splits, seed):
    """fold 를 가로질러 이어 붙인 held-out p̄ (쿼리×모델) — fold 분할은 호출자가 고정한다."""
    from .irt.mml import fit_mml
    from .encoder import IRTEncoder
    rng = np.random.default_rng(4242 + seed)
    parts = np.array_split(rng.permutation(np.asarray(train_idx)), n_splits)
    P, Y = [], []
    for f in range(n_splits):
        sel = parts[f]
        fit = np.concatenate([parts[g] for g in range(n_splits) if g != f])
        doms = np.zeros(ds.n, dtype=int)             # 런타임에 도메인이 없다 → 단일집단
        irt = fit_mml(ds.quality[fit], doms[fit], 1, per_domain_b=False)
        enc = IRTEncoder(irt["a"], irt["b"]).fit(features[fit], doms[fit],
                                                 ds.quality[fit], seed=seed)
        P.append(enc.predict(features[sel], doms[sel]))
        Y.append(ds.quality[sel])
    return np.vstack(P), np.vstack(Y)


def scan_encoders(prompts, ds, train_idx, candidates=None, seed: int = 0,
                  n_splits: int = N_SPLITS, margin: float = LOSS_MARGIN,
                  se_k: float = SE_K, verbose: bool = False) -> EncoderScan:
    """인코더 후보를 `p̄` 예측력으로 비교해 하나를 고른다 (테스트셋 미접촉).

    반환: `EncoderScan`. `.best_encoder` 를 `apply_encoder` 에 넘기면 데이터셋에 실린다.
    """
    import sys
    cands = dict(candidates or default_candidates())
    baseline = next(iter(cands))
    scan = EncoderScan(baseline=baseline, best=baseline, _encoders=cands)

    feats, losses = {}, {}
    seen_sig = {}
    for name, enc in list(cands.items()):
        if verbose:
            sys.stderr.write(f"  인코더 {name} 인코딩·적합 중...\n")
            sys.stderr.flush()
        X = np.asarray(enc.encode(list(prompts)), dtype=float)
        # 폴백으로 같은 인코더가 두 번 들어오면 팔이 같아진다 — D55 의 교훈(팔이 같으면
        # "효과 없음"이 아니라 "실험이 없다")을 이 축에도 적용해 즉시 제외한다.
        sig = (X.shape, float(X[:5].sum()))
        if sig in seen_sig:
            scan.notes.append(f"{name} 은 {seen_sig[sig]} 와 특성이 동일 → 제외 "
                              f"(백엔드 폴백으로 추정)")
            cands.pop(name)
            continue
        seen_sig[sig] = name
        feats[name] = X
        P, Y = _pbar_cv(X, ds, train_idx, n_splits, seed)
        losses[name] = (_logloss(P, Y).mean(axis=1), _auc(P, Y))   # 쿼리별 평균 손실

    if baseline not in losses:                       # 기준이 제외됐다면 남은 첫 팔이 기준
        baseline = next(iter(losses))
        scan.baseline = scan.best = baseline
    base_v = losses[baseline][0]
    for name, (v, auc) in losses.items():
        if name == baseline:
            scan.rows.append({"name": name, "logloss": float(v.mean()), "auc": float(auc),
                              "delta": None, "se": None})
            continue
        d = base_v - v                               # 양수 = 손실 감소 = 개선
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("inf")
        scan.rows.append({"name": name, "logloss": float(v.mean()), "auc": float(auc),
                          "delta": float(d.mean()), "se": round(se, 5),
                          "required": round(max(margin, se_k * se), 5)})
    scan.rows.sort(key=lambda r: r["logloss"])

    alts = [r for r in scan.rows if r["name"] != baseline]
    if alts:
        top = min(alts, key=lambda r: r["logloss"])
        if top["delta"] > max(margin, se_k * top["se"]):
            scan.best, scan.changed = top["name"], True
            scan.notes.append(
                f"기본값 변경: {baseline} → {top['name']} "
                f"(Δlog-loss {top['delta']:+.4f} > 요구 {top['required']:.4f})")
        else:
            scan.notes.append(
                f"기본값 유지 ({baseline}) — 최고 대안 {top['name']} 의 개선 "
                f"{top['delta']:+.4f} 이 요구 {top['required']:.4f} 에 미달. "
                f"D21 규칙대로 문턱을 넘지 않으면 어느 방향으로도 바꾸지 않는다")
    scan._encoders = cands
    return scan


def apply_encoder(ds, prompts, encoder):
    """선택된 인코더를 데이터셋에 싣는다 (D36 계약: fit() → 어댑터까지 자동 전파)."""
    ds.features = np.asarray(encoder.encode(list(prompts)), dtype=float)
    ds.text_encoder = encoder
    return ds
