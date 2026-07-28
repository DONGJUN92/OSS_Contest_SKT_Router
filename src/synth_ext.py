"""합성 시뮬레이터 확장: 모델당 다중 샘플 → 투표 상자 (Phase 4a, Phase 3 보류 해제).

연속성 보장: 기존 Dataset의 quality/verifier를 '샘플 1'로 그대로 사용하고,
추가 샘플만 독립 시드(4242)로 생성 — Phase 1~3의 모든 결과와 정합 유지.

투표 상자의 구현 전략: [모델 m × N샘플 다수결]을 '가상 모델' 열로 추가한
확장 Dataset을 만든다. 비용은 prefill×N + decode×(N·out_tokens) = N×단일비용.
→ cost_mirror/harness/engine/IRT 등 기존 인프라가 무변경으로 동작 (설계 이득).

다수결: 정답 표 수 > 최다 오답 표 수 (동률은 오답 처리 — 보수적).
Wu et al. (ICLR 2025) Thm 1·2의 포화 구조 검증은 run_phase4에서 수행.
"""
import numpy as np
from .schema import Dataset, ModelMeta
from .cost_mirror import vote_prefill_mult


def _true_p(ds: Dataset, cfg: dict) -> np.ndarray:
    """세계 생성기의 진짜 정답확률 (생성기 내부 — 정책은 접근 불가)."""
    if "true_p" in ds.extras:                          # Phase 7 신규 세계 호환
        return ds.extras["true_p"]
    if ds.world == "irt":
        th, a, b = ds.extras["theta"], ds.extras["a"], ds.extras["b"]
        return 1 / (1 + np.exp(-a[None, :] * (th[:, None] - b[None, :])))
    skill, diff = ds.extras["skill"], ds.extras["difficulty"]
    slope = cfg["synth"]["specialist"]["difficulty_slope"]
    return np.clip(skill[ds.domains] - slope * (diff[:, None] - 0.5), 0.02, 0.98)


def extend_with_votes(ds: Dataset, cfg: dict, Ns=(1, 3, 5), n_distract: int = 4,
                      seed: int = 4242, conc: float | None = None) -> Dataset:
    """conc (Phase 7, D14 수정): 오답 집중도 — 오답이 확률 conc로 '대표 오답' 하나에
    몰림 (None=균등 분산, 기존 세계 재현성 보존). 실제 LLM 오답은 한 곳에 몰리므로
    conc≈0.6이 더 현실적이며 투표의 만능화를 방지한다."""
    rng = np.random.default_rng(seed)
    N, M = ds.n, ds.m
    S = max(Ns)
    p = _true_p(ds, cfg)

    # 샘플 1 = 기존 quality (연속성). 샘플 2..S만 신규 생성
    correct = np.zeros((N, M, S), dtype=bool)
    correct[:, :, 0] = ds.quality.astype(bool)
    correct[:, :, 1:] = rng.uniform(size=(N, M, S - 1)) < p[:, :, None]
    if conc is None:
        wrong_id = rng.integers(0, n_distract, size=(N, M, S))   # 균등 분산 (기존)
    else:
        to_top = rng.uniform(size=(N, M, S)) < conc              # 대표 오답으로 집중
        others = rng.integers(1, n_distract, size=(N, M, S))
        wrong_id = np.where(to_top, 0, others)

    q_vote = {}
    for n in Ns:
        cnt_ok = correct[:, :, :n].sum(axis=2)
        # D9 수정: 오답 후보별 표 수를 벡터화로 일괄 계산 (동률 수 회계 정확화)
        cnt_wrong = np.stack(
            [((~correct[:, :, :n]) & (wrong_id[:, :, :n] == d)).sum(axis=2)
             for d in range(n_distract)], axis=2)          # (N, M, D)
        max_wrong = cnt_wrong.max(axis=2)
        n_tied = (cnt_wrong == max_wrong[:, :, None]).sum(axis=2)
        win = cnt_ok > max_wrong
        # D8 수정: 동률은 실제 다수결처럼 무작위 해소 (동률 그룹 중 균등 선택)
        tie = (cnt_ok == max_wrong) & (cnt_ok > 0)
        tie_win = rng.uniform(size=(N, M)) < 1.0 / (1.0 + n_tied)
        q_vote[n] = (win | (tie & tie_win)).astype(float)

    # 확장 Dataset 조립: 가상 모델 열 [m × N]
    model_ids, models = [], {}
    qual_cols, out_cols = [], []
    for m, mid in enumerate(ds.model_ids):
        meta = ds.models[mid]
        for n in Ns:
            vid = mid if n == 1 else f"{mid}#v{n}"
            model_ids.append(vid)
            models[vid] = ModelMeta(vid, meta.prefill_price * vote_prefill_mult(cfg, n),
                                    meta.decode_price)
            qual_cols.append(q_vote[n][:, m])
            out_cols.append(ds.out_tokens[:, m] * n)
    quality = np.stack(qual_cols, axis=1)
    out_tokens = np.stack(out_cols, axis=1)
    sigma_v = cfg["synth"]["verifier_noise"]
    v_raw = quality + rng.normal(0, sigma_v, size=quality.shape)
    if "judge_bias" in ds.extras:                      # World C: 심판 편향을 확장 열에도 적용
        bias_cols = np.repeat(ds.extras["judge_bias"], len(Ns))
        v_raw = v_raw + (1 - quality) * bias_cols[None, :]
    verifier = np.clip(v_raw, 0.0, 1.0)
    # 샘플1(N=1) 열은 기존 검증기 관측을 그대로 재사용 (연속성)
    for m in range(M):
        verifier[:, m * len(Ns)] = ds.verifier[:, m]

    return Dataset(
        model_ids=model_ids, models=models, domains=ds.domains, features=ds.features,
        in_tokens=ds.in_tokens, out_tokens=out_tokens, quality=quality,
        verifier=verifier, world=ds.world + "+votes",
        extras={**ds.extras, "Ns": Ns, "base_model_ids": ds.model_ids},
    )
