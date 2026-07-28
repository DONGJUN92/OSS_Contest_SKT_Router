"""검증기 특성 강화 측정 (권고 #4b) — 태스크-불가지 후보 특성이 held-out AUC를 올리는가?

동기(적대적 평가): 검증기가 전 스택의 병목(phase8: AUC 0.90에서 선택오류 8.3%). 실 도메인
에서 하드코딩 4태스크 구조검사는 더 낮을 수 있다. 그러나 verifier.py를 바로 고치면 textworld
헤드라인이 흔들리므로, **후보 특성을 별도로 붙여 AUC 델타만 먼저 측정**한다. 개선되면 채택
권고, 아니면 phase8 결론(진짜 개선 경로는 실 도메인 구조검사) 재확인.

후보 특성(전부 태스크 무관, 정답 접근 없음): 답의 숫자비율/비영숫자비율/단어수/필러비율/
문자다양성/불확실표지 — 형식 파괴·garble·얼버무림을 태스크 지식 없이 잡으려는 신호.

사용법: python eval/probe_verifier_features.py [--config config.yaml|config_ax3.yaml]
"""
import sys, pathlib, re

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from src.textworld import build_textworld, to_dataset, extract_answer, FILLERS
from src.verifier import feature_matrix, ScoringVerifier

ROOT = pathlib.Path(__file__).resolve().parents[1]
_FILLER_SET = {f.lower() for f in FILLERS}
_UNCERT = re.compile(r"\?|not sure|cannot|maybe|might|possibly|unclear", re.IGNORECASE)


def _auc(y, s):
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _extra_row(prompt, output):
    """태스크-불가지 후보 특성 (6개)."""
    ans = (extract_answer(output) or "").strip()
    out = output or ""
    a_len = max(len(ans), 1)
    digits = sum(c.isdigit() for c in ans) / a_len
    nonalnum = sum((not c.isalnum()) and (not c.isspace()) for c in ans) / a_len
    words = out.lower().split()
    wc = min(len(words) / 60.0, 3.0)
    filler = sum(w.strip(".,") in _FILLER_SET for w in words) / max(len(words), 1)
    distinct = len(set(ans)) / a_len                         # 문자 다양성(반복 낮으면 낮음)
    uncert = float(bool(_UNCERT.search(out)))
    return [digits, nonalnum, wc, filler, distinct, uncert]


def _extra_matrix(meta):
    prompts, rep = meta["prompts"], meta["rep_text"]
    n, cols = np.asarray(meta["vote_share"]).shape
    E = np.zeros((n, cols, 6))
    for i in range(n):
        for c in range(cols):
            E[i, c] = _extra_row(prompts[i], rep[i][c])
    return E


def _fold_auc(F, quality, folds):
    aucs = []
    d = F.shape[2]
    for te in folds:
        tr = np.setdiff1d(np.arange(F.shape[0]), te)
        sv = ScoringVerifier().fit(F[tr].reshape(-1, d), quality[tr].ravel())
        p = sv.score(F[te].reshape(-1, d))
        aucs.append(_auc(quality[te].ravel(), p))
    return float(np.mean(aucs)), float(np.std(aucs))


def main(cfg_path="config.yaml"):
    cfg = yaml.safe_load(open(ROOT / cfg_path, encoding="utf-8"))
    tw = build_textworld(cfg, seed=42)
    ds, meta = to_dataset(tw, cfg)
    folds = ds.stratified_folds(cfg["eval"]["k_folds"], cfg["seed"])

    F_base = feature_matrix(meta)                            # 13 dims (현행)
    F_aug = np.concatenate([F_base, _extra_matrix(meta)], axis=2)   # +6

    a0, s0 = _fold_auc(F_base, ds.quality, folds)
    a1, s1 = _fold_auc(F_aug, ds.quality, folds)
    print(f"\n[probe_verifier_features] textworld held-out AUC (5-fold, config={cfg_path})")
    print(f"  현행 특성({F_base.shape[2]}개)     AUC = {a0:.4f} ±{s0:.4f}")
    print(f"  +후보 특성({F_aug.shape[2]}개)     AUC = {a1:.4f} ±{s1:.4f}")
    print(f"  ΔAUC = {a1 - a0:+.4f}  (표준오차 문턱 {s0 / np.sqrt(len(folds)):.4f})")
    verdict = ("채택 권고" if a1 - a0 > s0 / np.sqrt(len(folds)) else
               "기각 — 표준오차 이하, 진짜 개선은 실 도메인 구조검사(phase8)")
    print(f"  판정: {verdict}")
    return a0, a1


if __name__ == "__main__":
    cfg = "config.yaml"
    if "--config" in sys.argv:
        cfg = sys.argv[sys.argv.index("--config") + 1]
    main(cfg)
