"""하이퍼파라미터 탐색 (B9) — fold 분산 페널티 포함.

**설계 원칙: 탐색 표면적을 늘리지 않는다.** v5 의사결정 로그는 "Optuna 변수 5+개 → 2개"를
분포 이동 방어의 근거로 명시한다. 따라서 이 모듈의 목적은 "더 많이 튜닝하기"가 아니라
**현재 수동 기본값이 이미 좋은지 검증하고, 실데이터에서 재튜닝이 필요할 때 쓸 도구를
준비해 두는 것**이다. 탐색 결과가 기본값을 유의미하게 이기지 못하면 기본값을 유지한다.

목적함수 (분포 이동 방어가 목적이므로 평균만 보지 않는다)
    score = mean(fold 종합점수) − penalty × std(fold 종합점수)
`penalty > 0` 은 "fold 간 편차가 큰 설정은 공개→비공개 이동에서 위험하다"는 성공 기준
③(PROJECT_PLAN §1.2)의 직접 구현이다.

백엔드
  · `builtin` (기본): 의존성 0의 재현 가능한 탐색 — Sobol 유사 저불일치 초기화 후
    상위 표본 주변을 좁혀가는 축차 정제. 시드 고정으로 결정론적이다.
  · `optuna` (선택): 설치돼 있으면 TPE 사용. **없어도 무방** — 챌린지의 오프라인·재현성
    요건 때문에 하드 의존성으로 만들지 않았다.
"""
import itertools
import json

import numpy as np

__all__ = ["SearchSpace", "search", "evaluate_config"]


class SearchSpace:
    """{이름: (저, 고)} 연속 구간 + {이름: [값,…]} 이산 선택."""

    def __init__(self, cont: dict = None, disc: dict = None):
        self.cont = dict(cont or {})
        self.disc = dict(disc or {})

    @property
    def names(self):
        return list(self.cont) + list(self.disc)

    def sample(self, u: np.ndarray) -> dict:
        """단위 정육면체 점 u ∈ [0,1]^d → 설정 딕셔너리."""
        out, i = {}, 0
        for k, (lo, hi) in self.cont.items():
            out[k] = float(lo + (hi - lo) * u[i])
            i += 1
        for k, choices in self.disc.items():
            out[k] = choices[min(int(u[i] * len(choices)), len(choices) - 1)]
            i += 1
        return out

    @property
    def dim(self):
        return len(self.cont) + len(self.disc)


def _halton(n: int, dim: int, skip: int = 13) -> np.ndarray:
    """저불일치 수열 — 무작위 표집보다 공간을 고르게 덮는다 (결정론)."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:dim]
    out = np.empty((n, dim))
    for j, p in enumerate(primes):
        for i in range(n):
            f, r, k = 1.0, 0.0, i + skip
            while k > 0:
                f /= p
                r += f * (k % p)
                k //= p
            out[i, j] = r
    return out


def evaluate_config(objective, cfg: dict, penalty: float) -> tuple:
    """objective(cfg) -> fold별 점수 리스트 → (패널티 적용 점수, 평균, 표준편차)."""
    scores = np.asarray(objective(cfg), dtype=float)
    mu, sd = float(scores.mean()), float(scores.std())
    return mu - penalty * sd, mu, sd


def search(objective, space: SearchSpace, n_trials: int = 24, penalty: float = 1.0,
           seed: int = 42, backend: str = "builtin", baseline: dict = None,
           verbose: bool = True, margin: float | str = "se") -> dict:
    """objective(config) -> fold 점수 리스트. 최대화. 반환: 결과 요약 딕셔너리.

    baseline을 주면 **현재 기본값을 함께 평가**하고, 탐색 최적이 기본값을 이기지
    못하면 `keep_baseline=True`로 표시한다 (이 모듈의 존재 이유 참조).

    margin (채택 문턱): 단순 크기 비교는 **fold 잡음을 승리로 오인한다** — 실제로
    첫 실행에서 탐색이 +0.0024 "우세"했으나 fold std가 0.010이었다. 기본값 "se"는
    기본값 fold 점수의 표준오차(std/√k)를 문턱으로 쓴다. 이보다 작은 이득은
    기각하고 기본값을 유지한다 (탐색 표면적 축소 원칙의 방어선).
    """
    trials = []

    n_folds = [0]                                   # 표준오차 계산용 (첫 평가에서 확정)

    def _run(cfg, tag):
        scores = np.asarray(objective(cfg), dtype=float)
        n_folds[0] = len(scores)
        mu, sd = float(scores.mean()), float(scores.std())
        pen = mu - penalty * sd
        trials.append({"config": cfg, "score": pen, "mean": mu, "std": sd, "tag": tag})
        if verbose:
            desc = " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                            for k, v in cfg.items())
            print(f"    [{tag:<8}] {desc:<52} 평균={mu:.4f} std={sd:.4f} → {pen:.4f}")
        return pen

    base_score = None
    if baseline is not None:
        base_score = _run(dict(baseline), "baseline")

    if backend == "optuna":
        try:
            import optuna
        except ImportError:
            if verbose:
                print("    [경고] optuna 미설치 → builtin 백엔드로 폴백")
            backend = "builtin"

    if backend == "optuna":
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def _obj(trial):
            cfg = {k: trial.suggest_float(k, lo, hi) for k, (lo, hi) in space.cont.items()}
            cfg.update({k: trial.suggest_categorical(k, v) for k, v in space.disc.items()})
            return _run(cfg, "optuna")

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(_obj, n_trials=n_trials)
    else:
        # 1단계: 저불일치 탐색 → 2단계: 상위 표본 주변 축소 재탐색
        n1 = max(n_trials * 2 // 3, 1)
        pts = _halton(n1, space.dim)
        for u in pts:
            _run(space.sample(u), "explore")
        if n_trials > n1 and space.cont:
            best_u = pts[int(np.argmax([t["score"] for t in trials[-n1:]]))]
            rng = np.random.default_rng(seed)
            for _ in range(n_trials - n1):
                u = np.clip(best_u + rng.normal(0, 0.12, size=space.dim), 0, 1)
                _run(space.sample(u), "refine")

    best = max(trials, key=lambda t: t["score"])
    thr = 0.0
    if base_score is not None:
        base_trial = next(t for t in trials if t["tag"] == "baseline")
        thr = (base_trial["std"] / np.sqrt(max(n_folds[0], 1))) if margin == "se" \
            else float(margin)
        keep = best["score"] <= base_score + thr
        if keep:
            best = base_trial
    else:
        keep = False
    return {"best": best, "baseline_score": base_score, "keep_baseline": bool(keep),
            "margin": float(thr), "n_trials": len(trials), "penalty": penalty,
            "backend": backend, "trials": trials}


def save(report: dict, path):
    json.dump(report, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=float)
