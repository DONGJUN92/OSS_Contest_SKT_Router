"""B4 사전 조사: 예약지수 닫힌형의 λc > p̄ 구간 정합성 점검.

Weitzman 방정식은 E[(Q−σ)⁺] = λc 이다. Bernoulli(p̄) 상금에서
  σ ∈ [0,1] 구간:  E[(Q−σ)⁺] = p̄(1−σ)      → σ = 1 − λc/p̄   (현행 닫힌형)
  σ < 0 구간   :  E[(Q−σ)⁺] = p̄ − σ        → σ = p̄ − λc     (현행 코드에 없음)
두 식은 λc > p̄ 에서 갈라지고, 부호는 같지만 **순서(argmax)가 달라질 수 있다**.
정책은 첫 강제 개봉 대상을 argmax σ로 고르므로 순서가 곧 점수다.

사용법: python eval/probe_negative_sigma.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from run_phase3 import dp_value, engine_value


def sigma_current(p, c, lam):
    return 1.0 - lam * c / np.maximum(p, 1e-9)


def sigma_exact(p, c, lam):
    """양 구간을 모두 만족하는 참 예약값 (Bernoulli)."""
    s = 1.0 - lam * c / np.maximum(p, 1e-9)
    return np.where(s >= 0.0, s, p - lam * c)


def scan(rng, n, p_rng, c_rng, lam_rng, label):
    gaps, order_diff, worst = [], 0, None
    for _ in range(n):
        M = int(rng.integers(2, 5))
        p = rng.uniform(*p_rng, size=M)
        c = rng.uniform(*c_rng, size=M)
        lam = float(rng.uniform(*lam_rng))
        if int(np.argmax(sigma_current(p, c, lam))) != int(np.argmax(sigma_exact(p, c, lam))):
            order_diff += 1
        g = dp_value(tuple(p), tuple(c), lam) - engine_value(p, c, lam)
        gaps.append(g)
        if worst is None or g > worst[0]:
            worst = (g, p.copy(), c.copy(), lam)
    gaps = np.array(gaps)
    print(f"\n[{label}]  n={n}  p∈{p_rng} c∈{c_rng} λ∈{lam_rng}")
    print(f"  argmax σ 불일치        : {order_diff}/{n} ({100 * order_diff / n:.1f}%)")
    print(f"  DP 격차 max / mean     : {gaps.max():.3e} / {gaps.mean():.3e}")
    print(f"  격차 > 1e-9 인 인스턴스: {(gaps > 1e-9).sum()}/{n}")
    if gaps.max() > 1e-9:
        g, p, c, lam = worst
        print(f"  최악 사례: gap={g:.4f}  λ={lam:.3f}")
        print(f"    p̄  = {np.round(p, 4).tolist()}")
        print(f"    c   = {np.round(c, 4).tolist()}")
        print(f"    λc  = {np.round(lam * c, 4).tolist()}   (λc>p̄ 인 상자 = 음수 σ 구간)")
        print(f"    σ_현행 = {np.round(sigma_current(p, c, lam), 4).tolist()} "
              f"→ 개봉 {int(np.argmax(sigma_current(p, c, lam)))}")
        print(f"    σ_참값 = {np.round(sigma_exact(p, c, lam), 4).tolist()} "
              f"→ 개봉 {int(np.argmax(sigma_exact(p, c, lam)))}")
    return gaps.max()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # ① 현행 3a 검증 범위 — 여기서 통과했기 때문에 결함이 드러나지 않았다
    m1 = scan(rng, 300, (0.2, 0.9), (0.01, 0.3), (0.5, 2.0), "3a 현행 범위")
    # ② λc > p̄ 가 흔한 범위 (저예산 tier = λ 큰 영역의 실제 동작 구간)
    m2 = scan(rng, 300, (0.05, 0.5), (0.05, 0.5), (1.0, 6.0), "고-λ 저-p̄ 구간")
    # ③ 극단: 거의 모든 상자가 음수 σ
    m3 = scan(rng, 300, (0.05, 0.35), (0.2, 0.6), (2.0, 10.0), "전 상자 음수 σ")
    print(f"\n판정: 최대 격차 = {max(m1, m2, m3):.4e} "
          f"({'결함 실재 — B4 일반해로 흡수 필요' if max(m1, m2, m3) > 1e-9 else '정합'})")
