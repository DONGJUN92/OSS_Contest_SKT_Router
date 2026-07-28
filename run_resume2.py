"""Phase 8 잔여 검증 통합 러너 — 단일 호출로 4단계 순차 실행.

  1. 베이지안 사후 선택 반사실 probe (Phase 6 기각 건의 재시험 근거)
  2. pytest 회귀 스위트 (B8)
  3. posterior prize 모드 A/B — textworld에서 calibrated vs posterior
  4. 무회귀 게이트 — 합성 세계(irt/corr)에서 posterior 모드 부작용 확인
"""
import subprocess, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

print("=" * 78)
print("STEP 1/4 — 베이지안 사후 선택 반사실 (textworld)")
print("=" * 78)
subprocess.run([sys.executable, "eval/probe_bayes_selection.py"], check=True)

print()
print("=" * 78)
print("STEP 2/4 — pytest 회귀 스위트 (B8)")
print("=" * 78)
subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], check=True)

print()
print("=" * 78)
print("STEP 3/4 — posterior prize A/B (textworld end-to-end)")
print("=" * 78)
from run_phase8 import stage_8b
cal_l, cal_c = stage_8b(tag="8b-cal", prize_mode="calibrated")
post_l, post_c = stage_8b(tag="8b-post", prize_mode="posterior")
print(f"\n  [A/B] calibrated 종합={cal_l:.4f}  posterior 종합={post_l:.4f}"
      f"  delta={post_l - cal_l:+.4f}  (cascade={cal_c:.4f})")

print()
print("=" * 78)
print("STEP 4/4 — 무회귀 게이트: 합성 세계에서 posterior 모드 (irt/corr)")
print("=" * 78)
from run_phase7 import stage_7a
stage_7a(worlds=["irt", "corr"], router_kwargs={"prize_mode": "posterior"},
         tag="8c-reg")
print("\n[run_resume2] 완료 — 비교 기준: phase7c(irt 0.9481, corr 0.9258)")
