"""잔여 검증 통합 러너: 7b2(약점 세계 재검증) → 7c(5세계 회귀 매트릭스) 순차 실행.

단일 호출로 Phase 7 잔여 작업을 전부 완료한다.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from run_phase7 import stage_7a, ALL_WORLDS

print("=" * 80)
print("STEP 1/2 — 7b2: 페이싱 v3 적용 후 약점 세계(corr/crossing) 재검증")
print("=" * 80)
stage_7a(worlds=["corr", "crossing"], tag="7b2")

print()
print("=" * 80)
print("STEP 2/2 — 7c: 회귀 매트릭스 (5세계 전체, 무회귀 확인)")
print("=" * 80)
stage_7a(worlds=ALL_WORLDS, tag="7c")
print("\n[run_resume] 완료 — 결과: eval/results/phase7b2.json, phase7c.json")
