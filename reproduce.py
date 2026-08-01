"""원커맨드 재현 스크립트 — 시드 고정 순차 실행.

사용법
  python reproduce.py            # 기본: 제출 문서의 **모든 헤드라인 수치** 경로
  python reproduce.py --quick    # 최소 검증 (테스트 + 2세계 점수표 + ablation)
  python reproduce.py --full     # 위 + 초기 단계 실험 전부 (Phase 1~5 재현)
  python reproduce.py --list     # 무엇이 어떤 단계에서 나오는지만 출력

★ Phase 17 수정 (레드팀 MAJOR): 이전 버전은 `run_phase1~6` 만 돌려 **2세계**(irt·specialist)
만 재생성했다. 그런데 SUBMISSION.md 헤드라인은 6세계였고, 배포 행(단일집단)·3모델 collapse·
Q2=No·RouterBench 실벤치는 전부 이 스크립트 밖이었다 — "원커맨드 재현"이 문서의 표를
재현하지 못했다. 기본 경로에 **문서에 실린 표를 만드는 스크립트를 전부** 넣었다.

각 단계가 어떤 산출물을 쓰는지는 STEPS 의 세 번째 항목에 적어 두었고, `--list` 로 볼 수 있다.
외부 데이터(HuggingFace RouterBench)가 필요한 단계는 `needs_net=True` 로 표시되고 기본
경로에서 제외된다 — 챌린지의 오프라인 규칙은 **라우터**에 적용되지만, 실벤치 실증은
연구 재현이므로 분리해 둔다 (`--bench` 로 포함).
"""
import subprocess, sys, time, os

QUICK = "--quick" in sys.argv
FULL = "--full" in sys.argv
BENCH = "--bench" in sys.argv

# (이름, 명령, 산출물/뒷받침하는 표, 외부 네트워크 필요)
CORE = [
    ("tests", ["-m", "pytest", "tests/", "-q"], "회귀 방벽 전체", False),
    ("phase6-final", ["run_phase6.py", "6a"], "final_scores.json — 2세계 end-to-end + 재현 delta", False),
    ("phase6-ablation", ["run_phase6.py", "6b"], "phase6b_ablation.json — SUBMISSION §구성요소 기여", False),
    ("phase7-worlds", ["run_phase7.py", "7c"], "phase7c*.json — 세계 C·D·E 점수", False),
    ("phase8-textworld", ["run_phase8.py", "8b"], "phase8b.json — World F 실텍스트", False),
    ("domain-free", ["eval/probe_domain_free.py"], "probe_domain_free.json — **배포 행(단일집단)**, fig1 입력", False),
    ("ax3-collapse", ["eval/probe_ax3_collapse.py", "--folds", "5"], "probe_ax3_collapse.json — A.X 3모델", False),
    ("ax3-q2no", ["eval/probe_ax3_collapse.py", "--folds", "5", "--single"],
     "probe_ax3_collapse(_q2no).json — **SUBMISSION §2 헤드라인**", False),
    ("observable-prize", ["eval/probe_observable_prize.py", "--folds", "2"],
     "probe_observable_prize.json — 관측상금 예약값·open_order A/B (Phase 17)", False),
    ("redteam-closure", ["eval/probe_redteam_closure.py", "--folds", "3"],
     "probe_redteam_closure.json — **레드팀 폐쇄 판정표** (Phase 17)", False),
    ("fast-tier", ["eval/probe_fast_tier.py", "--folds", "3"],
     "probe_fast_tier.json — fast tier 전용 레버 (λ 격자·p̄ 재보정) 판정 (Phase 18)", False),
    # ★ Phase 20: D34 수정이 남긴 유일한 열린 손잡이(`se_k`)를 측정으로 닫는다. 이 단계의
    # 판정이 `LPBRouter(lam_se_k=...)` 기본값의 근거이므로, 기본값을 바꾸려면 여기 판정이
    # D21 문턱(max(0.005, 1×쌍대 SE))을 넘어야 한다.
    ("se-k-sweep", ["eval/probe_se_k.py", "--folds", "5", "--seeds", "0,1,2"],
     "probe_se_k.json — **se_k × Platt 격자 (5-fold × 3시드)**, 기본값 근거 (Phase 20)", False),
    # ★ D50 해소: §3 법칙 표의 인위 열화 3행을 만드는 스크립트. 초판에는 이 파일이 없어
    # 교차점 상수 0.845 가 재현 불가능했다 ("산문만 있는 수치는 쓰지 않는다"의 위반).
    ("verifier-sensitivity", ["eval/probe_verifier_sensitivity.py", "--folds", "3"],
     "probe_verifier_sensitivity.json — **§3 검증기 예리함 스윕·교차점** (Phase 20 신설)", False),
    # ★ D47: 아래 두 단계는 SUBMISSION §2.2(달성 가능 상한·축별 분해)와 §2.7(예산 스윕)을
    # 만드는데 초판 reproduce.py 에 **없었다** — 그런데 §10 은 "표를 만드는 스크립트를 전부
    # 포함한다"고 적었다. §2.2 는 이 보고서가 스스로 '핵심 수치'라 부르는 표다.
    ("predictor-ceiling", ["eval/probe_predictor_ceiling.py"],
     "probe_predictor_ceiling.json — **SUBMISSION §2.2 달성 가능 상한·축별 분해**", False),
    ("budget-sweep", ["eval/probe_budget_sweep.py"],
     "probe_budget_sweep.json — **SUBMISSION §2.7 예산 수준 강건성**", False),
    ("gate", ["eval/probe_gate.py", "--skip-bench"],
     "probe_gate.json — **배포 게이트** 분기 A (Phase 18). 분기 B 는 이전 값 보존, --bench 로 재계산", False),
    # ◆ Phase 21 — D47 의 규칙("표를 만드는 스크립트는 전부 여기 있어야 한다")을 신규 표에도
    # 적용한다. 세 단계 모두 SUBMISSION 의 특정 표에 대응한다.
    ("latency", ["eval/probe_latency.py", "--n", "600", "--reps", "3"],
     "probe_latency.json — **SUBMISSION §9.2 결정 지연** (tie-break 축, Phase 21 신설)", False),
    ("closure-encoder", ["eval/probe_redteam_closure.py", "--folds", "3",
                         "--encoder", "hashing:512"],
     "probe_redteam_closure_enc-hashing512.json — **§2.8 인코더 축 A/B** (Phase 21)", False),
    ("closure-paired-se", ["eval/probe_redteam_closure.py", "--folds", "3",
                           "--margin-mode", "paired_se"],
     "probe_redteam_closure_paired_se.json — **§2.9 채택 규칙 A/B** (Phase 21)", False),
    ("closure-argmax", ["eval/probe_redteam_closure.py", "--folds", "3",
                        "--margin-mode", "argmax"],
     "probe_redteam_closure_argmax.json — **§2.9 채택 규칙 극단 팔** (Phase 21)", False),
    # ★ 이 단계가 배포 인코더 기본값(hashing:512)의 **유일한 근거**다. 점추정이 D21 절대
    # 문턱 바로 위(+0.0054 vs 0.0050)라 쌍대 SE 가 없으면 기본값을 바꿀 수 없다.
    ("encoder-axis", ["eval/probe_encoder_axis.py", "--folds", "3"],
     "probe_encoder_axis.json — **§2.8 인코더 쌍대 A/B (SE 포함)**, 배포 기본값 근거", False),
    ("figures", ["eval/make_figures.py"], "figures/*.png (결과 JSON만 읽음)", False),
]
BENCH_STEPS = [
    ("verifier-real", ["eval/probe_verifier_real.py"],
     "probe_verifier_real.json — 실텍스트 검증기 AUC 게이트 (PASS/FAIL)", True),
    # `--router` 는 제출 대상 `LPBRouter` 자체를 함께 잰다 (D56 — 이전에는 dead flag 였고
    # 벤치가 수제 대용 정책만 쟀다). 대용 정책과 나란히 보고하므로 둘의 차이도 기록된다.
    ("routerbench", ["eval/routerbench_real.py", "--subset", "4000", "--router"],
     "routerbench_real_noisy_n4000.json — 실물 11모델 실벤치 (잡음 관측 = 챌린지 동형) + ★제출 라우터", True),
    ("gate-bench", ["eval/probe_gate.py"],
     "probe_gate.json — 게이트를 실벤치 분기까지 (Phase 18)", True),
]
EARLY = [
    ("phase1-baselines", ["run_phase1.py"], "phase1.json", False),
    ("phase2a-irt", ["run_phase2.py", "2a"], "phase2a.json", False),
    ("phase2b-encoder", ["run_phase2.py", "2b"], "phase2b.json", False),
    ("phase2c-routing", ["run_phase2.py", "2c"], "phase2c.json", False),
    ("phase3a-optimality", ["run_phase3.py", "3a"], "DP 최적성 (이상화 엔진)", False),
    ("phase3b-comparison", ["run_phase3.py", "3b"], "phase3b.json", False),
    ("phase3c-ablation", ["run_phase3.py", "3c"], "phase3c.json", False),
    ("phase4a-votes", ["run_phase4.py", "4a"], "phase4b.json (투표 상자)", False),
    ("phase4b-pacing", ["run_phase4.py", "4b"], "페이싱 순서 스트레스", False),
    ("phase4c-fullstack", ["run_phase4.py", "4c"], "phase4c.json", False),
    ("phase5-certificate", ["run_phase5.py", "5d"], "phase5a.json (M4 인증서)", False),
]

if QUICK:
    STEPS = [CORE[0], CORE[1], CORE[2]]
elif FULL:
    STEPS = EARLY + CORE
else:
    STEPS = CORE
if BENCH:
    STEPS = STEPS + BENCH_STEPS

if "--list" in sys.argv:
    print(f"{'단계':<22}{'산출물 / 뒷받침하는 표'}")
    for name, args, what, net in (EARLY + CORE + BENCH_STEPS):
        tag = " [네트워크]" if net else ""
        print(f"  {name:<20}{what}{tag}")
    sys.exit(0)

env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
t0 = time.time()
failed = []
for name, args, what, net in STEPS:
    print(f"\n{'=' * 74}\n[reproduce] {name}  —  {what}\n{'=' * 74}", flush=True)
    ret = subprocess.run([sys.executable, *args], env=env).returncode
    if ret != 0:
        print(f"[reproduce] FAILED at {name} (exit {ret})")
        failed.append(name)
        if name == "tests":                      # 회귀가 깨지면 나머지는 의미 없다
            sys.exit(ret)
print(f"\n[reproduce] 완료 — 총 {time.time() - t0:.0f}s. 결과: eval/results/")
if failed:
    print(f"[reproduce] 실패 단계: {failed}")
    sys.exit(1)
