# Contributing to LPB-Router

기여를 환영합니다. 이 프로젝트는 **"측정하지 않은 개선은 채택하지 않는다"**(measure, don't
assume)는 규칙 하나로 운영됩니다. 아래는 그 규칙을 코드로 강제하는 방법입니다.

## 1. 개발 환경

```bash
git clone <repo> && cd lpb-router
python -m venv .venv && . .venv/Scripts/activate   # Linux/macOS: . .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -q
```

점수 대조가 목적이면 `numpy 1.26.x` 를 쓰세요. numpy 2.x 에서는 부동소수 누적 순서가
λ 이분탐색을 통해 증폭되어 World F 종합점수가 0.8192 → 0.8189 로 이동합니다(결론·서열 불변,
`docs/reflections/phase11.md` D19).

## 2. PR 이 통과해야 하는 것

1. `python -m pytest tests/ -q` 전부 통과. 회귀 테스트는 **챌린지 규칙을 불변식으로** 담고
   있으므로(예: 최종 답은 호출한 모델만, 예산 초과 불가, 호출 상한 준수) 이걸 깨는 PR 은
   기능이 아니라 규칙 위반입니다.
2. 정책 동작을 바꾸는 PR 은 **숫자를 첨부**하세요. 최소 `eval/probe_redteam_closure.py`
   (실전 근접 구성) 결과를 before/after 로 붙입니다.
3. 이득이 fold 표준오차를 넘지 못하면 **기본값을 바꾸지 않습니다**. `docs/reflections/phase12.md`
   의 D21 이 그 교훈입니다 — 잡음을 승리로 오인한 판정을 표준오차 문턱으로 뒤집은 기록.

## 3. 새 아이디어를 추가하는 올바른 형태

플래그로 넣고, 기본값은 끄고, 프로브로 재고, 결과를 문서에 쓰세요. 실측이 음수면
**그 사실을 남기고 폐기**합니다. 실제 사례:

| 아이디어 | 결과 | 기록 |
|---|---|---|
| conformal 정지 제어 | 기각 (전역 예산과 충돌) | `phase5.md` D12 |
| KV-cache prefix sharing | 기각 (실측 −0.007~−0.018) | `phase13.md` |
| 검증기 특성 강화(task-불가지 6종) | 기각 (ΔAUC 0.0012 < 표준오차) | `phase14.md` |
| per-tier 메타 선택 | 기각 (+0.0002) | `phase16.md` |
| cascade-routing 베이스라인 | **채택** (LPB 를 앞서 후보로 승격) | `phase17.md` |
| 무감독 군집 pseudo-domain | 기각 (−0.0044) | `phase17.md` N4 |
| 실도메인 STRUCT_CHECKS | 기각 (직접 기여 +0.007) — 단 진단이 파서 버그를 찾아냈다 | `phase18.md` §3 |
| fast tier λ 세밀격자 + p̄ Platt 재보정 | **채택** (2.1×쌍대 SE) | `phase18.md` §2 |
| 교차점 AUC 상수를 배포 결정 근거로 쓰기 | **기각** (두 실데이터에서 오예측) | `phase18.md` §1 |

## 4. 코드 규약

- 의존성은 최소로. 기본 경로는 `numpy`/`scipy`/`pyyaml` 만 씁니다 (챌린지의 오프라인·재현성
  요건). `sentence-transformers`·`optuna`·`pandas` 는 optional extras 입니다.
- 배포 경로(`src/`, `baselines/`)는 실험 스크립트(`run_*.py`, `phase2_stages.py`)를
  **import 하지 않습니다**. `tests/test_packaging.py` 가 이를 강제합니다.
- 주석은 "무엇을" 이 아니라 **"왜 이 형태인가"** 를 씁니다. 특히 결함을 고친 자리에는
  결함 번호(D17, D21 …)와 근거 문서를 남깁니다.
- 한국어 주석이 기본입니다. 영어 기여도 환영하며 혼용을 강제하지 않습니다.

## 5. 이슈

버그 리포트에는 (1) 재현 명령 (2) numpy/scipy 버전 (3) 기대값과 실측값을 적어 주세요.
정책 품질에 대한 의견은 **어떤 구성에서 측정했는지**를 함께 적어야 논의가 가능합니다 —
이 저장소의 모든 수치는 구성 의존적입니다.
