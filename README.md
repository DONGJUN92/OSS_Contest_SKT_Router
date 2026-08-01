# LPB-Router

[![ci](https://github.com/DONGJUN92/OSS_Contest_SKT_Router/actions/workflows/ci.yml/badge.svg)](https://github.com/DONGJUN92/OSS_Contest_SKT_Router/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Latent-trait Pandora's Box Router** — SK텔레콤 Efficient LLM Routing Challenge 출품작.
프롬프트 난이도에 따라 최적 후보 LLM(×추론 전략)을 선택하는 compute-efficient 로컬 라우터.

핵심 아이디어: 챌린지 규칙("최종 답 = 호출한 후보 출력 중 하나")이 Weitzman(1979)의
**Pandora's Box 최적 탐색 문제와 수학적으로 동형**임을 이용해, 라우팅·승급·중단 정책을
튜닝이 아니라 **닫힌 형태(closed-form)로 유도**한다.

```bash
pip install -e ".[dev]"
python demo.py                      # 3분 시연 (네트워크 불필요)
python -m src.gate <데이터경로> --quick   # 실데이터 수령 시 첫 명령
```

### 이 저장소가 실제로 하는 일 (30초 요약)

| | |
|---|---|
| **입력** | 프롬프트(텍스트) · budget tier · 호출 이력 · 후보 모델 메타데이터(비용 **또는 단가 정책**) · 남은 예산·호출 수 |
| **출력** | `Action("call", id)` · `Action("answer", id)` · `Action("abstain", "")` |
| **결정 규칙** | 예약값 `E[(X−σ_m)⁺]=λc_m` 로 열고, `max 관측상금 ≥ max 잔여 σ` 면 멈춘다 (전체 5줄, 아래 §정책) |
| **예산 안전** | shadow price λ 페이싱 + 생존 모드 → 적대적 도착 순서에서 무응답 **96–266 → 0** |
| **규칙 준수** | 답은 항상 호출한 모델 · 외부 API 0 · 회귀 테스트 **110종**이 규칙을 불변식으로 고정 |
| **지연** | 스텝당 **0.24–0.76 ms**, 문항당 가중 **1.12 ms** (tie-break 축 — `eval/probe_latency.py`, 오프라인 적합 3–6s 는 채점 중 미발생이라 분리 보고) |
| **배포 정책 선택** | 사람이 아니라 **게이트가 데이터로** 고른다 (`src/gate.py`) |

### 검증된 것 / 검증되지 않은 것 (숨기지 않는 것이 이 저장소의 규칙)

| | |
|---|---|
| ✅ **검증됨** | 규칙↔Weitzman 동형성(DP 대조 ≤3.3e-16, 이상화 엔진 한정) · 결정론적 재현 · 실물 RouterBench 실벤치 · 배포 어댑터 관통(텍스트·단가·거절·상한) |
| ⚠️ **조건부** | 순차 관측 기구의 순가치는 **검증기 예리함**에 달려 있다 — 교차점 ≈0.885 인데 처음 보는 도메인에서의 정직한 추정은 **0.52~0.75** (LOBO, §3.2). 즉 **SKT 비공개셋에서 이 기구가 순가치를 낸다는 보장이 없다.** 그래서 게이트가 장식이 아니라 **필수 안전장치**다 |
| ⚠️ **조건부** | 헤드라인에서 `cascade-routing`(arXiv:2410.10347 계열)이 LPB 단독을 **앞선다**(0.5091 vs 0.5024). 방어하지 않고 후보로 승격시켰고, 제출 대상 `SelectiveRouter` 는 인코더 개선 후 **0.5111 로 처음 1위**가 됐다(+0.0054, 1.3×쌍대SE — 여유가 8% 뿐임을 함께 적는다) |
| ❌ **미검증** | **실데이터 0건** · A.X 실모델 0건 · 한국어 프롬프트 0건 · tier 가중·예산 수준은 우리 **가정**(Q7 미답변) |

자세한 근거와 정정 이력은 [`SUBMISSION.md`](SUBMISSION.md), 경쟁 오픈소스와의 좌표는
[`docs/related_work.md`](docs/related_work.md).

English: [`README.en.md`](README.en.md) · 기여: [`CONTRIBUTING.md`](CONTRIBUTING.md) ·
라이선스 명세: [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md)

## 정책 (전체가 5줄)

```
1. 미개봉 상자(모델×전략)의 예약지수:  E[(X − σ_m)⁺] = λ·c_m
2. 최고 지수 상자 열기(호출) → 검증기 관측 v
3. θ 사후 갱신 → 전 상자 σ 재계산 (상관 상자 처리)
4. 정지 판정: max 관측 상금 ≥ max 잔여 σ  →  연 상자 중 최고 답 반환
5. 쿼리 종료 시 λ 갱신: 누적 페이싱 오차 기반 비대칭 컨트롤러 + 생존 모드
```

`X`는 **관측 상금** `E[Q|v]`이다 — 잠재 품질 `Q`가 아니다. 상자를 열어 실제로 얻는 것이
검증기를 통해 본 값이고 정지 규칙·최종 선택도 그 값을 쓰므로, Weitzman 방정식의 올바른
입력은 `X`다. 이진 상금·정확 관측에서는 기존 닫힌형 `σ = 1 − λc/p̄`와 일치한다
(Phase 17 결함 수정, 근거: [`src/engine/prize.py`](src/engine/prize.py) `perbox_reservation`).

- `p̄_m`: 다집단 2PL IRT — `p_m(θ) = σ(a_m(θ − b_{d,m}))`, 쿼리 사전분포
  `q(θ|x) = N(μφ(x), σφ²(x))`는 경량 인코더 1회 순전파 (probit 근사 닫힌형)
- `λ`: 전역 tier 예산의 shadow price — 오프라인 quality-first 튜닝(웜스타트) +
  온라인 페이싱 (Fast tier에서는 자동으로 "최저비용 1회 호출"로 퇴화)
- **M4 인증서**: 보정셋 조기정지 후회율의 Clopper–Pearson 상한(95%)을 정책 무개입으로 제시

## 실증 결과 (검증 세계 6종, 5-fold)

> ✅ **2026-07-28 재산출 완료.** 외부 감사가 찾은 λ 튜너 래칫(D34)·대조군 페이싱 불일치(D35)
> ·산출물 파일명 충돌(D54)·**프로브 기준 팔이 자기 자신이 된 결함(D55)** 을 모두 고치고
> 오프라인 아티팩트 전체와 실벤치(§3.1·§4·§4.5)를 다시 만들었다. 상세: `docs/reflections/phase20.md`.

**런타임에 도메인/task 라벨이 제공되지 않으므로**(대회 상세), 실제 배포 구성은 다집단 2PL을
단일집단으로 퇴화시킨 것이다(`use_domain=False`). 헤드라인은 **배포 구성(단일집단)** 이고,
다집단은 "도메인이 주어졌을 때의 상한"으로 함께 싣는다.

> ⚠️ **모든 절대 점수는 자작 합성 검증 세계 기준의 잠정치다.** 주장의 대상은 절대값이 아니라
> 정책 간 **상대 서열·강건성 구조**이며, 실데이터·실 3모델(A.X)에서는 미검증이다.
> numpy 버전에 민감하다(D19). (Phase 20 부터 배포 행은 **재구성이 아니라 직접 측정**이다.)
> 3모델 재실행·학습형 라우터 대조·외부 벤치 실증은 `docs/reflections/phase14.md`·`external_review.md`.
>
> ⚠️⚠️ **아래 표는 5모델 + 투표 상자(Q2=Yes) 구성이다 — 실전 구성이 아니다.** 실전 근접
> 구성(A.X 3모델·모델당 출력 1개)의 수치는 훨씬 낮고 경쟁도 훨씬 치열하며, 그 표가 제출
> 문서의 헤드라인이다(`SUBMISSION.md` §2, 출처 `eval/results/probe_redteam_closure.json`).
> 이 표는 **상한 참조**로 남긴다. 자세한 자기 비판: `docs/reflections/phase17.md`.
>
> *(정정 D52: 이전 판은 여기에 "LPB 0.502 vs cascade-routing 0.514 vs learned 0.495" 를 적고
> `probe_redteam_closure.json` 을 출처로 달았으나 **그 파일에 없는 값**이었다 — 특히 0.514 는
> 어떤 아티팩트에도 없다. 수치를 두 곳에 복제하지 않고 단일 출처를 가리키도록 바꾼다.)*

**Phase 20 재산출** (D34 튜너 수정 후, 5-fold). 배포 행·다집단 행이 이제 **같은 실행에서 직접
측정**된다 (`eval/results/probe_domain_free.json`) — 이전 판처럼 "다집단 − Δ" 로 재구성하지 않는다.

| 종합 점수 (tier 가중 0.5/0.3/0.2) | A irt | B specialist | C corr | D crossing | E nosignal | F textworld |
|---|---|---|---|---|---|---|
| 정적 threshold cascade | 0.6844 | 0.9985 | 0.8690 | 0.9935 | 0.7869 | 0.6330 |
| **LPB-Router (배포·단일집단)** | **0.9451** | 0.9950 | **0.9365** | 0.9937 | **0.9605** | **0.8362** |
| LPB-Router (다집단) | 0.9476 | 0.9980 | 0.9286 | 0.9937 | 0.9484 | 0.8395 |

- ★ **단일집단 퇴화는 손실이 아니다 — 두 세계에서는 오히려 이득이다.** Δ(단일−다집단):
  irt −0.0025 · specialist −0.0030 · **corr +0.0079** · crossing 0.0000 · **nosignal +0.0121** ·
  textworld −0.0032. 이전 판이 "손실이 specialist −0.0079 · corr −0.0053 에 집중"이라 적은 것은
  구 튜너 기준이었고, 재산출에서 **corr·nosignal 의 부호가 뒤집혔다.** 즉 런타임에 도메인
  라벨이 없다는 제약(R2/R5)의 비용은 **거의 0 이거나 음수**다. 분해: `docs/reflections/phase13.md`
- 실텍스트 세계 F 가 구 튜너 대비 **0.8192 → 0.8395 (+0.0203)** 으로 가장 크게 올랐다.
- **라우팅이 중요한 4세계(A·C·E·F)에서 cascade를 크게 앞선다**: A **+0.2607** · C **+0.0675** ·
  E **+0.1736** · F **+0.2032**. 모든 정책이 천장(~0.99)에 몰리는 2 easy 세계에서는 차이가
  잡음 수준이다 (B specialist **−0.0035** · D crossing **+0.0002**). World F는 실텍스트·
  실출력 관통 검증 (실물 ScoringVerifier + TextEncoder)
- Weitzman 최적성: 무작위 300 인스턴스에서 DP 전역 최적해와 오차 ≤ 3.3e-16 —
  **단 이상화 엔진**(정확 관측 `ExactNoise` + 갱신 없는 `FixedBelief`) 한정이다. 배포
  라우터는 잡음 관측 + θ 갱신(=상관 상자)이라 이 증명이 전이되지 않는다
- 강건성: 시드 9/9 셀 서열 유지, 검증기 잡음 ×2에 품질 −0.01, 적대적 도착 순서에 무응답 0
- 회귀 방벽: `pytest tests/ -q` → **110/110** (규칙 불변식 포함 — 예산 초과·**미호출 모델 답
  지명**·호출 상한·패키징·Phase 17 결함 13종·**배포 게이트 4종**(예측층 무권한·SE 문턱)
  ·**Phase 20 배포 경로 14종**(텍스트 프롬프트·관측 이력·거절 루프·문항 경계·생존 모드 발동
  ·λ 선택 래칫) ·**Phase 21 신규 19종**(단가 정책 비용 규약·문항별 호출 상한·게이트 진행/부분저장
  ·인코더 선택층·쌍대 SE 채택 규칙))
- 상세 수치·결함 수정 이력(D1~D69, R1·R2): `docs/reflections/phase1~21.md`

### 스스로 찾은 한계 (Phase 17 레드팀 — 숨기지 않는 것이 이 저장소의 규칙)

> **Phase 20~21 재산출 반영 완료.** 그리고 **Phase 20 의 D34~D62 와 Phase 21 의 D63~D67 은
> 자체 발견이 아니라 외부 감사가 찾았다** — 구분해 적는 것이 이 규칙에 맞다 (`SUBMISSION.md` §7.1·§7.2).
> ◆ = Phase 21 (2026-08-01).

| 한계 | 실측 | 상태 |
|---|---|---|
| 실텍스트 검증기가 무정보에 가까움 | 실응답 AUC 0.611 → **0.867** (재현 확인). 최대 기여는 교차모델 답 일치 **+0.178** | 개선 |
| ★★ 그 0.867 이 **설계 오염값**이었다 | LOBO 평균 **0.7543** (−0.0595), 설계 대상 밖 도메인 **0.5206=무작위**. 교차점 ≈0.885 보다 낮다 | **정량화 완료(D60)**. 순차 기구는 미지 도메인에서 순가치를 못 낸다 — 게이트가 필수 |
| 임계값 AUC≈0.845가 결정 근거로 쓰였음 | 폐기 근거가 바뀌었다 — "오예측"이 아니라 **재현 스크립트 부재**(D50). 재산출에서 오예측은 **1/2** | 결정권 제거 완료 |
| fast tier(가중 0.5)에서 1회 호출 퇴화 | 호출/쿼리 **1.01** 여전. σ<0 을 0.68→0.59 로 낮춰도 종합 이득은 **+0.0012 뿐** | **퇴화 미해결이지만, 완화해도 점수가 오르지 않는다**는 것이 측정됐다 (§2.6) |
| ★ fast tier 레버 채택 근거 | **"+0.0083 (2.1×SE)" 가 재현되지 않았다** — 프로브 기준 팔이 자기 자신이 돼 있었다(D55). 재측정 **+0.0028 (1.0×SE)** | 프로브 수정 완료, **채택 서술 하향** |
| cascade 베이스라인이 예산맹목 strawman | 합성 격차 0.156 → 0.034. ★ 실벤치에서도 확인: strawman 0.5704 vs 예산인지 **0.6644** | 수정 완료 |
| cascade-routing(결합 라우팅)이 LPB를 앞섬 | 합성 0.5091 vs **0.5024**. ◆ 인코더 개선 후 제출 대상 `SelectiveRouter` **0.5111 > 0.5104** | 후보로 승격 → **역전(단 1.3×쌍대SE, 여유 8%)** |
| ★ 실벤치가 **제출 대상이 아닌 대용 정책**을 재고 있었다 | `--router` 가 dead flag(D56). 제출 대상은 **0.7166** 으로 학습형(0.7146)을 앞선다. 메타 후보에 넣자(D59) 메타가 **0.7220** 으로 1위 | 수정 완료 |
| ★ 핵심 명제가 선행연구에 있었다 | [arXiv:2606.07392](https://arxiv.org/abs/2606.07392) (2026-06-05) 가 Pandora×LLM 동형성을 먼저 발표 | 인용·우선권 고지 완료. 남는 차이는 예산제약·IRT·구현 |
| 예약값을 잠재 Q로 풀던 결함 | 잡음 크면 σ 과대 +0.47, 상자 순서 역전 | 수정 완료 |
| `open_order` 인자 미전달로 A/B 미실행 | 실제로는 체제 의존(irt −0.013 / textworld +0.007) | 수정 완료 |
| 호출 전 비용을 오라클로 가정 | `cost_model.py` + `cost_margin` 신설 | 수정 완료 |
| ◆ **어댑터가 단가 메타데이터에서 죽음** (배포 차단) | 대회는 호출 전 **비용이 아니라 단가**를 준다. `KeyError: 'cost'`. 추정기는 있었는데 배포 경로 미접속(D36·D43의 3회차) | **수정 완료** (`CostSpec`, 회귀 8종) |
| ◆ 게이트가 1,800쿼리에서 900s 무출력·부분산출 0 | `--quick`·ETA·tier 부분저장 후 **253s 판정** | 수정 완료 |
| ◆ 여지를 tier 별로 쪼갠 적이 없었음 | fast 는 가중 0.5 인데 **잔여여지 8.0%** (balanced 47.1 / premium 44.9) | **투자 우선순위 수정** — fast 추가 최적화 중단 |
| ◆ 인코더가 CLI 하드코딩·`st` 는 영어 전용 고정 | 96→512 로 p̄ log-loss +0.0897(3.3×SE)이나 **LPB 종합은 −0.0013** | 축을 게이트 후보로 승격, 배포 기본값만 변경 |
| ◆ *(기각)* "절대 문턱 0.01이 전환을 막는다"는 감사 진단 | 문턱 제거 **0.0000** · argmax −0.0009 · SE 추가 −0.0050 → **어느 방향도 이득 없음** | **측정으로 기각**, 기본값 복원·재현 확인 |
| ◆ **인코더가 배포에 안 실려 있었다** (`demo.py` 포함 **21개 파일**) | 텍스트 프롬프트에서 어댑터 즉시 사망. 오프라인은 특성 행렬만 써서 9 Phase 동안 무증상 | 21곳 일괄 수정 + 회귀 2종. ★ "전역 경고" 처방은 **측정 후 철회**(108중 41 오탐) |

## 저장소 구조

```
config.yaml            5모델 합성 세계 / config_ax3.yaml  A.X 3모델 (실전 근접)
src/
  schema.py            정규화 데이터 스키마 (모든 다운스트림의 단일 계약)
  loader.py            실데이터 로더 + 스키마 검증기 (포맷 자동 판독, CLI 제공)
  synth.py, synth2.py, synth_ext.py  합성 5세계 (+투표 상자 확장)
  textworld.py         World F 실텍스트 세계 (실프롬프트·실출력)
  cost_mirror.py       채점 비용 산식 복제 (주최측 산식으로 교체 지점)
  cost_model.py        ★ 사전 비용 추정 — 호출 전엔 decode 길이를 모른다
  cluster.py           ★ 무감독 프롬프트 군집 → pseudo-domain (런타임 라벨 부재 대응)
  encoder.py           ★ Amortized IRT 인코더 + 판별식 대조군 (배포 경로 부품)
  encoder_scan.py      ◆ 인코더 선택층 — 예측기 축도 데이터가 고른다 (p̄ 쌍대 CV)
  verifier.py          출력 텍스트 → 품질 신호 (구조·교차모델 일치·텍스트 특성)
  text_encoder.py      프롬프트 → 특성 (hashing[:dim] / st[:model], 다국어 후보 포함)
  harness.py           채점 리플레이 하네스 (Session = Pandora 동형, 추정/과금 비용 분리)
  irt/mml.py           다집단 2PL MML / beta_mml.py  연속 / grm_mml.py  다수준
  engine/prize.py      예약값 방정식의 상금족별 정확해 (+관측 상금 일반형)
  engine/pandora.py    Weitzman 예약지수 엔진 + λ 튜너
  engine/pacing.py     쌍대가격 페이싱 v3
  tune.py              하이퍼 탐색 + fold 분산 페널티 (optuna 선택)
  router.py            LPBRouter 조립 (fit / policy / certificate / diagnostics)
  gate.py              ★ 배포 게이트 — 예측층(AUC) + 결정층(직접 비교) → 배포 정책 결정
  submission.py        제출 어댑터 — 정책 형태 3종, 규칙 불변식, ◆`CostSpec`(단가 정책 흡수)
baselines/policies.py  cascade(정적·★예산인지) · 학습형 단일호출 · ★cascade-routing · 메타 선택기
demo.py                3분 시연 (docs/demo_script.md 내레이션과 짝)
reproduce.py           원커맨드 재현 — `--list` 로 단계↔표 매핑 확인
eval/                  실험·프로브. results/*.json 이 커밋된 아티팩트
eval/probe_se_k.py     ★ λ 선택 문턱 se_k 스윕 — 기본값의 근거 (Phase 20)
eval/probe_latency.py  ◆ 결정 지연 측정 — tie-break 축 (Phase 21)
eval/make_figures.py   제출용 그림 (결과 JSON만 읽음 — 하드코딩 제거)
tests/                 회귀 방벽 (규칙 불변식·DP 최적성·로더 왕복·호출상한·★패키징)
docs/reflections/      단계별 self-reflection (결함 D1~D67. phase20~21 = 외부 감사 대응)
docs/related_work.md   ◆ 선행연구·경쟁 OSS 대조 (우선권 고지 · LLMRouter/RouteLLM 비교표)
docs/branch_decision_table.md   주최측 답변 → 코드 분기 매핑 + 수령 당일 런북
CONTRIBUTING.md · THIRD_PARTY_LICENSES.md · .github/workflows/ci.yml

★ = Phase 17 (레드팀 대응) 신설 · ◆ = Phase 21 (외부 레드팀 2차) 신설
```

## 재현

```bash
pip install -r requirements.txt
python reproduce.py          # 전 단계 재현 (시드 고정, 결정론적)
# 또는 개별: python run_phase6.py 6a
```

Windows PowerShell에서는 `$env:PYTHONIOENCODING="utf-8"` 설정을 권장.

점수표는 **numpy 1.26.x** 기준이다. numpy 2.x에서는 부동소수 누적 순서 차이가 λ 탐색을
통해 증폭되어 World F가 0.8192 → 0.8189로 이동한다 (결론·서열 불변). 상세는
`requirements.txt` 주석 참조.

## 실데이터 적용 (챌린지 데이터 수령 시)

```bash
python -m src.loader <데이터경로> --peek 3                    # 포맷·컬럼명 확인
python -m src.loader <데이터경로> --map quality=<실제컬럼>     # 판독→검증→분기 판정
python -m pytest tests/ -q && python run_phase6.py 6a         # 회귀 방벽 → 점수표
```

0. ★ **가장 먼저 배포 게이트를 돌린다 — 정책은 사람이 아니라 데이터가 고른다.**

   ```bash
   python -m src.gate <데이터경로> --map quality=<컬럼> --config config_ax3.yaml --quick --scan-encoders
   ```

   ◆ **`--quick` 부터 돌릴 것** (3-fold × 1반복). 기본 구성(5-fold × 3반복)은 tier 3개에서
   **라우터 적합 135회**라 데이터 크기에 따라 수십 분~수 시간이다. `--quick` 으로 파이프라인
   전 구간이 도는 것을 먼저 확인하고, 시간이 허락하면 기본 구성으로 다시 돌린다.
   진행 표시가 stderr 로 나오고(잔여 시간 추정 포함), **tier 하나가 끝날 때마다 중간 판정이
   디스크에 저장**되므로 중간에 끊겨도 이미 끝난 tier 의 결과는 남는다.

   게이트는 3층이다.
   - ◆ **인코더층** (`--scan-encoders`): 인코더 후보를 `p̄` 의 held-out log-loss 로 쌍대
     비교해 고른다 (`src/encoder_scan.py`). 인코더는 정책이 아니라 정책의 **입력**이고
     예측기 여지는 §2.2 기준 9.2% 다 — 그 축을 하드코딩으로 두지 않는다.
   - **예측층**: 받은 데이터로 검증기를 적합해 held-out AUC를 재고 임계값과 비교해 어느
     분기가 될지 *예상*한다 (보고·조기경보용, **결정권 없음**).
   - **결정층** (권위): train 안에서 LPB·학습형 단일호출·cascade-routing을 반복 교차검증으로
     직접 리플레이 비교해 tier별로 고른다. 대안 채택은 `max(절대 margin, 1×쌍대 표준오차)`
     를 넘고 **반복 과반에서 이겨야** 한다 — D21·D61의 교훈.

   왜 임계값에 결정권을 주지 않는가: Phase 17이 얻은 교차점 AUC≈0.845는 **데이터 간 전이되지
   않는다.** Phase 18에서 두 실데이터 모두 그 임계값이 판정을 오예측했다
   (`eval/results/probe_gate.json`). 방향(검증기↑ → LPB 유리)은 견고하나 상수는 아니다.
1. **로더는 이미 있다** — `src/loader.py`가 JSON/JSONL/CSV/TSV/parquet를 읽어
   `Dataset`으로 변환한다. 실제 컬럼명은 `FieldSpec` 값으로 흡수하므로 **코드 수정 불필요**.
   `validate_dataset()`이 형상·범위·정합성을 assert하고, 데이터에서 Q2(다중 샘플)·
   Q4(연속 품질) 분기를 자동 판정한다 → `docs/branch_decision_table.md`
2. ◆ **비용 규약은 `CostSpec` 이 흡수한다 — 코드 수정 불필요.** 호스트가 문항별 비용을
   주면 그대로 쓰고, **토큰당 단가만** 주면 프리필(프롬프트 길이)+디코드(추정 길이)로
   조립한다. 키 이름·단가 단위(1K/1M)가 다르면 `CostSpec(...)` 인자만 바꾼다.
   이때는 **`LPBRouter(cost_mode="ridge", cost_margin=0.05)` 로 적합할 것** — λ 와 배포가
   같은 비용 척도를 쓰게 된다 (아니면 어댑터가 `RuntimeWarning` 을 낸다).
   채점 산식 자체가 다르면 `src/cost_mirror.py` 를 교체 (단위 테스트로 오차 0 확인).
3. ◆ **인코더는 `--scan-encoders` 가 고른다.** 한국어 데이터라면 `st:multilingual`
   (bge-m3 등, 설치·캐시 필요)이 후보에 들어가고, 없으면 해싱 차원(96/256/512) 중에서
   고른다. 게이트 **기본값은 `hashing:512`** — 쌍대 A/B 에서 제출 대상 selective3 가
   +0.0054(요구 0.0050)로 D21 문턱을 통과했다. 단 LPB 단독은 −0.0013 으로 회귀하므로
   "정답"이 아니라 기본 후보다(`eval/results/probe_encoder_axis.json`).
   전역 `get_encoder("hashing")`(=96)은 **아티팩트 재현을 위해 그대로**다.
   `IRTEncoder`·어댑터는 무변경 (Phase 8에서 2백엔드 실검증).
4. `config.yaml`의 모델·가격·tier를 주최측 스펙으로 교체
5. ◆ **호출 상한이 있으면** `sub.step(..., remaining_calls=N)` 으로 문항마다 넘긴다
   (대회 상세의 "남은 호출 수"). 전역 상한만 있으면 `SubmissionRouter(max_calls=N)`.

답변별 코드 분기 전체 매핑과 수령 당일 런북: **`docs/branch_decision_table.md`**

## License

Apache-2.0
