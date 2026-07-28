# LPB-Router

**Latent-trait Pandora's Box Router** — SK텔레콤 Efficient LLM Routing Challenge 출품작.
프롬프트 난이도에 따라 최적 후보 LLM(×추론 전략)을 선택하는 compute-efficient 로컬 라우터.

핵심 아이디어: 챌린지 규칙("최종 답 = 호출한 후보 출력 중 하나")이 Weitzman(1979)의
**Pandora's Box 최적 탐색 문제와 수학적으로 동형**임을 이용해, 라우팅·승급·중단 정책을
튜닝이 아니라 **닫힌 형태(closed-form)로 유도**한다.

English: [`README.en.md`](README.en.md) · 기여: [`CONTRIBUTING.md`](CONTRIBUTING.md) ·
라이선스 명세: [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) · 시연: `python demo.py`

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
- 회귀 방벽: `pytest tests/ -q` → **86/86** (규칙 불변식 포함 — 예산 초과·**미호출 모델 답
  지명**·호출 상한·패키징·Phase 17 결함 13종·**배포 게이트 4종**(예측층 무권한·SE 문턱)
  ·**Phase 20 배포 경로 14종**(텍스트 프롬프트·관측 이력·거절 루프·문항 경계·생존 모드 발동
  ·λ 선택 래칫))
- 상세 수치·결함 수정 이력(D1~D55, R1·R2): `docs/reflections/phase1~20.md`

### 스스로 찾은 한계 (Phase 17 레드팀 — 숨기지 않는 것이 이 저장소의 규칙)

> **Phase 20 재산출 반영 완료.** 그리고 **Phase 20 결함 15종(D34~D55)은 자체 발견이 아니라
> 외부 감사가 찾았다** — 구분해 적는 것이 이 규칙에 맞다 (`SUBMISSION.md` §7.1).

| 한계 | 실측 | 상태 |
|---|---|---|
| 실텍스트 검증기가 무정보에 가까움 | 실응답 AUC 0.611 → **0.867** (재현 확인). 최대 기여는 교차모델 답 일치 **+0.178** | 개선(단 특성 설계가 이 데이터를 본 뒤 나옴) |
| 임계값 AUC≈0.845가 결정 근거로 쓰였음 | 폐기 근거가 바뀌었다 — "오예측"이 아니라 **재현 스크립트 부재**(D50). 재산출에서 오예측은 **1/2** | 결정권 제거 완료 |
| fast tier(가중 0.5)에서 1회 호출 퇴화 | 호출/쿼리 **1.01** 여전. σ<0 을 0.84→0.68 로 낮춰도 종합 이득은 **+0.0011 뿐** | **퇴화 미해결이지만, 완화해도 점수가 오르지 않는다**는 것이 측정됐다 (§2.6) |
| ★ fast tier 레버 채택 근거 | **"+0.0083 (2.1×SE)" 가 재현되지 않았다** — 프로브 기준 팔이 자기 자신이 돼 있었다(D55). 재측정 **+0.0028 (1.0×SE)** | 프로브 수정 완료, **채택 서술 하향** |
| cascade 베이스라인이 예산맹목 strawman | 합성 격차 0.156 → 0.034. ★ 실벤치에서도 확인: strawman 0.4968 vs 예산인지 **0.6612** | 수정 완료 |
| cascade-routing(결합 라우팅)이 LPB를 앞섬 | 합성 0.5091 vs **0.5024** (격차 **0.0067 로 확대**). 단 실벤치에서는 둘 다 학습형에 진다 | 후보로 승격(`SelectiveRouter`), **미역전** |
| ★ 핵심 명제가 선행연구에 있었다 | [arXiv:2606.07392](https://arxiv.org/abs/2606.07392) (2026-06-05) 가 Pandora×LLM 동형성을 먼저 발표 | 인용·우선권 고지 완료. 남는 차이는 예산제약·IRT·구현 |
| 예약값을 잠재 Q로 풀던 결함 | 잡음 크면 σ 과대 +0.47, 상자 순서 역전 | 수정 완료 |
| `open_order` 인자 미전달로 A/B 미실행 | 실제로는 체제 의존(irt −0.013 / textworld +0.007) | 수정 완료 |
| 호출 전 비용을 오라클로 가정 | `cost_model.py` + `cost_margin` 신설 | 수정 완료 |

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
  verifier.py          출력 텍스트 → 품질 신호 (구조·교차모델 일치·텍스트 특성)
  text_encoder.py      프롬프트 → 특성 (hashing 기본 / sentence-transformers 선택)
  harness.py           채점 리플레이 하네스 (Session = Pandora 동형, 추정/과금 비용 분리)
  irt/mml.py           다집단 2PL MML / beta_mml.py  연속 / grm_mml.py  다수준
  engine/prize.py      예약값 방정식의 상금족별 정확해 (+관측 상금 일반형)
  engine/pandora.py    Weitzman 예약지수 엔진 + λ 튜너
  engine/pacing.py     쌍대가격 페이싱 v3
  tune.py              하이퍼 탐색 + fold 분산 페널티 (optuna 선택)
  router.py            LPBRouter 조립 (fit / policy / certificate / diagnostics)
  gate.py              ★ 배포 게이트 — 예측층(AUC) + 결정층(직접 비교) → 배포 정책 결정
  submission.py        제출 어댑터 — 정책 형태 3종 지원, 규칙 불변식 강제
baselines/policies.py  cascade(정적·★예산인지) · 학습형 단일호출 · ★cascade-routing · 메타 선택기
demo.py                3분 시연 (docs/demo_script.md 내레이션과 짝)
reproduce.py           원커맨드 재현 — `--list` 로 단계↔표 매핑 확인
eval/                  실험·프로브. results/*.json 이 커밋된 아티팩트
eval/probe_se_k.py     ★ λ 선택 문턱 se_k 스윕 — 기본값의 근거 (Phase 20)
eval/make_figures.py   제출용 그림 (결과 JSON만 읽음 — 하드코딩 제거)
tests/                 회귀 방벽 (규칙 불변식·DP 최적성·로더 왕복·호출상한·★패키징)
docs/reflections/      단계별 self-reflection (결함 D1~D55. phase20 = 외부 감사 대응)
docs/branch_decision_table.md   주최측 답변 → 코드 분기 매핑 + 수령 당일 런북
CONTRIBUTING.md · THIRD_PARTY_LICENSES.md · .github/workflows/ci.yml

★ = Phase 17 (레드팀 대응) 신설
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
   python -m src.gate <데이터경로> --map quality=<컬럼> --config config_ax3.yaml
   ```

   게이트는 2층이다. **예측층**은 받은 데이터로 검증기를 적합해 held-out AUC를 재고 임계값과
   비교해 어느 분기가 될지 *예상*한다(보고·조기경보용, 결정권 없음). **결정층**은 train 안에서
   LPB·학습형 단일호출·cascade-routing을 직접 리플레이 비교해 tier별로 고른다(권위).
   대안 채택은 `max(절대 margin, 1×쌍대 표준오차)`를 넘어야 한다 — D21의 교훈.

   왜 임계값에 결정권을 주지 않는가: Phase 17이 얻은 교차점 AUC≈0.845는 **데이터 간 전이되지
   않는다.** Phase 18에서 두 실데이터 모두 그 임계값이 판정을 오예측했다
   (`eval/results/probe_gate.json`). 방향(검증기↑ → LPB 유리)은 견고하나 상수는 아니다.
1. **로더는 이미 있다** — `src/loader.py`가 JSON/JSONL/CSV/TSV/parquet를 읽어
   `Dataset`으로 변환한다. 실제 컬럼명은 `FieldSpec` 값으로 흡수하므로 **코드 수정 불필요**.
   `validate_dataset()`이 형상·범위·정합성을 assert하고, 데이터에서 Q2(다중 샘플)·
   Q4(연속 품질) 분기를 자동 판정한다 → `docs/branch_decision_table.md`
2. `src/cost_mirror.py`를 주최측 비용 산식으로 교체 (단위 테스트로 오차 0 확인)
3. 인코더 백본 교체가 필요하면 `src/text_encoder.py`의 `st` 백엔드를 다국어 모델
   (bge-m3 등)로 — 어댑터 동일, `IRTEncoder`는 무변경 (Phase 8에서 2백엔드 실검증)
4. `config.yaml`의 모델·가격·tier를 주최측 스펙으로 교체

답변별 코드 분기 전체 매핑과 수령 당일 런북: **`docs/branch_decision_table.md`**

## License

Apache-2.0
