# 관련 연구·경쟁 오픈소스와의 위치

작성 2026-08-01 (Phase 21) · 계기: 외부 레드팀이 **2026년 신규 문헌 3편과 대표 OSS 라이브러리
1종이 저장소 어디에도 없다**고 지적. `참고자료/` 25편은 2026-07 이전 자료다.

이 문서의 목적은 자랑이 아니라 **좌표 표시**다 — 심사자가 "이 분야 대표 오픈소스와 무엇이
다른가"를 물었을 때 답할 자리가 저장소에 없었다.

---

## 1. 우선권 고지 (다시 한번, 여기에도)

이 프로젝트의 **출발 명제는 우리 것이 아니다.**

> [arXiv:2606.07392](https://arxiv.org/abs/2606.07392) — *Online Pandora's Box for Contextual
> LLM Cascading* (Belloni, Chen, Wei · Duke Fuqua · 2026-06-05)

"LLM 질의 = 상자 개봉 / 배포할 출력 선택 = keep-best-opened" 동형성과 문맥 Weitzman 지수를
먼저 제시했고 Õ(√T) regret 증명을 포함한다. 우리는 개발 이후 외부 감사로 인지했다.

**남는 차이** (증분이며, 그 이상으로 주장하지 않는다):
1. tier 예산의 **경성 제약**과 shadow price 페이싱 — 그 논문에 없다
2. IRT 잠재특성 신념 + 상관 상자 사후 갱신
3. 구현·실증·오픈소스 라우터 — **그 논문에는 수치실험이 전혀 없다**

---

## 2. 우리 결론을 **독립적으로 뒷받침하는** 신규 문헌

### ★ arXiv:2606.07587 — *The Routing Plateau* (2026-05)

21개 라우팅 기법을 5개 벤치마크에서 비교해, **서로 다른 방법들이 좁은 대역으로 수렴하며
oracle 라우터에 한참 못 미친다**고 보고한다. 원인 진단은 "예측 가능성 병목" — 라우터가
쿼리별 신호가 아니라 **모델 성능의 전역 평균 경향**을 학습한다는 것이다.

**우리와의 관계 — 이것이 이 문서에서 가장 중요한 항목이다.**

| 우리 측정 (§2.2) | 이 논문 |
|---|---|
| 달성 가능 구간의 **68.7%**, 명목 구간의 22.5%는 알레아토릭 | "좁은 대역으로 수렴, oracle 에 한참 못 미침" |
| 예측기 용량 확장 6종 전부 회귀 (Phase 19 §7) | "예측 가능성 병목 — 전역 평균 경향만 학습" |
| Phase 20 이 예측기 여지를 0% → **9.2%** 로 철회·정정 | 돌파 레버 3종: ① 더 큰 학습 데이터 ② **더 강한 인코더** ③ end-to-end 파인튜닝 |

즉 **독립적인 두 측정이 같은 천장을 가리킨다.** 그리고 그 논문이 지목한 3대 레버 중
우리가 적용한 것은 **0개**였다 — 기본 인코더가 96차원 문자 3-gram 해싱이었다.
Phase 21 에서 ②를 `src/encoder_scan.py` 로 **측정 가능한 축**으로 만들었다
(§N5/N6, `SUBMISSION.md` §2.8). ①·③은 미착수이며 §8 한계에 적는다.

### arXiv:2606.27457 — *Cluster, Route, Escalate* (2026-06)

군집 → 라우팅 → 승급의 3단 캐스케이드로 비용 인지 서빙을 구성한다. `src/cluster.py`
(무감독 pseudo-domain)와 우리 승급 구조의 **직접 경쟁자**다. 우리는 군집을 도메인 라벨
부재의 대응으로만 썼고 라우팅 단계와 결합하지 않았다 — 비교 대상 후보로 §8 에 남긴다.

### arXiv:2603.04445 — *Dynamic Model Routing and Cascading for Efficient LLM Inference: A Survey*

이 분야의 서베이. 관련연구 절의 표준 인용이며 우리 문헌 목록에 빠져 있었다.

### arXiv:2410.10347 — *A Unified Approach to Routing and Cascading for LLMs* (Dekoninck et al.)

**이미 반영돼 있다** — `baselines.CascadeRouting` 이 이 계열의 1스텝 lookahead 구현이고,
우리 헤드라인에서 **LPB 를 앞선다**(§2). 즉 이 저장소는 "지수(index) 정책이 이 체제에서
최적이 아니다"를 스스로 실증했고, 방어 대신 후보로 승격시켰다.

---

## 3. 대표 오픈소스 라우터와의 대조

| | **LPB-Router (이 저장소)** | **LLMRouter** (ulab-uiuc, 2.2k★, MIT) | **RouteLLM** (LMSYS) | **LiteLLM** |
|---|---|---|---|---|
| 목적 | 이 챌린지의 **예산 제약 하 순차 선택** | 라우팅 **연구 라이브러리** | 이진 강/약 라우팅 | LLM **게이트웨이/프록시** |
| 라우터 수 | 1 (+ 공정 베이스라인 5) | 16+ (KNN·MF·RouterDC·AutoMix·Hybrid LLM·GraphRouter·Router-R1 …) | 4 (MF·BERT·causal LLM·유사도) | 라우팅 아님 |
| **예산을 정책이 본다** | ✅ shadow price λ + 페이싱 + 생존 모드 | ✗ (대부분 단일라운드) | ✗ (비용 목표는 임계값 보정으로) | ✗ (예산 추적만) |
| **호출 후 관측으로 재결정** | ✅ Weitzman 재지수·정지 | 일부 (AutoMix) | ✗ | ✗ |
| 최종 답 = 호출한 출력 중 하나 | ✅ 규칙 불변식·테스트 고정 | 해당 없음 | 해당 없음 | 해당 없음 |
| 데이터셋·모델 zoo | ✗ (합성 6세계 + RouterBench 1종) | 11 데이터셋, 멀티모달·개인화 | Arena 선호 데이터 | 100+ 프로바이더 |
| CLI/UI·생태계 | CLI 만 | CLI·Gradio·ComfyUI·API 서버 | 서버·SDK | 프로덕션 프록시 |
| 평가 엄밀성 | **LOBO 설계누수 정량화·반복 CV·대조군 짝맞춤·결함 로그 67건** | 표준 벤치 비교 | 논문 수준 | 해당 없음 |

**정직한 판독 3가지**

1. **재사용 가능한 라이브러리로서는 명확히 하위다.** LLMRouter 는 라우터 16종·데이터셋
   11종·UI 를 갖춘 범용 도구이고, 우리는 단일 챌린지 하네스에 결합된 연구 코드다.
   오픈소스 "활용성" 축에서 이 격차는 실재한다.
2. **그런데 축이 다르다.** 위 표에서 우리만 ✅ 인 세 줄(예산을 정책이 본다 / 호출 후
   재결정 / 최종 답 규칙)이 정확히 **이 챌린지가 요구하는 것**이다. LLMRouter 의 16개
   라우터 중 어느 것도 "제한 예산 안에서 순차 개봉하며 멈출 때를 정하는" 문제를 풀지 않는다.
   비교가 아니라 **비교 불가**에 가깝다.
3. **그리고 우리 학습형 베이스라인은 그들 수준이 아니다.** `baselines.LearnedRouter` 의
   예측기는 해싱 특성 위 모델별 로지스틱 회귀(`encoder.DiscLR`)로, MF·BERT 기반
   RouteLLM 이나 RouterDC 급이 아니다. 우리는 cascade 쪽 strawman 은 스스로 고쳤는데
   (D25 → `BudgetCascade`) **learned 쪽에는 같은 잣대를 적용하지 않았다.**
   헤드라인의 LPB − learned = **+0.0072** 라는 얇은 격차가 더 강한 예측기 앞에서
   유지된다는 증거는 **없다.** §8 미해결 항목이다.

---

## 4. 이 저장소가 실제로 기여하는 것 (좁혀서)

동형성도, 결합 라우팅·캐스케이딩도, 학습형 라우팅도 우리 발명이 아니다. 남는 것:

1. **경성 tier 예산 하의 결합** — Weitzman 예약값 × shadow price 페이싱 × IRT 신념의
   3중 결합, 그리고 그것이 **적대적 도착 순서에서 무응답 96–266 → 0** 을 만든다는 실증.
   저예산 tier 에서 가장 실전적인 차별점이고, 위 표에서 어느 경쟁자도 ✅ 가 아닌 줄이다.
2. **예약값 방정식의 상금족별 정확해** (Bernoulli/Beta/GRM) 와 **관측 상금** 정정 —
   `σ = 1 − λc/p̄` 가 σ≥0 특수해임을 발견해 고친 것(D17)은 작지만 실재하는 수정이다.
3. **measure-don't-assume 문화 자체** — 자기 주장 다수를 측정 후 기각했고(conformal 정지
   제어·prefix sharing·검증기 특성공학 3회·전략적 기권·per-tier 메타선택),
   간판 결론이 관측 조건의 산물이었음을 스스로 찾아 뒤집었다(D24·D49·D56).
   이것은 논문 기여는 아니지만 **재현 가능한 오픈소스로서의 기여**다.

---

## 5. 인용

```
[1] Belloni, Chen, Wei. Online Pandora's Box for Contextual LLM Cascading. arXiv:2606.07392, 2026.
[2] The Routing Plateau: Understanding and Breaking the Accuracy Limits of LLM Routers. arXiv:2606.07587, 2026.
[3] Cluster, Route, Escalate: Cascaded Framework for Cost-Aware LLM Serving. arXiv:2606.27457, 2026.
[4] Dynamic Model Routing and Cascading for Efficient LLM Inference: A Survey. arXiv:2603.04445, 2026.
[5] Dekoninck et al. A Unified Approach to Routing and Cascading for LLMs. arXiv:2410.10347, 2024.
[6] Weitzman. Optimal Search for the Best Alternative. Econometrica 47(3), 1979.
[7] Ong et al. RouteLLM: Learning to Route LLMs with Preference Data. ICLR 2025.
[8] Hu et al. RouterBench: A Benchmark for Multi-LLM Routing System. 2024.
[9] Gergatsouli, Tzamos. Weitzman's Rule for Pandora's Box with Correlations. NeurIPS 2023 / arXiv:2301.13534.
[10] Balseiro et al. Dual Mirror Descent for Online Allocation Problems. (페이싱 regret 구조)
[11] ulab-uiuc. LLMRouter: An Open-Source Library for LLM Routing. github.com/ulab-uiuc/LLMRouter
```
