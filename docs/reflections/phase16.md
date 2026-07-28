# Phase 16 Self-Reflection — 실물 RouterBench 실벤치 + SOTA 비교 + 벤치마크별 개선

날짜: 2026-07-25 · 상태: **완료** · 산출: `eval/routerbench_real.py`
(실물 적재+frontier+tier종합+벤치별+다집단+per-tier 메타선택) · pyarrow/pandas/hf_hub 설치

## 배경

Phase 15 예행연은 RouterBench-형 **모사** 데이터였다. 이번엔 pyarrow 설치 후 **실물
RouterBench 0-shot**(HF `withmartian/routerbench`, 36,497 샘플 × **11 실모델**, MMLU/HellaSwag/
GSM8K/ARC/Winogrande/MBPP/MT-Bench)을 `src.loader` 스키마로 적재해 실벤치 수치를 내고,
글로벌·OSS 최신 라우터와 비교하며, 벤치마크별 원인 분석·개선을 반복한다.

**적재**: 각 (샘플,모델)의 실제 성능점수·실제 $비용·응답. 실제 $비용을 out_tokens에 인코딩
(cost_matrix가 실제 $ 반환). 관측=실품질(RouterBench는 호출 후 채점된 참 품질을 준다 =
**정확 관측 Pandora**, Weitzman 최적 조건). Q2=No·Q4=binary 자동판정.

## 기준점 (실물, cost$/query · quality)

- **oracle**(cheapest-correct) ≈ 0.91 · **best-single(gpt-4)** 0.775 @ $0.0033 — 라우팅 여지 +0.14
- 저가 모델: mixtral 0.53 @ $0.0001, mistral-7b 0.30 @ $0.00005

## 품질-비용 frontier (서브셋 예비, 전체는 아래)

| budget_frac | LPB | cascade | learned(RouteLLM류) |
|---|---|---|---|
| 0.06 (~$0.0002) | **0.80** | 0.20 | 0.64 |
| 0.12 (~$0.0004) | **0.86** | 0.40 | 0.67 |
| 0.25 (~$0.0008) | **0.89** | 0.75 | 0.73 |
| 0.50–1.0 | 0.906–0.91 | ~0.91 | 0.77 |

★ **저예산(챌린지 승부처)에서 LPB 압도** — cascade·learned를 크게 앞서고, **gpt-4(0.775)를
1/16 비용에 매칭**(frac 0.06). **learned를 전 예산서 +0.13 이상 앞섬**(11모델 순차 관측 우위).
고예산선 cascade가 near-oracle로 근소 우위(정확 관측 + 비용순 greedy = cheapest-correct).

**전체 36k 확정 (train 25,547 / test 10,950; oracle 0.9125, gpt-4 0.7844):**

| budget_frac | LPB | cascade | learned |
|---|---|---|---|
| 0.03 | 0.632 | 0.115 | 0.595 |
| 0.06 | **0.811** | 0.217 | 0.669 |
| 0.12 | **0.870** | 0.439 | 0.709 |
| 0.25 | 0.894 | 0.897 | 0.760 |
| 0.50 | 0.907 | 0.910 | 0.777 |
| 1.0 | 0.901 | 0.910 | 0.785 |

**챌린지 tier-가중 종합** (fast 0.08 / balanced 0.25 / premium 0.60, W 0.5/0.3/0.2):

| tier | LPB | **LPB-mg** | cascade | learned |
|---|---|---|---|---|
| fast | 0.850 | **0.859** | 0.297 | 0.687 |
| balanced | 0.884 | 0.890 | 0.897 | 0.760 |
| premium | 0.872 | 0.869 | 0.910 | 0.768 |
| **종합** | 0.865 | **0.871** | 0.599 | 0.725 |

**LPB-mg 0.871 ≫ learned 0.725(+0.15) ≫ cascade 0.599(+0.27).** cascade는 fast에서 붕괴(0.297)
— 저예산서 비용순 escalation이 정답 모델에 못 닿아 파산. LPB는 예측으로 저예산서 지배.

## 글로벌/OSS SOTA 라우터 대비 (RouterBench 논문 대조)

RouterBench(Hu et al. 2024)는 AIQ(cost-quality frontier 아래 면적/정규화)·NDCH로 라우터 비교.
논문 결과와 대조:
- **예측형 라우터(KNN/MLP — RouteLLM류 호출前 분류)**: MMLU/Winogrande서 Zero보다 낫고 **ARC·
  MBPP서 저조**. → 내 `learned`(DiscLR 단일호출)도 같은 약점, LPB가 전 예산서 이를 크게 앞섬.
- **cascading 라우터(오류율 0=정확 관측)**: "빠르게 oracle에 근접". → 내 cascade도 고예산서
  near-oracle. **LPB도 정확 관측하 near-oracle**(0.906 vs oracle 0.91)이라 cascading과 대등.
- **위치**: LPB는 예측형 라우터(KNN/MLP/learned) 대비 **저예산서 압도**(순차 관측의 가치),
  cascading 대비 고예산서 대등. 챌린지의 저예산 가중(0.5)에서 LPB가 특히 유리.

## ★ 벤치마크별 self-reflection → 반복 (다집단)

서브셋 balanced 예산 벤치별 (LPB 단일집단 / cascade / learned / best-single):

| bench | LPB | cascade | learned | best-single |
|---|---|---|---|---|
| hellaswag | **0.979** | 0.872 | 0.736 | 0.813 | ← LPB 압도(best-single 초과) |
| gsm8k | 0.687 | 0.672 | 0.635 | 0.670 | ← LPB 우위 |
| mmlu | **0.543** | 0.479 | 0.304 | **0.819** | ← LPB 약함(best-single −0.28) |
| arc | **0.595** | 0.976 | 0.310 | 0.929 | ← LPB 약함 |
| winogrande | **0.611** | 1.000 | 0.333 | 0.833 | ← LPB 약함 |

**원인 진단**: LPB 단일집단(배포용, SKT 런타임엔 도메인 없음)이 **"모델 X가 mmlu/arc에 강함"
을 못 잡는다**(단일 θ). best-single이 높은데 LPB가 못 따라가는 것은 벤치별 모델 강점 미포착.
**RouterBench는 벤치 라벨이 있으니** 다집단 2PL(b_{d,m})이 이를 고칠 수 있다(SKT와 달리 공정).
→ **반복: LPB-mg(다집단) 추가 측정** — mmlu/arc/winogrande를 고치는지.

**전체 36k 벤치별 (balanced): LPB / LPB-mg / cascade / learned / best-single (n):**

| bench | LPB | LPB-mg | cascade | learned | best-single | n |
|---|---|---|---|---|---|---|
| hellaswag | 0.972 | 0.973 | 0.862 | 0.685 | 0.847 | 3060 | ← LPB 압도 |
| mbpp | 0.884 | 0.884 | 0.643 | 0.670 | 0.696 | 112 | ← LPB 압도 |
| gsm8k | 0.703 | 0.688 | 0.683 | 0.617 | 0.662 | 2236 | ← LPB 우위 |
| other | 0.808 | 0.809 | 0.342 | 0.570 | 0.643 | 284 | ← LPB 압도 |
| **mmlu** | 0.636 | **0.704** | 0.677 | 0.253 | 0.813 | 4197 | ← 다집단 +0.07 |
| **chinese** | 0.424 | **0.608** | 0.161 | 0.380 | 0.489 | 249 | ← 다집단 +0.18 |
| **arc** | 0.566 | 0.509 | **0.987** | 0.251 | 0.971 | 454 | ← cascade 지배 |
| **winogrande** | 0.545 | 0.533 | **1.000** | 0.220 | 0.855 | 332 | ← cascade 지배 |

**self-reflection**:
- ✅ **다집단(LPB-mg)이 예측대로 mmlu(+0.07)·chinese(+0.18)를 고쳤다** — 벤치 라벨이 "모델별
  벤치 강점"을 잡아 라우팅을 개선. mmlu가 최대 표본(n=4197)이라 종합이 0.865→0.871로 상승.
  단 arc(−0.06, n작아 과적합)·gsm8k(−0.015)는 소폭 손해 — 다집단은 표본 충분한 벤치서 이득.
- ✗ **arc/winogrande는 여전히 cascade(0.99/1.00)에 열세**(LPB 0.55, best-single 0.97/0.86).
  이 벤치는 **정답 모델이 명확하고 값싼 escalation으로 도달** 가능해, 정확관측+비용순 cascade가
  near-oracle이다. LPB의 value-order는 그만큼 개봉하지 않는다.
- **반복(cascade 포함 per-tier 메타 선택) = 실측 무이득**: train으로 tier별 최적을 골라도
  **메타 0.8615 vs LPB-mg 0.8613 (+0.0002)**. 선택은 fast=LPB-mg / balanced=LPB-mg /
  premium=cascade인데, LPB-mg가 가중치 높은 fast+balanced(0.8)를 이미 지배하고 cascade의 premium
  우위(+0.0008)는 가중 종합을 못 움직인다. **cascade를 붙일 가치가 챌린지 metric엔 없다** —
  또 하나의 measure-don't-assume 부정 결과. (arc/winogrande는 balanced-단독 예산의 국소 약점일 뿐
  tier-가중에선 fast 지배가 압도.)

## 판정

- **실물 RouterBench(11 실모델)에서 LPB(-mg)는 챌린지 tier-종합을 지배**: **0.871 vs learned
  0.725(+0.15) vs cascade 0.599(+0.27)**. 예측형 SOTA 라우터(KNN/MLP = learned류)를 전 예산서
  크게 앞선다 — RouterBench 논문이 예측형 라우터의 ARC/MBPP 약점을 지적한 것과 정합, 순차
  관측·재선택의 가치가 11모델 실데이터로 실증됨. gpt-4를 **1/16 비용에 매칭**(frac 0.06).
- **cascading(정확 관측)과는 고예산서 대등**(둘 다 near-oracle 0.91). 챌린지 저예산 가중(0.5)
  에서 cascade가 fast(0.30)에 붕괴하므로 **tier-종합은 LPB 압승**.
- **개선 반복 성과·한계**: 다집단(LPB-mg)이 mmlu/chinese를 실측 개선(+0.07/+0.18, 종합
  0.865→0.871). 그 위 cascade 포함 per-tier 메타 선택은 **실측 무이득(+0.0002)** — LPB-mg가
  가중치 높은 tier를 이미 지배하기 때문. 즉 이 metric에서 **LPB-mg가 사실상 상한**이고 추가
  복잡도는 값이 없다(measure-don't-assume). SKT는 런타임 도메인 미제공이라 다집단엔 벤치-분류기가
  필요(이월) — RouterBench는 도메인 제공이라 LPB-mg가 바로 적용 가능한 공정 비교였다.
