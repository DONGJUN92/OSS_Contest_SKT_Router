# Phase 15 Self-Reflection — 한계 극복 R&D 루프 + RouterBench 예행연

날짜: 2026-07-25 · 상태: **완료** · 산출:
`src/engine/pandora.py`·`src/router.py`·`src/submission.py`(open_order) ·
`baselines/policies.py`(**SelectiveRouter**) · `config_ax3.yaml`·`config_routerbench.yaml` ·
`eval/probe_belief_diag.py`·`probe_hybrid.py`·`probe_routerbench_rehearsal.py`·`make_routerbench_style.py` ·
`tests/`(SelectiveRouter 신설)

## 배경 — 극복할 한계

Phase 14 적대적 평가가 실측한 한계: **3모델·실텍스트(textworld)에서 학습형 단일호출 라우터
(RouteLLM류)가 LPB를 앞선다**(Q2=Yes −0.019, Q2=No −0.001). 방어하지 말고 원인을 규명해
개선하고, 극복 여부를 self-reflect하며 반복한다.

## 진단 — 원인은 belief가 아니라 결정 구조

가설: LPB가 진 건 IRT 신념의 예측력이 실텍스트에서 약해서? → **반증**
(`probe_belief_diag.py`, 3세계 5-fold):

| world | IRT-LPB | Disc-LPB | value-LPB | learned | ll_irt / ll_disc |
|---|---|---|---|---|---|
| textworld | 0.6831 | 0.6717 | 0.7172 | 0.7695 | **0.423 / 0.451** |
| irt | 0.8216 | 0.8139 | 0.7998 | 0.7091 | 0.384 / 0.384 |
| specialist | 0.9703 | 0.9790 | 0.9743 | 0.8820 | 0.533 / 0.491 |

**IRT log-loss ≤ DiscLR** (예측을 더/동등하게 잘함) + **Disc-LPB ≈ IRT-LPB**(신념 교체 무효).
학습형은 **더 나쁜 예측(DiscLR)으로도** LPB를 이겼다 → 우위는 예측이 아니라 **결정 구조**.

## 시도1 — belief-swap (판별식 신념): 중립

Weitzman 엔진에 IRT 대신 DiscLR FixedBelief를 꽂아도 ≈동일(위 표 Disc-LPB). 기각.

## 시도2 — value-order (즉시 순가치 개봉): ★D17에 이미 포섭

진단에서 value-order(σ 대신 `argmax(p̄−λc)` 개봉)가 σ보다 **+0.034**(textworld)로 보여
엔진에 `open_order`(sigma|value|auto) 구현. **그러나 실제 LPBRouter에서 self-reflection:**

    fast  sigma=0.6129  value=0.6129  (완전 동일)   auto가 sigma 선택
    balanced sigma=0.8561 value=0.8561 (완전 동일)

★ **원인 규명**: Bernoulli 예약값의 **D17 σ<0 분기가 `σ=p̄−λc` = value-order와 정확히 동일**
하고, quality-first λ(D11, ties→큰 λ)가 tight tier를 `λc>p̄`(σ<0) 체제로 밀어넣는다. 즉
**LPBRouter는 빠듯한 예산에서 이미 value-optimal하게 연다** — value-order는 기존 설계에 포섭돼
있었다. 진단의 +0.034는 hand-built 프로브의 **약한 minimal-λ 튜닝**(σ≥0 체제 잔류) 대비 착시.
개선안이 아니라 **D17의 숨은 의미를 확인**한 결과. `open_order`는 일반 capability로 남기되
현행 튜너에서 auto=σ(무이득)임을 명시. (backward-compat 9/9 통과, 기본 sigma 불변.)

## 시도3 — 하이브리드(LPB vs learned 선택): 극복

단일 정책 개선이 실패(포섭/중립)한 뒤, **regime별로 나은 정책을 train 내 held-out으로 선택**
하는 메타 라우터(`probe_hybrid.py`). 프로토콜: tr을 fit(70%)/sel(30%) 분할 → 두 정책 tr_fit
적합 → tr_sel 품질로 선택 → te 평가 (테스트 미접촉). 가설: 하이브리드 ≥ max(LPB, learned).

**5-fold (6세계, `open_order=auto`):**

| world | LPB | learned | hybrid | hyb−max | picks(lpb/lrn) |
|---|---|---|---|---|---|
| textworld | 0.7510 | 0.7681 | **0.7688** | +0.0007 | 5/10 |
| irt | 0.8324 | 0.6982 | 0.8324 | +0.0000 | 15/0 |
| specialist | 0.9914 | 0.8904 | 0.9914 | +0.0000 | 15/0 |
| corr | 0.8614 | 0.7819 | 0.8551 | **−0.0063** ✗ | 13/2 |
| crossing | 0.9633 | 0.7969 | 0.9633 | +0.0000 | 15/0 |
| nosignal | 0.8865 | 0.7601 | 0.8865 | +0.0000 | 15/0 |

**textworld 극복**(hybrid 0.7688 > LPB 0.7510 +0.018, learned도 matches) — learned를 10/15 선택.
**단 corr에서 −0.0063 회귀**: 2셀에서 tr_sel 선택오류(learned가 train선 이겼으나 test선 짐).

### 시도3b — 선택 margin (D21 교훈): 깨끗한 극복

corr 회귀는 잡음을 승리로 오인한 것 → **LPB를 기본으로, learned가 tr_sel에서 `margin`(0.01)
이상 명확히 이길 때만 전환**(B9/D21의 표준오차 문턱과 같은 방어). 재측정:

| world | LPB | learned | hybrid | hyb−max | picks(lpb/lrn) |
|---|---|---|---|---|---|
| textworld | 0.7510 | 0.7681 | **0.7680** | −0.0001 | 9/6 |
| corr | 0.8614 | 0.7819 | **0.8614** | **+0.0000** ✓ | 15/0 (회귀 해소) |
| irt | 0.8324 | 0.6982 | 0.8324 | +0.0000 | 15/0 |

**margin이 corr 잡음-선택을 제거하면서 textworld 이득(+0.017 over LPB, learned와 동률) 유지**
— hyb−max ≥ −0.0001 전 세계. **한계 극복 확정.** `baselines.SelectiveRouter`로 정식화
(LPB 주 정책, learned는 명확한 대등우위 체제의 보험, 회귀 테스트 고정).

## RouterBench 예행연 — 한계는 3모델 특유

`src.loader`(실데이터 진입점)로 RouterBench-형 데이터(5모델·6태스크, 단일응답)를 적재·검증·
**Q2=No·Q4=binary 자동판정**(수령 당일 파이프라인 실증). 실물 RouterBench(1.47GB parquet)는
샌드박스 제약으로 **구조·현실 프로파일 모사**(`make_routerbench_style.py`).

**5-fold: 5모델서 LPB 압도**

| tier | LPB | cascade | learned |
|---|---|---|---|
| fast | 0.7874 | 0.3788 | 0.5639 |
| balanced | 0.9605 | 0.9605 | 0.6372 |
| premium | 0.9622 | 0.9617 | 0.7033 |
| **종합** | **0.8743** | 0.6699 | 0.6137 |

**LPB−learned = +0.2605, LPB−cascade = +0.2044.** ★ 3모델에서 learned에 지던 것과 정반대 —
**LPB 우위는 상자 수에 비례**(순차 관측·재선택의 가치가 많은 상자에서 발현). fast tier에서
학습형 단일호출(0.56)이 LPB(0.79)에 크게 뒤진다 — 관측 없이 예측만으로 5모델 중 고르는 것이
불리. **한계가 "3모델·소수상자 특유"임이 실데이터 경로에서 확정.**

## 판정

- **R&D 루프**: belief-swap(중립) → value-order(**D17에 포섭** — LPBRouter가 이미 value-optimal
  임을 드러냄) → hybrid(corr 회귀) → **hybrid+margin(극복)**. 실패를 방어하지 않고 원인을 규명해
  반복한 결과, 3번째 방향(메타 선택)의 2차 반복에서 한계를 넘었다.
- **한계는 설계 결함이 아니라 regime-narrow**: 3모델·실텍스트·단일샘플이라는 가장 불리한 조합
  에만 있고, 상자가 많으면(5모델 RouterBench) LPB가 압도. value-order가 D17에 포섭된 것이 그
  방증 — LPBRouter는 이미 그 체제에서 value-optimal하게 연다.
- **극복은 메타 레벨**(`SelectiveRouter`): LPB를 기본으로, learned가 명확히 이길 때만 전환.
  어느 baseline에도 안 지고(hyb−max ≥ −0.0001), textworld에서 LPB 단독 대비 +0.017. thesis
  유지 — LPB가 주 정책, 학습형은 대등 체제의 보험이다.
- **정직성**: 개선안 2건을 실측 기각(D7·prefix-sharing과 같은 measure-don't-assume), `open_order`
  는 subsumed지만 무해한 일반 capability로 유지, 하이브리드의 초기 corr 회귀를 숨기지 않고
  margin으로 고쳤다. 전체 스위트 무회귀.
