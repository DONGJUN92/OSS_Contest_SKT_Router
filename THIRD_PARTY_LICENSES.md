# 제3자 라이선스 및 데이터 출처

이 문서는 2026 오픈소스 개발자대회 **2차 심사의 라이선스 검증**(라이선스 충돌 여부 및 위반
사항 검증)에 대응하기 위한 의존성·데이터 출처 명세입니다.

**본 저작물 라이선스: Apache License 2.0** (`LICENSE`)

---

## 1. 런타임 의존성 (기본 경로 — 라우터가 실제로 쓰는 것)

| 패키지 | 라이선스 | Apache-2.0 배포와의 호환 | 용도 |
|---|---|---|---|
| numpy (>=1.26,<2.0) | BSD 3-Clause | 호환 (permissive) | 전 수치 연산 |
| scipy (>=1.11) | BSD 3-Clause | 호환 | `betainc`(Beta 상금), `beta.ppf`(인증서) |
| pyyaml (>=6.0) | MIT | 호환 | `config*.yaml` 판독 |
| matplotlib (>=3.8) | PSF-based (BSD 계열) | 호환 | 제출용 그림 생성 (라우터 경로 아님) |

라우터의 **추론 경로**는 numpy·scipy 만 사용합니다. 즉 배포 산출물에 카피레프트(GPL/AGPL)
의존성이 없습니다.

## 2. 선택 의존성 (extras — 기본 설치에 포함되지 않음)

| 패키지 | 라이선스 | 비고 |
|---|---|---|
| sentence-transformers | Apache-2.0 | `TextEncoder` 의 `st` 백엔드. **기본은 `hashing`** 이라 미설치로도 전 경로 동작 |
| optuna | MIT | `src/tune.py` 선택 백엔드. 기본은 의존성 0 내장 탐색기 |
| pytest | MIT | 개발용 |
| pandas / pyarrow | BSD 3-Clause / Apache-2.0 | 실벤치 재현(`eval/routerbench_real.py`) 전용 |
| huggingface_hub | Apache-2.0 | 실벤치 데이터 다운로드 전용 |

**주의**: `sentence-transformers` 의 `st` 백엔드가 내려받는 **모델 가중치**는 패키지와 별도
라이선스입니다. 기본값 `all-MiniLM-L6-v2` 는 Apache-2.0 이지만, 다른 백본으로 교체할 때는
그 모델의 라이선스를 확인해야 합니다. 챌린지 제출 구성은 `hashing` 백엔드(의존성·다운로드 0)를
기본으로 하므로 이 문제를 회피합니다.

## 3. 데이터 출처

| 데이터 | 출처 | 라이선스/조건 | 저장소 포함 여부 |
|---|---|---|---|
| 합성 검증 세계 A~E | 본 저장소 생성기(`src/synth*.py`) | Apache-2.0 (본 저작물) | 코드만 (결정론 재생성) |
| World F textworld | 본 저장소 생성기(`src/textworld.py`) | Apache-2.0 | 코드만 |
| RouterBench 0-shot | HuggingFace `withmartian/routerbench` | 원 배포자 조건에 따름 | **미포함** — 실행 시 다운로드 |
| SKT 챌린지 공개/비공개 데이터 | 주최측 | 주최측 조건 | **미포함** |

RouterBench 는 재현용 외부 벤치마크이며 **저장소에 재배포하지 않습니다**. 파생 산출물은
`eval/results/routerbench_*.json`(집계 수치)뿐이고 원문·응답 텍스트를 포함하지 않습니다.

## 4. 인용한 선행 연구

코드가 구현하거나 대조하는 논문입니다. 구현은 본 저작물이며 원문 코드를 복제하지 않았습니다.

- Weitzman (1979) *Optimal Search for the Best Alternative* — 예약지수 정책의 원전
- **Belloni, Chen, Wei (2026) arXiv:2606.07392 — *Online Pandora's Box for Contextual LLM
  Cascading*.** ★ **선행연구 우선권 고지**: 이 논문(2026-06-05 투고)이 본 프로젝트의 출발
  명제 — "LLM API 질의 = 상자 개봉, 추론비용 = 검사비용, 배포할 출력 선택 = keep-best-opened"
  라는 Pandora's Box 동형성 — 을 **먼저 발표했다.** 문맥 Weitzman 지수와 output-mediated
  feedback(배포한 출력의 보상만 관측)까지 같고, Õ(√T) regret 을 증명한다. 우리는 이 논문을
  **본 저장소 개발 이후(2026-07-27 외부 감사)에 인지**했으며, 따라서 동형성 자체는
  본 프로젝트의 신규 기여가 아니다.
  **남는 차이**(그 논문에 없는 것, 우리가 가진 것): ① tier 예산이라는 **경성 제약**과
  그 shadow price 페이싱(그 논문은 비용을 보상에서 차감할 뿐 예산 제약이 없다)
  ② **IRT 잠재특성 신념 + 상관 상자 사후갱신**(그 논문은 상자별 출력분포로만 이질성을 표현)
  ③ **구현·실증·오픈소스 라우터**(그 논문은 수치실험이 전혀 없는 이론 논문)
- Kalayci, Raman, Dughmi (2025) arXiv:2510.01394 — 추론시 정지에 Weitzman/Pandora 적용
  (best-of-N 대비 생성 수 절감). 순차 정지 규칙의 인접 선행연구
- Balseiro, Lu, Mirrokni (ICML 2020) arXiv:2002.10421 — 쌍대 mirror descent 페이싱.
  ⚠ 그 O(√T) 는 i.i.d. 확률적 도착 + 자원이 T 에 비례 확장을 가정하며, 적대적 도착에서는
  고정 경쟁비만 보장한다 — 본 저장소는 이 보증을 **전이 주장하지 않는다**(SUBMISSION §5)
- Angelopoulos et al. (ICLR 2024) arXiv:2208.02814 — Conformal Risk Control
- Ong et al. (ICLR 2025) arXiv:2406.18665 — RouteLLM (학습형 단일호출 비교군)
- Dekoninck et al. (2025) arXiv:2410.10347 — **결합 라우팅+캐스케이딩**
  (`baselines.CascadeRouting` 이 이 계열의 1스텝 lookahead 결정 규칙을 구현)
- **Madaan et al. (NeurIPS 2024) arXiv:2310.12963 — AutoMix.** 자기검증(entailment) +
  POMDP 메타검증기로 **검증기 주도 순차 라우팅**을 하는 가장 가까운 선행연구. 본 저장소의
  `ScoringVerifier` → 정지 판정 구조와 대비되며, 초판에는 인용도 비교군도 없었다
- Chen, Zaharia, Zou (2023) arXiv:2305.05176 — FrugalGPT (`StaticCascade` 계열의 원전)
- Ding et al. (ICLR 2024) arXiv:2404.14618 — Hybrid LLM (임계값 기반 이진 라우팅)
- Hu et al. (2024) arXiv:2403.12031 — RouterBench (실벤치 및 비교 방법론)
- **Song et al.** (ACL 2025 Main) arXiv:2506.01048 — IRT-Router (IRT × 라우팅 선행).
  *초판은 저자를 "Zhang et al." 로 잘못 적었다 — 정정한다.* 20 LLM · 12 데이터셋 규모의
  MIRT 라우터로, **IRT 를 라우팅에 쓰는 것 자체도 본 프로젝트의 신규 기여가 아니다**
- Wu et al. (ICLR 2025) arXiv:2408.00724 — Inference Scaling Laws (투표 상자 상금 상한)
- arXiv:2301.13534 (Gergatsouli & Tzamos, NeurIPS 2023) — 상관 Pandora's Box.
  ⚠ 정확히는 "상관이면 지수 정책을 버려야 한다"가 아니라, **다항 표본으로 Weitzman 규칙이
  상수근사로 동작한다**는 결과다. 본 저장소가 "NP-hard 영역"이라 쓴 것은 과장이며,
  엄밀한 경성 결과는 **비의무 검사(non-obligatory inspection)** 변형에 붙는다 — 그리고
  챌린지 규칙은 "호출한 출력 중에서만 선택"이므로 **검사가 의무**라 그 변형이 아니다

## 5. 확인 방법

```bash
pip install pip-licenses && pip-licenses --format=markdown
```

기본 설치(`pip install -e .`)에서 카피레프트 라이선스가 나오면 회귀입니다 — 이슈로 알려 주세요.
