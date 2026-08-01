# AI 모델 활용 및 라이선스 기술 명세서 (결과보고서 붙임2 대응)

대상: **LPB-Router** (`github.com/DONGJUN92/OSS_Contest_SKT_Router`)
근거: 2026 오픈소스 개발자대회 「운영규정」 **제9조(AI 모델 활용의 기준)** · 별표2
작성 2026-08-01 · 검증 방법은 §5

---

## 1. 결론 요약

> **제출 기본 구성에 탑재된 제3자 AI 모델: 0건.**
> 라우터의 추론 경로는 `numpy`·`scipy` 만 사용하며, 프롬프트 인코더 기본값은
> **학습된 가중치가 없는 결정론적 해시 함수**(`HashingEncoder`)다.
> 외부 AI API 호출은 **코드 전체에 0건**이다.

선택적으로 켤 수 있는 사전학습 임베딩 모델 4종은 **전부 오픈웨이트**이며 로컬에서 직접
구동된다(제9조 제2항 제1호 다목의 '독립 구동 가능성' 충족). 상용 API 전용 임베딩
(OpenAI Embedding 등)은 코드 어디에도 없고, 사용할 수 없도록 백엔드 목록이 닫혀 있다.

---

## 2. 제출물이 포함하는 "모델"의 전수 목록

### 2.1 자체 학습 모델 — 본 저작물이 공개 데이터로 직접 적합한다 (제3자 모델 아님)

| 구성요소 | 파일 | 형태 | 학습 데이터 | 가중치 출처 |
|---|---|---|---|---|
| 다집단 2PL IRT / GRM / Beta 반응모형 | `src/irt/*.py` | 주변최대우도(MML) 적합 | **주최측 공개 데이터** | 실행 시 생성 (결정론) |
| Amortized IRT 인코더 (θ 사전분포 헤드) | `src/encoder.py` | 선형/tanh 헤드 | 동상 | 실행 시 생성 |
| 판별식 예측기 `DiscLR` (비교군) | `src/encoder.py` | 모델별 로지스틱 회귀 | 동상 | 실행 시 생성 |
| 검증기 채널 보정 `NoiseModel`·Platt | `src/engine/pandora.py` | 로지스틱 보정 | 동상 | 실행 시 생성 |
| 디코드 길이 추정기 | `src/cost_model.py` | ridge 회귀 (닫힌형) | 동상 | 실행 시 생성 |
| 프롬프트 군집기 (pseudo-domain) | `src/cluster.py` | k-means | 동상 | 실행 시 생성 |
| 출력 채점기 `ScoringVerifier` | `src/verifier.py` | 특성 기반 + 로지스틱 | 동상 | 실행 시 생성 |

**공개 수준**: 이들은 **사전학습 제3자 모델이 아니라 제출 코드가 만들어내는 파라미터**다.
구조·학습 코드·데이터 처리가 전부 저장소에 있고 시드 고정 결정론이므로, 오픈웨이트를
넘어 **완전 공개(오픈소스)** 수준이다. 별도 라이선스 제약이 없다(본 저작물 Apache-2.0).

### 2.2 프롬프트 인코더 — **기본값은 AI 모델이 아니다**

| 백엔드 | 기본값 | 실체 | 가중치 | 네트워크 |
|---|---|---|---|---|
| `hashing` (기본, `HashingEncoder`) | ✅ | 문자 3-gram **signed feature hashing** (blake2b) + 수치 4특성 | **없음** | **없음** |
| `st` (`STEncoder`, 선택) | ✗ | sentence-transformers 백본 | 오픈웨이트 (아래 §2.3) | 최초 1회 가중치 **다운로드** |

`HashingEncoder` 는 해시 함수이므로 학습·추론·가중치 개념이 없다. **제9조의 "AI 모델"에
해당하지 않는다.** 대회 제출 기본 구성은 이것이다.

### 2.3 선택적 사전학습 임베딩 모델 (활성화 시에만 탑재 — 붙임2 기재 대상)

`--with-st` 또는 `get_encoder("st:...")` 로 **명시적으로 켤 때만** 사용된다.
라이선스는 2026-08-01 **HuggingFace Hub API 로 직접 조회해 확인**했다(§5).

| 모델 | 라이선스 | 공개 수준 | 로컬 구동 | 용도 |
|---|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | **Apache-2.0** | 오픈웨이트 | ✅ | `st` 기본 (영어) |
| `BAAI/bge-m3` | **MIT** | 오픈웨이트 | ✅ | `st:multilingual` 1순위 (한국어 대응) |
| `intfloat/multilingual-e5-small` | **MIT** | 오픈웨이트 | ✅ | 2순위 폴백 |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | **Apache-2.0** | 오픈웨이트 | ✅ | 3순위 폴백 |

- 전부 **오픈웨이트 이상**(가중치 공개·재배포 허용) → 제9조 제1항 충족
- 전부 **로컬/자체 서버 직접 구동** → 별표2 '독립 구동 가능성' 충족
- 전부 **permissive**(MIT/Apache-2.0) → 본 저작물 Apache-2.0 배포와 충돌 없음 (제9조 제3항)
- 저장소는 **가중치를 재배포하지 않는다** — 사용자가 원 배포처에서 직접 내려받는다

### 2.4 후보 LLM (A.X 3종)

라우터가 **호출 대상으로 지정**하는 모델이며 **주최측이 제공**한다. 제출물에 탑재되지
않고, 비공개 채점에서는 선택한 action 에 대응하는 사전 계산 결과로 점수가 산정되므로
제출물이 모델 가중치를 포함하거나 호출하지 않는다.

---

## 3. 외부 호출 전수 조사

### 3.1 라우터 추론 경로 — 외부 호출 **0건**

`requests`·`urllib`·`httpx`·`aiohttp`·`socket`·`openai`·`anthropic`·`cohere`·
`google.generativeai`·`boto3`·`azure` — **저장소 전체에서 0건**(§5 재현 명령).
추론 경로 import 는 `numpy`·`scipy`·표준 라이브러리뿐이며, 회귀 테스트가 이를 고정한다.

### 3.2 네트워크가 발생하는 지점 (전부 추론 경로 밖·선택 사항)

| 지점 | 무엇을 받는가 | AI 모델인가 | 발동 조건 |
|---|---|---|---|
| `src/text_encoder.py::STEncoder` | 임베딩 **모델 가중치** | **예** (§2.3, 오픈웨이트) | `st` 백엔드를 명시할 때만 |
| `eval/routerbench_real.py`·`probe_verifier_real.py` | RouterBench **데이터셋** | 아니오 (평가용 데이터) | `reproduce.py --bench` 일 때만 |

두 경우 모두 **평가·설정 단계**의 자원 내려받기이며, 채점 중 라우터가 외부 서비스를
호출하는 것이 아니다. 챌린지 규칙("외부 모델·API·네트워크 서비스 호출 금지 — 로컬 코드로
동작해야 하는 라우터")과도 충돌하지 않는다.

---

## 4. 이번 점검에서 **고친 것** (자세를 문서가 아니라 코드로 맞췄다)

점검 전에도 규정 위반은 없었으나, **선언한 자세와 실제 기본값이 어긋난 곳이 두 군데** 있었다.

| # | 문제 | 조치 |
|---|---|---|
| 1 | `requirements.txt` 가 `sentence-transformers` 를 **기본 설치**에 포함 — README 안내 명령이 `pip install -r requirements.txt` 이므로 **기본 설치가 실제로 임베딩 모델 스택(torch·transformers)을 끌어왔다.** 문서는 "기본은 hashing, 의존성 0"이라 적고 있었다 | 기본 설치에서 제거하고 `pip install -e ".[st]"` 로 분리. 근거를 파일 안에 명시 |
| 2 | `encoder_scan.default_candidates()` 가 `st:multilingual` 을 기본 후보로 넣어, `--scan-encoders` 를 켜면 **`BAAI/bge-m3`(≈2.2GB)를 자동 다운로드**했다 (실측 확인) | 기본은 해싱 후보만. 사전학습 임베딩은 **`--with-st` 로 명시**해야 포함 |

두 조치의 효과: **기본 제출 구성이 "탑재된 제3자 AI 모델 0건" 상태를 유지**하고,
임베딩 모델을 쓰는 경우에는 운영자가 그것을 **의식적으로 켜고 이 문서에 적게** 된다.

---

## 5. 검증 재현 (심사자가 직접 확인 가능)

```bash
# ① 외부 AI API 호출 0건 확인 (출력이 없어야 정상)
grep -rn "import requests\|import urllib\|httpx\|aiohttp\|openai\|anthropic\|cohere\|boto3\|azure" --include=*.py .

# ② 기본 인코더에 가중치가 없음을 확인 (HashingEncoder, 네트워크 없음)
python -c "from src.text_encoder import get_encoder; e=get_encoder('hashing'); print(type(e).__name__, e.encode(['테스트'])[0].shape)"

# ③ 기본 설치 의존성에 AI 모델 스택이 없음을 확인
pip install -e . && pip list | grep -iE "torch|transformers|sentence" ; echo "(출력 없으면 정상)"

# ④ 선택 임베딩 모델의 라이선스 조회 (본 문서 §2.3 의 출처)
python -c "
from huggingface_hub import HfApi; a=HfApi()
for m in ['sentence-transformers/all-MiniLM-L6-v2','BAAI/bge-m3','intfloat/multilingual-e5-small','sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2']:
    print(m, a.model_info(m).card_data.get('license'))"

# ⑤ 규칙 불변식 회귀 (외부 호출 금지·답은 호출한 모델 중 하나 등)
python -m pytest tests/ -q
```

§2.3 표의 라이선스는 위 ④ 명령의 2026-08-01 실행 결과다:
`apache-2.0` / `mit` / `mit` / `apache-2.0`.

---

## 6. 제9조 항목별 대조

| 규정 | 요구 | 본 출품작 | 판정 |
|---|---|---|---|
| 제9조 제1항 | 탑재·적용 AI 모델은 **최소 오픈웨이트** | 기본 구성 탑재 모델 0건. 선택 임베딩 4종 전부 오픈웨이트 | **충족** |
| 제9조 제2항 제1호 다목 | **독립 구동 가능성** (상용 API 전용 불가) | 상용 API 호출 0건. 선택 임베딩은 로컬 구동 | **충족** |
| 제9조 제3항 | 모델 **라이선스·이용약관** 별도 확인 | §2.3 에 HF API 직접 조회로 확인·기재. 전부 MIT/Apache-2.0 | **충족** |
| 제9조 제5항 | 개발 과정의 상용 AI 보조 활용은 허용 | 코드 작성·리뷰 보조로 활용 (제출물에 탑재 아님) | 해당·허용 |
| 붙임2 기재 | 임베딩 모델 탑재 시 명세서 제출 | **이 문서**. 기본 구성은 탑재 0건이므로 "해당 없음"으로 제출 가능하며, `--with-st` 를 켠 경우 §2.3 표를 그대로 기재 | **대응 완료** |

**미해결·주의 1건**: `--with-st` 를 켜서 임베딩 모델을 실제로 채택하면 붙임2 기재 대상이
되고, 심사 환경이 오프라인이면 가중치 캐시를 함께 제출하거나 해싱 구성으로 되돌려야 한다.
현재 기본값은 해싱이므로 **아무 조치 없이도 제출 가능한 상태**다.
