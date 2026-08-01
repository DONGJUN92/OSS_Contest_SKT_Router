"""패키징 회귀 (Phase 17, 레드팀 지적 #8).

지적: `pyproject.toml` 은 `src*`·`baselines*` 만 패키징하는데 `src/router.py` 가 루트의
실험 스크립트 `phase2_stages` 를 import 했고, 그 스크립트는 모듈 로드 시점에
`run_phase2` → `config.yaml` **파일 읽기**까지 끌어왔다. 결과적으로 설치된 패키지에서
`LPBRouter.fit()` 이 ImportError 로 죽었다 — 오픈소스 재사용성의 직접 타격.

여기서 고정하는 불변량 3개
  ① 배포 모듈(`src.*`, `baselines.*`)은 실험 스크립트 없이 import 된다
  ② 배포 모듈은 `config.yaml` 파일 없이 동작한다 (설정은 dict 로 주입)
  ③ 하위호환: `phase2_stages.IRTEncoder is src.encoder.IRTEncoder`
"""
import subprocess
import sys
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 배포에 필요한 최소 설정 — 파일이 아니라 dict 이다 (실데이터 수령 시에도 같은 형태).
# fast frac 은 "전 쿼리에 최저가 1회"가 실현 가능한 수준으로 잡는다 (그보다 낮으면 어떤
# 정책이든 파산이 정답이고, 그것은 패키징 테스트가 검증할 대상이 아니다).
MIN_CFG = {
    "tiers": {"fast": {"budget_frac": 0.5, "weight": 0.5},
              "balanced": {"budget_frac": 0.7, "weight": 0.3},
              "premium": {"budget_frac": 0.95, "weight": 0.2}},
}


def _tiny_dataset(n=240, M=3, d=6, seed=0):
    """합성 시뮬레이터(`src.synth`, config 의존)를 거치지 않는 손제작 Dataset."""
    from src.schema import Dataset, ModelMeta
    rng = np.random.default_rng(seed)
    ids = [f"m{k}" for k in range(M)]
    models = {mid: ModelMeta(mid, 0.02 * (k + 1), 0.04 * (k + 1))
              for k, mid in enumerate(ids)}
    theta = rng.normal(size=n)
    b = np.array([0.8, 0.0, -0.8])[:M]
    p = 1.0 / (1.0 + np.exp(-(theta[:, None] - b[None, :])))
    q = (rng.uniform(size=(n, M)) < p).astype(float)
    feats = np.column_stack([theta + rng.normal(0, 0.3, n),
                             rng.normal(size=(n, d - 1))])
    return Dataset(model_ids=ids, models=models,
                   domains=np.zeros(n, dtype=int), features=feats,
                   in_tokens=np.full(n, 50), out_tokens=np.full((n, M), 120),
                   quality=q, verifier=np.clip(q + rng.normal(0, 0.2, (n, M)), 0, 1),
                   world="tiny")


def test_deployment_modules_import_without_experiment_scripts():
    """`src.router`/`src.submission` 이 phase2_stages·run_phase* 없이 import 되는가."""
    code = (
        "import sys\n"
        "BLOCKED = ('phase2_stages', 'run_phase2', 'run_phase3', 'run_phase4',\n"
        "           'run_phase5', 'run_phase6')\n"
        "class Block:\n"
        "    def find_module(self, name, path=None):\n"
        "        return self.find_spec(name, path)\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name in BLOCKED:\n"
        "            raise AssertionError('배포 모듈이 실험 스크립트를 import 했다: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "import src.router, src.submission, src.verifier, baselines.policies\n"
        "assert src.router.LPBRouter is not None\n"
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, f"import 실패:\n{r.stdout}\n{r.stderr}"
    assert "OK" in r.stdout


def test_lpbrouter_fits_without_config_file():
    """설정 파일 없이 dict 설정 + 손제작 Dataset 으로 fit→policy→route 가 도는가."""
    from src.router import LPBRouter
    from src.harness import run_tier, tier_budgets
    ds = _tiny_dataset()
    tr, te = np.arange(60, 240), np.arange(0, 60)
    router = LPBRouter(MIN_CFG, 1, seed=0, use_domain=False).fit(ds, tr, "fast")
    b_te = tier_budgets(ds, te, MIN_CFG)["fast"]
    res = run_tier(ds, te, router.policy(), "fast", b_te)
    assert res.total_cost <= b_te + 1e-9
    assert 0.0 <= res.mean_quality <= 1.0
    assert res.unanswered == 0, "배포 경로가 파산했다 (설정 주입만으로 동작해야)"
    assert res.calls_per_query >= 1.0 - 1e-9


def test_phase2_stages_shim_is_same_object():
    """하위호환 shim — 기존 실행기/프로브가 무변경 동작해야 한다."""
    import phase2_stages
    from src import encoder
    assert phase2_stages.IRTEncoder is encoder.IRTEncoder
    assert phase2_stages.DiscLR is encoder.DiscLR
    assert phase2_stages.C_PROBIT == encoder.C_PROBIT


# ═══════ 대회 운영규정 제9조 (AI 모델 활용의 기준) — 컴플라이언스 불변식 ═══════
#
# 규정 요지: 출품작에 탑재·적용되는 모든 AI 모델은 최소 오픈웨이트여야 하고, 판단 기준은
# **로컬/자체 서버에서 직접 구동 가능한가**다. 상용 API 전용 모델(임베딩 포함)은 불가.
#
# 이 저장소는 위반이 없었지만, **선언한 자세와 실제 기본값이 어긋난 곳이 두 군데** 있었다
# (requirements.txt 가 임베딩 스택을 기본 설치에 포함 · 인코더 스캔이 bge-m3 를 자동
# 다운로드). 문서로만 지키면 또 어긋나므로 불변식으로 고정한다.
# 상세: docs/ai_model_disclosure.md

def test_no_external_ai_api_anywhere_in_the_repository():
    """상용 AI API 클라이언트가 저장소 어디에도 없어야 한다 (독립 구동 가능성)."""
    import pathlib, re
    root = pathlib.Path(__file__).resolve().parents[1]
    banned = re.compile(
        r"^\s*(import|from)\s+(requests|urllib|httpx|aiohttp|openai|anthropic|cohere|"
        r"boto3|azure|google\.generativeai)\b", re.M)
    hits = []
    for f in root.rglob("*.py"):
        if any(p in f.parts for p in (".git", "__pycache__", ".venv", "build", "dist")):
            continue
        m = banned.search(f.read_text(encoding="utf-8", errors="ignore"))
        if m:
            hits.append(f"{f.relative_to(root)}: {m.group(0).strip()}")
    assert not hits, f"외부 API 클라이언트 발견 (제9조 제2항 제1호 다목 위반 위험): {hits}"


def test_default_install_pulls_no_pretrained_model_stack():
    """기본 설치가 임베딩 모델 스택을 끌어오면 안 된다.

    `requirements.txt` 가 `sentence-transformers` 를 기본에 넣고 있었다 — README 안내
    명령이 `pip install -r requirements.txt` 이므로 기본 설치가 실제로 torch·transformers
    까지 끌어왔다. 문서는 "기본은 hashing, 의존성 0"이라 적고 있었다.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    active = [ln.split("#")[0].strip()
              for ln in (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
              if ln.strip() and not ln.strip().startswith("#")]
    joined = " ".join(active).lower()
    for pkg in ("sentence-transformers", "torch", "transformers", "huggingface"):
        assert pkg not in joined, (
            f"기본 설치에 {pkg} 가 있다 — 제출물의 AI 모델 표면이 불필요하게 넓어진다. "
            f"extras 로 옮길 것 (`pip install -e \".[st]\"`)")


def test_default_encoder_carries_no_learned_weights():
    """기본 인코더는 결정론적 해시 함수여야 한다 (제9조의 'AI 모델'에 해당하지 않음)."""
    from src.text_encoder import get_encoder, HashingEncoder
    e = get_encoder("hashing")
    assert isinstance(e, HashingEncoder)
    assert not hasattr(e, "model"), "기본 인코더가 모델 객체를 들고 있다"
    a = get_encoder("hashing").encode(["한국어 프롬프트 테스트"])
    b = get_encoder("hashing").encode(["한국어 프롬프트 테스트"])
    assert np.array_equal(a, b), "해시 인코더는 결정론적이어야 한다 (학습 상태 없음)"


def test_encoder_scan_does_not_download_a_model_by_default():
    """`--scan-encoders` 기본 후보가 사전학습 모델을 자동으로 내려받으면 안 된다.

    초판은 `st:multilingual` 이 기본 후보라 BAAI/bge-m3(≈2.2GB)를 자동 다운로드했다.
    라이선스 문제는 없으나(오픈웨이트·MIT) ① 기본값이 모델을 끌어오면 "탑재 모델 0건"이라는
    가장 깨끗한 상태를 잃고 붙임2 기재 대상이 되며 ② 오프라인 심사 환경에서 실패한다.
    """
    from src.encoder_scan import default_candidates
    from src.text_encoder import HashingEncoder
    d = default_candidates()
    assert d and all(isinstance(v, HashingEncoder) for v in d.values()), (
        f"기본 후보에 사전학습 모델이 있다: "
        f"{ {k: type(v).__name__ for k, v in d.items()} }")


def test_ai_model_disclosure_document_exists_and_lists_licenses():
    """붙임2(AI 모델 활용 및 라이선스 기술 명세서) 대응 문서가 있어야 한다."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    doc = root / "docs" / "ai_model_disclosure.md"
    assert doc.exists(), "제9조 대응 명세서가 없다"
    txt = doc.read_text(encoding="utf-8")
    for model in ("all-MiniLM-L6-v2", "BAAI/bge-m3", "multilingual-e5-small"):
        assert model in txt, f"{model} 이 명세서에 없다"
    for lic in ("Apache-2.0", "MIT"):
        assert lic in txt, f"{lic} 라이선스 표기가 없다"
