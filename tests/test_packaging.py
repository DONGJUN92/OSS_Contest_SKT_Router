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
