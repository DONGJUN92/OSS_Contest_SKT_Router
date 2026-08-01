"""실데이터 로더 회귀 테스트 (B2).

핵심 검증은 **왕복 동일성**이다: World F를 공개 데이터처럼 레코드로 직렬화한 뒤
로더로 되읽어 `textworld.to_dataset`과 동일한 Dataset이 나오는지 확인한다.
로더가 투표 상자 집계·토큰 합산·가격 스케일링을 원본과 한 치도 다르게 하지 않음을
고정하므로, 수령 당일 로더 결함이 조용히 점수를 오염시키는 사고를 막는다.
"""
import sys, pathlib, json

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import yaml

from src.loader import (FieldSpec, SchemaError, answer_key_marker, build_dataset,
                        load_dataset, read_records, validate_dataset, wide_to_long)
from src.textworld import TASKS, build_textworld, extract_answer, to_dataset, _norm_answer

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
SMALL = dict(CFG, synth=dict(CFG["synth"], n_queries=120))
SPEC = FieldSpec()


def _records_from_textworld(tw):
    """World F → 공개 데이터 형식의 long 레코드 (챌린지 §5 '후보 모델별 출력' 대응)."""
    base_ids = list(CFG["models"])
    recs = []
    for i, prompt in enumerate(tw.prompts):
        truth = _norm_answer(tw.truth[i])
        for m, mid in enumerate(base_ids):
            for s, out in enumerate(tw.samples[i][m]):
                a = extract_answer(out)
                norm = _norm_answer(a) if a is not None else "__none__"
                recs.append({
                    "query_id": f"q{i:05d}", "prompt": prompt,
                    "domain": TASKS[int(tw.domains[i])], "model": mid, "sample": s,
                    "output": out,
                    "quality": float(norm != "__none__" and norm == truth),
                    "in_tokens": max(len(prompt) // 4, 12),
                    "out_tokens": max(len(out) // 4, 8),
                })
    return recs


@pytest.fixture(scope="module")
def world():
    tw = build_textworld(SMALL, seed=11)
    ds_ref, meta_ref = to_dataset(tw, SMALL)
    return tw, ds_ref, meta_ref, _records_from_textworld(tw)


def _load(recs, **kw):
    return build_dataset(recs, SMALL, SPEC, answer_key=answer_key_marker,
                         domain_order=TASKS, **kw)


# ---------- 1. 왕복 동일성 (로더 == 원본 파이프라인) ----------

def test_roundtrip_matches_reference_dataset(world):
    _, ds_ref, meta_ref, recs = world
    ds, meta = _load(recs)
    assert ds.model_ids == ds_ref.model_ids
    assert np.array_equal(ds.domains, ds_ref.domains)
    assert np.array_equal(ds.in_tokens, ds_ref.in_tokens)
    assert np.array_equal(ds.out_tokens, ds_ref.out_tokens)
    assert np.allclose(ds.quality, ds_ref.quality)
    assert np.allclose(meta["vote_share"], meta_ref["vote_share"])
    assert meta["rep_text"] == meta_ref["rep_text"]
    assert meta["prompts"] == meta_ref["prompts"]
    for vid in ds.model_ids:                       # 투표 상자 가격 스케일링 (프리필 ×N)
        assert ds.models[vid].prefill_price == pytest.approx(ds_ref.models[vid].prefill_price)
        assert ds.models[vid].decode_price == pytest.approx(ds_ref.models[vid].decode_price)


# ---------- 2. 포맷 비의존 (JSONL / JSON / CSV 동등) ----------

def test_format_parity_jsonl_json_csv(world, tmp_path):
    _, _, _, recs = world
    recs = recs[:600]
    (tmp_path / "d.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8")
    (tmp_path / "d.json").write_text(json.dumps({"records": recs}, ensure_ascii=False),
                                     encoding="utf-8")
    import csv as _csv
    with (tmp_path / "d.csv").open("w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)

    base = None
    for name in ("d.jsonl", "d.json", "d.csv"):
        ds, meta = load_dataset(tmp_path / name, SMALL, SPEC,
                                answer_key=answer_key_marker, domain_order=TASKS)
        cur = (ds.quality, ds.out_tokens, ds.in_tokens, ds.domains, meta["vote_share"])
        if base is None:
            base = cur
        else:
            for a, b in zip(base, cur):
                assert np.allclose(a, b), f"{name} 포맷 간 불일치"


def test_wide_to_long_equivalent(world):
    _, _, _, recs = world
    single = [r for r in recs if r["sample"] == 0][:400]     # wide는 샘플 1개 가정
    rows, order = {}, []
    for r in single:
        row = rows.setdefault(r["query_id"], {"query_id": r["query_id"],
                                              "prompt": r["prompt"],
                                              "domain": r["domain"],
                                              "in_tokens": r["in_tokens"]})
        if r["query_id"] not in order:
            order.append(r["query_id"])
        row[f"quality_{r['model']}"] = r["quality"]
        row[f"output_{r['model']}"] = r["output"]
        row[f"out_tokens_{r['model']}"] = r["out_tokens"]
    wide = [rows[q] for q in order if len(rows[q]) == 4 + 3 * len(CFG["models"])]
    long = wide_to_long(wide, list(CFG["models"]), SPEC)
    ds_w, _ = _load(long)
    ds_l, _ = _load([r for r in single if r["query_id"] in {w["query_id"] for w in wide}])
    assert np.allclose(ds_w.quality, ds_l.quality)
    assert ds_w.model_ids == ds_l.model_ids


# ---------- 3. Q2 분기 (다중 샘플 미제공 → 상자 집합 축소, 정책 무변경) ----------

def test_single_sample_branch_shrinks_boxes(world):
    _, _, _, recs = world
    ds, meta = _load([r for r in recs if r["sample"] == 0], Ns=(1,))
    assert ds.m == len(CFG["models"]) and meta["Ns"] == (1,)
    assert (meta["vote_share"] == 1.0).all()
    rep = validate_dataset(ds, meta, SMALL)
    assert rep["branch"]["Q2_multi_sample"].startswith("NO")


# ---------- 4. 검증기: 위반을 조용히 통과시키지 않는다 ----------

def test_validator_accepts_clean_and_reports_branches(world):
    _, _, _, recs = world
    ds, meta = _load(recs)
    rep = validate_dataset(ds, meta, SMALL)
    assert rep["quality_kind"] == "binary"
    assert rep["samples_per_cell"] == 5
    assert rep["branch"]["Q2_multi_sample"].startswith("YES")
    assert rep["cost_spread"] > 2.0
    assert rep["features_attached"] is False          # 인코더 부착 전


def test_validator_flags_continuous_quality_as_b4_branch(world):
    _, _, _, recs = world
    ds, meta = _load(recs)
    rng = np.random.default_rng(0)
    ds.quality = rng.uniform(0, 1, size=ds.quality.shape)
    rep = validate_dataset(ds, meta, SMALL)
    assert rep["quality_kind"] == "continuous"
    assert "B4" in rep["branch"]["Q4_quality"]


def test_validator_rejects_out_of_range_and_nan(world):
    _, _, _, recs = world
    ds, meta = _load(recs)
    good = ds.quality.copy()
    ds.quality = good * 5.0
    with pytest.raises(SchemaError, match="범위 이탈"):
        validate_dataset(ds, meta, SMALL)
    ds.quality = good.copy()
    ds.quality[0, 0] = np.nan
    with pytest.raises(SchemaError, match="NaN"):
        validate_dataset(ds, meta, SMALL)
    ds.quality = good
    ds.features = np.zeros((ds.n, 1))
    with pytest.raises(SchemaError, match="features 미부착"):
        validate_dataset(ds, meta, SMALL, require_features=True)


def test_loader_rejects_malformed_records(world):
    _, _, _, recs = world
    with pytest.raises(SchemaError, match="필수 필드"):
        _load([{k: v for k, v in r.items() if k != "quality"} for r in recs[:50]])
    with pytest.raises(SchemaError, match="완전 행렬"):     # 모델 열 자체가 통째로 결측
        _load([r for r in recs[:400] if r["model"] != "mid-7b"] +
              [r for r in recs[:400] if r["model"] == "mid-7b" and r["query_id"] != "q00000"])
    with pytest.raises(SchemaError, match="불균일"):        # 상자 비용 ≠ 실제 표 수
        _load([r for r in recs[:400]
               if not (r["model"] == "mid-7b" and r["sample"] == 0)])
    with pytest.raises(SchemaError, match="중복"):
        _load(recs[:50] + recs[:50])
    with pytest.raises(SchemaError, match="단가 미상"):     # config에 없는 모델 (Q3 미해결)
        _load([dict(r, model="unknown-99b") if r["model"] == "tiny-1b" else r
               for r in recs[:50]])
    with pytest.raises(SchemaError, match="레코드 0건"):
        _load([])
    bad = [dict(r) for r in recs[:50]]                   # 조인 키 오류: 같은 id·다른 프롬프트
    bad[7]["prompt"] = bad[7]["prompt"] + " (다른 문항)"
    with pytest.raises(SchemaError, match="조인 키"):
        _load(bad)
    bad2 = [dict(r) for r in recs[:50]]
    bad2[9]["domain"] = "sort" if bad2[9]["domain"] != "sort" else "arith"
    with pytest.raises(SchemaError, match="서로 다른 도메인"):
        _load(bad2)


def test_read_records_rejects_unknown_format(tmp_path):
    p = tmp_path / "x.xlsx"
    p.write_text("nope", encoding="utf-8")
    with pytest.raises(SchemaError, match="미지원 확장자"):
        read_records(p)


# ---------- 5. 수령 당일 경로: 로더 → 인코더 → verifier → 라우터 → 하네스 ----------

def test_end_to_end_from_loader_to_harness(world):
    """로더 산출물이 Phase 8과 동일한 코드 경로로 끝까지 흐르는지 (계약의 최종 증명)."""
    from src.harness import run_tier, tier_budgets
    from src.text_encoder import get_encoder
    from src.verifier import feature_matrix, fold_verifier_matrix
    from src.router import LPBRouter

    _, _, _, recs = world
    ds, meta = _load(recs)
    validate_dataset(ds, meta, SMALL)
    _enc = get_encoder("hashing")
    ds.features = _enc.encode(meta["prompts"])
    ds.text_encoder = _enc          # D70: 배포 텍스트 경로 계약
    tr, te = np.arange(0, 90), np.arange(90, ds.n)
    ds.verifier, _ = fold_verifier_matrix(feature_matrix(meta), ds.quality, tr)
    validate_dataset(ds, meta, SMALL, require_features=True)

    router = LPBRouter(SMALL, len(TASKS), seed=0).fit(ds, tr, "balanced")
    budget = tier_budgets(ds, te, SMALL)["balanced"]
    r = run_tier(ds, te, router.policy(), "balanced", budget)
    assert r.total_cost <= budget + 1e-9        # 규칙 강제가 로더 경로에서도 유효
    assert 0.0 <= r.mean_quality <= 1.0
    assert r.unanswered == 0
