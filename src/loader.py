"""실데이터 로더 스켈레톤 + 스키마 검증기 (B2 — PROJECT_PLAN.md §9 P0).

목표는 공개 데이터 수령 **당일** 하네스 연결이다. 이를 위해 세 가지를 코드가 아니라
데이터·설정으로 흡수한다.
  1) 포맷      — JSON / JSONL / CSV / TSV / parquet 자동 판독
  2) 컬럼명    — FieldSpec 값만 바꾸면 되고 로더 본문은 불변
  3) 분기 판정 — validate_dataset()이 데이터에서 Q2(다중 샘플)·Q4(연속 품질)를 직접
                 읽어 docs/branch_decision_table.md의 해당 행을 가리킨다

**산출 계약**: `textworld.to_dataset`과 동일한 `(Dataset, meta)` 쌍 —
ScoringVerifier·TextEncoder·LPBRouter가 무변경으로 소비한다.

    ds, meta = load_dataset(path, CFG, FieldSpec(quality="score"))
    ds.features = get_encoder("hashing").encode(meta["prompts"])
    V, _ = fold_verifier_matrix(feature_matrix(meta), ds.quality, train_idx)
    ds.verifier = V                      # 이후는 Phase 8 경로와 완전히 동일

기대 레코드 (long — 1행 = 쿼리 × 모델 × 샘플):

    {"query_id": "q0001", "prompt": "...", "domain": "math", "model": "tiny-1b",
     "sample": 0, "output": "...", "quality": 1.0,
     "in_tokens": 42, "out_tokens": 88, "verifier": 0.73}

  필수: query_id · prompt · domain · model · quality
  선택: sample(없으면 S=1) · output(없으면 투표 상자·verifier 특성 불가)
        in_tokens/out_tokens(없으면 4자≈1토큰 추정 — Q3 확인 전 임시값)
        verifier(없으면 0 행렬 → fold별 ScoringVerifier가 채움)

wide 포맷(1행 = 쿼리, 모델별 컬럼)은 `wide_to_long()`으로 변환 후 동일 경로.
"""
import csv
import json as jsonlib
import pathlib
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .schema import Dataset, ModelMeta
from .textworld import extract_answer, _norm_answer
from .cost_mirror import vote_prefill_mult

CHARS_PER_TOKEN = 4          # 토큰 추정 관례 (textworld와 동일) — Q3 답변 시 폐기


class SchemaError(ValueError):
    """스키마 위반 — 수령 당일 데이터 문제를 즉시 특정하기 위한 전용 예외."""


@dataclass
class FieldSpec:
    """공개 데이터의 실제 컬럼명 → 정규 스키마 필드. **수령 당일 이 값만 채운다.**"""
    query_id: str = "query_id"
    prompt: str = "prompt"
    domain: str = "domain"
    model: str = "model"
    quality: str = "quality"
    sample: str = "sample"
    output: str = "output"
    in_tokens: str = "in_tokens"
    out_tokens: str = "out_tokens"
    verifier: str = "verifier"


# ---------------- 답 키 (투표 상자 다수결의 동치 판정) ----------------

def answer_key_marker(text: str) -> str:
    """'Final answer:' 류 마커 추출 전용 — 실패 시 __none__ (textworld 관례)."""
    a = extract_answer(text)
    return _norm_answer(a) if a is not None else "__none__"


def answer_key_fulltext(text: str) -> str:
    """기본값: 마커 추출을 먼저 시도하고, 실패하면 출력 전문을 정규화해 비교.

    실데이터의 자유 형식 출력에서 마커 전용 키는 전 샘플을 __none__으로 뭉개
    다수결을 무의미하게 만든다. 전문 폴백은 최악의 경우 vote_share=1/N (정보 없음)
    으로 퇴화할 뿐 라벨을 왜곡하지 않는다. 태스크별 추출기가 있으면 그것으로 교체.
    """
    a = extract_answer(text)
    return _norm_answer(a if a is not None else text)


# ---------------- 판독 ----------------

def _coerce(v):
    """CSV 문자열 → 가능한 경우 수치."""
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return None
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return v
    return v


def read_records(path) -> list[dict]:
    """확장자로 포맷을 판정해 레코드 리스트 반환 (CSV는 수치 자동 캐스팅)."""
    p = pathlib.Path(path)
    if not p.exists():
        raise SchemaError(f"경로 없음: {p}")
    suf = p.suffix.lower()
    if suf in (".jsonl", ".ndjson"):
        return [jsonlib.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines()
                if ln.strip()]
    if suf == ".json":
        obj = jsonlib.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return obj
        for k in ("records", "data", "rows", "items", "samples"):
            if isinstance(obj.get(k), list):
                return obj[k]
        raise SchemaError(f"JSON 최상위가 리스트가 아니고 알려진 키도 없음: {list(obj)[:8]}")
    if suf in (".csv", ".tsv"):
        with p.open(encoding="utf-8-sig", newline="") as f:
            rd = csv.DictReader(f, delimiter="\t" if suf == ".tsv" else ",")
            return [{k: _coerce(v) for k, v in row.items()} for row in rd]
    if suf in (".parquet", ".pq"):
        try:
            import pandas as pd
        except ImportError as e:                       # pragma: no cover - 선택 의존
            raise SchemaError("parquet 판독에는 pandas+pyarrow 필요 "
                              "(또는 JSONL로 덤프 후 재시도)") from e
        return pd.read_parquet(p).to_dict("records")
    raise SchemaError(f"미지원 확장자 {suf!r} — json/jsonl/csv/tsv/parquet")


def wide_to_long(rows: list[dict], model_ids: list[str], spec: FieldSpec = None,
                 quality_fmt: str = "quality_{model}",
                 output_fmt: str = "output_{model}",
                 out_tokens_fmt: str = "out_tokens_{model}") -> list[dict]:
    """wide(1행=쿼리, 모델별 컬럼) → long 레코드. 포맷 문자열만 맞추면 된다."""
    spec = spec or FieldSpec()
    out = []
    for r in rows:
        base = {k: r[k] for k in (spec.query_id, spec.prompt, spec.domain, spec.in_tokens)
                if k in r}
        for mid in model_ids:
            qk = quality_fmt.format(model=mid)
            if qk not in r:
                raise SchemaError(f"wide 컬럼 없음: {qk}")
            rec = dict(base, **{spec.model: mid, spec.quality: r[qk]})
            for fmt, key in ((output_fmt, spec.output), (out_tokens_fmt, spec.out_tokens)):
                col = fmt.format(model=mid)
                if col in r:
                    rec[key] = r[col]
            out.append(rec)
    return out


# ---------------- 조립 ----------------

def _need(rec: dict, key: str, where: str):
    if key not in rec or rec[key] is None:
        raise SchemaError(f"필수 필드 {key!r} 없음 ({where}) — FieldSpec으로 실제 "
                          f"컬럼명을 지정하세요. 보유 키: {sorted(rec)[:10]}")
    return rec[key]


def _as_float(v, where: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError) as e:
        raise SchemaError(f"실수 변환 불가 {v!r} ({where})") from e


def build_dataset(records: list[dict], cfg: dict, spec: FieldSpec = None,
                  Ns: tuple = None, answer_key=None, model_order: list = None,
                  domain_order: list = None, prices: dict = None):
    """long 레코드 → (Dataset, meta).

    Ns          : 투표 상자 크기 (기본 = 관측 샘플 수 S 이하의 (1,3,5)). Q2=No면 (1,)
    answer_key  : 다수결 동치 판정 함수 (기본 answer_key_fulltext)
    model_order : 상자 열 순서 (기본 = config.yaml models 순서, 불일치 시 데이터 순서)
    domain_order: 도메인 라벨→정수 매핑 순서 (기본 = 정렬된 유일값)
    prices      : {model_id: (prefill, decode)} — config에 없는 모델의 단가 공급 (Q3)
    """
    spec = spec or FieldSpec()
    answer_key = answer_key or answer_key_fulltext
    if not records:
        raise SchemaError("레코드 0건 — 경로·포맷을 확인하세요")

    # 1) 쿼리 단위 그룹핑 (등장 순서 보존 = 재현성)
    order_q, qmap, warn = [], {}, []
    for r in records:
        qid = _need(r, spec.query_id, "레코드")
        if qid not in qmap:
            order_q.append(qid)
            qmap[qid] = {"prompt": str(_need(r, spec.prompt, f"쿼리 {qid}")),
                         "domain": _need(r, spec.domain, f"쿼리 {qid}"),
                         "in_tokens": r.get(spec.in_tokens), "cells": {}}
        else:
            # 같은 query_id의 행이 서로 다른 프롬프트·도메인을 들고 있으면 조인 키가
            # 잘못된 것이다 — 첫 행을 조용히 채택하면 전 행렬이 어긋난 채로 흘러간다.
            prev = qmap[qid]
            if str(r.get(spec.prompt, prev["prompt"])) != prev["prompt"]:
                raise SchemaError(f"query_id {qid!r}에 서로 다른 프롬프트 — 조인 키 확인")
            if str(r.get(spec.domain, prev["domain"])) != str(prev["domain"]):
                raise SchemaError(f"query_id {qid!r}에 서로 다른 도메인 "
                                  f"({prev['domain']!r} vs {r.get(spec.domain)!r})")
        mid = str(_need(r, spec.model, f"쿼리 {qid}"))
        s_idx = int(r.get(spec.sample) or 0)
        cell = qmap[qid]["cells"].setdefault(mid, {})
        if s_idx in cell:
            raise SchemaError(f"중복 레코드: query={qid} model={mid} sample={s_idx}")
        cell[s_idx] = r

    # 2) 모델 열 순서와 단가
    seen = sorted({m for v in qmap.values() for m in v["cells"]})
    cfg_models = list(cfg.get("models", {}))
    if model_order is None:
        model_order = [m for m in cfg_models if m in seen] if set(seen) <= set(cfg_models) \
            else seen
    price_map = dict(prices or {})
    for mid in model_order:
        if mid not in price_map:
            meta = cfg.get("models", {}).get(mid)
            if meta is None:
                raise SchemaError(f"모델 단가 미상: {mid!r} — config.yaml의 models에 추가하거나 "
                                  f"prices 인자로 공급 (Phase 0 Q3)")
            price_map[mid] = (meta["prefill_price"], meta["decode_price"])

    # 3) 샘플 수 S와 투표 상자 크기 Ns
    #    셀마다 샘플 수가 다르면 [모델×N표] 상자의 비용(프리필 ×N)과 실제 표 수가
    #    어긋나 조용히 점수를 오염시킨다 — 균일성을 여기서 강제한다.
    sizes = Counter(len(c) for v in qmap.values() for c in v["cells"].values())
    if len(sizes) > 1:
        raise SchemaError(f"셀별 샘플 수가 불균일: {dict(sizes)} — 투표 상자의 비용·표 수가 "
                          f"어긋납니다. 최소값으로 절단하거나 결측 샘플을 보완하세요")
    S = next(iter(sizes))
    for v in qmap.values():
        for mid, c in v["cells"].items():
            if sorted(c) != list(range(S)):
                raise SchemaError(f"샘플 인덱스가 0..{S - 1} 연속이 아님 (model={mid}): {sorted(c)}")
    if Ns is None:
        Ns = tuple(n for n in (1, 3, 5) if n <= S) or (1,)
    if max(Ns) > S:
        raise SchemaError(f"Ns={Ns}가 관측 샘플 수 S={S}를 초과")

    # 4) 도메인 라벨 → 정수
    labels = [qmap[q]["domain"] for q in order_q]
    domain_names = list(domain_order) if domain_order is not None \
        else sorted({str(d) for d in labels}, key=str)
    dom_idx = {str(d): i for i, d in enumerate(domain_names)}
    unknown = {str(d) for d in labels} - set(dom_idx)
    if unknown:
        raise SchemaError(f"domain_order에 없는 라벨: {sorted(unknown)}")

    # 5) 행렬 조립
    n, M, J = len(order_q), len(model_order), len(Ns)
    cols = M * J
    quality = np.zeros((n, cols))
    verifier = np.zeros((n, cols))
    vote_share = np.ones((n, cols))
    out_tokens = np.zeros((n, cols), dtype=np.int64)
    rep_text = [[""] * cols for _ in range(n)]
    model_ids, models = [], {}
    for mid in model_order:
        pf, dc = price_map[mid]
        for N in Ns:
            vid = mid if N == 1 else f"{mid}#v{N}"
            model_ids.append(vid)
            models[vid] = ModelMeta(vid, pf * vote_prefill_mult(cfg, N), dc)   # 프리필 N회(기본), 디코드는 토큰합에 반영

    est_tok, has_v, has_out = 0, False, True
    for i, qid in enumerate(order_q):
        rec_q = qmap[qid]
        for m, mid in enumerate(model_order):
            cell = rec_q["cells"].get(mid)
            if cell is None:
                raise SchemaError(f"(쿼리×모델) 완전 행렬 필요 — query={qid}에 model={mid} 행 없음")
            smp = [cell[k] for k in sorted(cell)]
            outs, quals, toks, vs = [], [], [], []
            for j, r in enumerate(smp):
                o = r.get(spec.output)
                if o is None:
                    has_out = False
                    o = ""
                outs.append(str(o))
                quals.append(_as_float(_need(r, spec.quality, f"{qid}/{mid}/{j}"),
                                       f"{qid}/{mid}/{j}.quality"))
                t = r.get(spec.out_tokens)
                if t is None:
                    t = max(len(outs[-1]) // CHARS_PER_TOKEN, 1)
                    est_tok += 1
                toks.append(int(t))
                v = r.get(spec.verifier)
                has_v = has_v or v is not None
                vs.append(float(v) if v is not None else 0.0)
            keys = [answer_key(o) for o in outs]
            for j, N in enumerate(Ns):
                col = m * J + j                    # 균일 S 강제로 표 수 = N이 보장됨
                maj, c_maj = Counter(keys[:N]).most_common(1)[0]
                match = [u for u in range(N) if keys[u] == maj]   # 다수 답과 일치한 샘플
                rep_text[i][col] = outs[keys[:N].index(maj)]
                vote_share[i, col] = c_maj / N
                quality[i, col] = float(np.mean([quals[u] for u in match]))
                verifier[i, col] = float(np.mean([vs[u] for u in match]))
                out_tokens[i, col] = int(sum(toks[:N]))

    in_tokens = np.array([int(rec_q["in_tokens"]) if rec_q["in_tokens"] is not None
                          else max(len(rec_q["prompt"]) // CHARS_PER_TOKEN, 1)
                          for rec_q in (qmap[q] for q in order_q)], dtype=np.int64)
    if any(qmap[q]["in_tokens"] is None for q in order_q):
        warn.append("in_tokens 부재 → 프롬프트 길이로 추정 (Q3 확인 전 임시값)")
    if est_tok:
        warn.append(f"out_tokens 부재 셀 {est_tok}건 → 출력 길이로 추정 (Q3 확인 전 임시값)")
    if not has_out:
        warn.append("output 부재 → 투표 상자 다수결·ScoringVerifier 특성 사용 불가 "
                    "(verifier 컬럼이 있어야 파이프라인 성립)")
    if not has_v:
        warn.append("verifier 부재 → 0 행렬. fold별 fold_verifier_matrix()로 채울 것")
    if len(Ns) > 1 and has_out:
        deg = float(np.mean(vote_share[:, [m * J + J - 1 for m in range(M)]]))
        if deg <= 1.05 / max(Ns):
            warn.append(f"vote_share가 1/N에 붙음({deg:.3f}) → answer_key가 태스크와 "
                        f"불일치. 태스크별 추출기로 교체 권장")

    ds = Dataset(model_ids=model_ids, models=models,
                 domains=np.array([dom_idx[str(d)] for d in labels], dtype=np.int64),
                 features=np.zeros((n, 1)), in_tokens=in_tokens, out_tokens=out_tokens,
                 quality=quality, verifier=verifier, world="real",
                 extras={"query_ids": order_q, "domain_names": domain_names, "S": S})
    meta = {"prompts": [qmap[q]["prompt"] for q in order_q],
            "tasks": [str(d) for d in labels],
            "rep_text": rep_text, "vote_share": vote_share,
            "query_ids": order_q, "model_ids": model_ids, "base_models": list(model_order),
            "Ns": tuple(Ns), "S": S, "has_output": has_out, "has_verifier": has_v,
            "warnings": warn}
    return ds, meta


def load_dataset(path, cfg: dict, spec: FieldSpec = None, **kw):
    """편의 진입점: 경로 → (Dataset, meta)."""
    return build_dataset(read_records(path), cfg, spec, **kw)


# ---------------- 검증 ----------------

def _finite(a: np.ndarray, name: str):
    if not np.all(np.isfinite(a)):
        raise SchemaError(f"{name}에 NaN/Inf 포함 ({int((~np.isfinite(a)).sum())}개)")


def validate_dataset(ds: Dataset, meta: dict = None, cfg: dict = None,
                     require_features: bool = False) -> dict:
    """필드 존재·형상·범위·정합성 assert. 위반 시 SchemaError, 통과 시 리포트 반환.

    리포트는 데이터에서 직접 읽히는 분기(Q2·Q4)의 판정을 함께 담는다 —
    docs/branch_decision_table.md와 짝을 이룬다.
    """
    n, M = ds.n, ds.m
    if n < 2 or M < 2:
        raise SchemaError(f"규모 부족: n={n}, m={M} (최소 2×2)")
    for name, arr, shape in (("quality", ds.quality, (n, M)),
                             ("verifier", ds.verifier, (n, M)),
                             ("out_tokens", ds.out_tokens, (n, M))):
        if arr.shape != shape:
            raise SchemaError(f"{name} 형상 {arr.shape} ≠ 기대 {shape}")
        _finite(arr, name)
    if ds.in_tokens.shape != (n,):
        raise SchemaError(f"in_tokens 형상 {ds.in_tokens.shape} ≠ ({n},)")
    if ds.features.shape[0] != n:
        raise SchemaError(f"features 행수 {ds.features.shape[0]} ≠ {n}")
    _finite(ds.features, "features")
    missing = [m for m in ds.model_ids if m not in ds.models]
    if missing:
        raise SchemaError(f"ModelMeta 없는 모델: {missing}")
    for lo, hi, name, arr in ((0.0, 1.0, "quality", ds.quality),
                              (0.0, 1.0, "verifier", ds.verifier)):
        if arr.min() < lo - 1e-9 or arr.max() > hi + 1e-9:
            raise SchemaError(f"{name} 범위 이탈 [{arr.min():.4g}, {arr.max():.4g}] ⊄ [{lo},{hi}]"
                              f" — 정규화가 필요하면 로더에서 스케일링하세요")
    if ds.in_tokens.min() <= 0:
        raise SchemaError(f"in_tokens ≤ 0 존재 (min={ds.in_tokens.min()})")
    if ds.out_tokens.min() < 0:
        raise SchemaError(f"out_tokens < 0 존재 (min={ds.out_tokens.min()})")
    d = np.unique(ds.domains)
    if d.min() < 0 or not np.array_equal(d, np.arange(len(d))):
        raise SchemaError(f"domains가 0..D-1 연속 정수가 아님: {d[:10]}")
    if require_features and ds.features.shape[1] < 2:
        raise SchemaError("features 미부착 — TextEncoder.encode(meta['prompts']) 결과를 넣으세요")

    from .cost_mirror import cost_matrix                # 순환 의존 회피용 지연 import
    cm = cost_matrix(ds)
    _finite(cm, "cost_matrix")
    if cm.min() <= 0:
        raise SchemaError(f"비용 ≤ 0 존재 (min={cm.min():.4g}) — 단가·토큰 확인 (Q3)")

    uniq = np.unique(ds.quality)
    binary = bool(np.all(np.isin(uniq, (0.0, 1.0))))
    k = (cfg or {}).get("eval", {}).get("k_folds", 5)
    counts = np.bincount(ds.domains)
    rep = {
        "n_queries": n, "n_boxes": M, "n_domains": int(len(d)),
        "quality_kind": "binary" if binary else "continuous",
        "quality_levels": int(len(uniq)), "base_rate": round(float(ds.quality.mean()), 4),
        "samples_per_cell": int(ds.extras.get("S", 1)) if ds.extras else 1,
        "vote_boxes": list((meta or {}).get("Ns", (1,))),
        "cost_range": [round(float(cm.min()), 6), round(float(cm.max()), 6)],
        "cost_spread": round(float(cm.max() / max(cm.min(), 1e-12)), 2),
        "verifier_attached": bool(np.any(ds.verifier > 0)),
        "features_attached": bool(ds.features.shape[1] > 1),
        "min_domain_count": int(counts.min()) if len(counts) else 0,
        "warnings": list((meta or {}).get("warnings", [])),
    }
    if rep["min_domain_count"] < k:
        rep["warnings"].append(f"최소 도메인 표본 {rep['min_domain_count']} < k_folds={k} "
                               f"— 층화 fold가 비어 편향 가능")
    if rep["cost_spread"] < 2.0:
        rep["warnings"].append(f"모델 간 비용 격차 {rep['cost_spread']}× — 라우팅 여지가 작음 "
                               f"(단가 매핑 오류 가능)")
    rep["branch"] = branch_report(rep)
    return rep


def branch_report(rep: dict) -> dict:
    """데이터에서 직접 판정 가능한 분기(Q2·Q4)를 결정표 행동으로 번역."""
    s, kind = rep["samples_per_cell"], rep["quality_kind"]
    return {
        "Q2_multi_sample": (
            f"YES (S={s}) → 투표 상자 활성: Ns={rep['vote_boxes']} (무변경)" if s > 1 else
            "NO (S=1) → Ns=(1,)로 상자 집합 축소, 정책 무변경 (리스크 R1 대응)"),
        "Q4_quality": (
            "binary → 닫힌형 예약지수 σ=1−λc/p̄ 그대로 (현행)" if kind == "binary" else
            f"continuous ({rep['quality_levels']}수준) → **B4 필요**: Beta 상금 예약값 "
            f"수치해 (E[(Q−σ)⁺]=λc). 미구현 분기"),
    }


def format_report(rep: dict) -> str:
    """수령 당일 콘솔 확인용 한 화면 요약."""
    lines = ["[loader] 스키마 검증 통과",
             f"  규모      n={rep['n_queries']}  상자={rep['n_boxes']}  "
             f"도메인={rep['n_domains']}  샘플/셀={rep['samples_per_cell']}",
             f"  품질      {rep['quality_kind']} ({rep['quality_levels']}수준)  "
             f"기저율={rep['base_rate']}",
             f"  비용      {rep['cost_range']}  격차={rep['cost_spread']}×",
             f"  부착      verifier={rep['verifier_attached']}  "
             f"features={rep['features_attached']}",
             "  분기 판정 (docs/branch_decision_table.md)"]
    lines += [f"    - {k}: {v}" for k, v in rep["branch"].items()]
    if rep["warnings"]:
        lines += ["  경고"] + [f"    ! {w}" for w in rep["warnings"]]
    return "\n".join(lines)


if __name__ == "__main__":                         # python -m src.loader <경로> [k=컬럼명 ...]
    import argparse
    import yaml

    ap = argparse.ArgumentParser(description="공개 데이터 판독 → 스키마 검증 → 분기 판정")
    ap.add_argument("path")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--map", nargs="*", default=[], metavar="필드=컬럼",
                    help="예: --map quality=score model=model_name")
    ap.add_argument("--peek", type=int, default=0, help="원본 레코드 N건 출력 후 종료")
    args = ap.parse_args()

    if args.peek:
        for r in read_records(args.path)[:args.peek]:
            print(jsonlib.dumps(r, ensure_ascii=False)[:400])
        raise SystemExit(0)
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    spec = FieldSpec(**dict(kv.split("=", 1) for kv in args.map))
    ds, meta = load_dataset(args.path, cfg, spec)
    print(format_report(validate_dataset(ds, meta, cfg)))
    print(f"  모델      {meta['base_models']}")
    print(f"  상자      {ds.model_ids}")
