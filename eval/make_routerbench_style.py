"""RouterBench-스타일 데이터 생성 (Phase 15 예행연 — 실제 RouterBench 값이 아니라 **구조·
현실적 프로파일 모사**임을 명시).

실제 RouterBench(Hu et al. 2024): 30k+ 프롬프트 × 11 LLM, 각 (프롬프트,모델)에 응답·비용·
성능점수. SKT 과제와 동일 구조(사전계산 출력 선택). 실물은 1.47GB parquet + pyarrow 필요라
샌드박스에서 비현실적 → 동일 **long 포맷**(query_id·prompt·domain·model·quality·verifier·
tokens)으로 현실적 모델×태스크 프로파일을 모사해 `src.loader`가 그대로 적재하게 한다.

현실성 근거: RouterBench 태스크(GSM8K/MMLU/MBPP/HellaSwag/Winogrande/MT-Bench)와 모델 사다리
(소형 7B ~ GPT-4급)의 태스크별 정확도·비용 차이를 대략 반영. 각 모델은 태스크별 능력이 다르고
(코드 특화 등), 프롬프트 난이도가 정답률을 좌우한다. **단일 응답(S=1)** = RouterBench와 동일
(Q2=No 자동 판정).

사용법: python eval/make_routerbench_style.py [출력경로]
"""
import sys, json, pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

TASKS = ["gsm8k", "mmlu", "mbpp", "hellaswag", "winogrande", "mtbench"]

# 모델 사다리 (이름·상대비용·태스크별 기본 능력) — RouterBench류 현실적 프로파일 모사.
# skill[model][task] ∈ 능력(로짓 오프셋). 비용은 1k토큰당 상대가(소형→GPT-4급).
MODELS = {
    #                prefill  decode   avg_out   [gsm8k, mmlu, mbpp, hella, wino, mtbench]
    "mistral-7b":   dict(pf=0.02, dc=0.04, out=180, skill=[-0.3, 0.1, -0.6, 0.4, 0.3, -0.2]),
    "mixtral-8x7b": dict(pf=0.06, dc=0.12, out=210, skill=[0.4, 0.6, 0.2, 0.7, 0.6, 0.3]),
    "llama2-70b":   dict(pf=0.10, dc=0.20, out=230, skill=[0.2, 0.5, 0.0, 0.8, 0.7, 0.4]),
    "gpt-3.5":      dict(pf=0.15, dc=0.30, out=200, skill=[0.6, 0.7, 0.7, 0.6, 0.6, 0.6]),
    "gpt-4":        dict(pf=0.60, dc=1.20, out=260, skill=[1.6, 1.5, 1.5, 1.2, 1.1, 1.4]),
}


def generate(n_queries=1800, seed=42, verifier_noise=0.18):
    rng = np.random.default_rng(seed)
    records = []
    model_ids = list(MODELS)
    for q in range(n_queries):
        task = TASKS[rng.integers(len(TASKS))]
        t = TASKS.index(task)
        difficulty = float(rng.normal(0.0, 1.0))          # 프롬프트 난이도(로짓, 라벨 — 텍스트 미노출)
        # 난이도를 **명시하지 않고** 길이로 약하게 인코딩 (현실적: 어려운 문제일수록 길다 + 잡음).
        # 인코더는 task(대괄호)와 대략적 난이도(길이)를 얻는다 — 완벽하지 않은 실제 특성 모사.
        base_len = int(np.clip(rng.lognormal(4.6, 0.5) * (1.0 + 0.35 * difficulty), 30, 2000))
        in_tok = base_len
        body = " ".join(f"tok{rng.integers(9999)}" for _ in range(max(base_len // 8, 3)))
        prompt = f"[{task}] {body}"
        for mid in model_ids:
            m = MODELS[mid]
            p_correct = float(1 / (1 + np.exp(-(m["skill"][t] - difficulty))))
            p_correct = np.clip(p_correct, 0.02, 0.98)
            quality = float(rng.uniform() < p_correct)
            verifier = float(np.clip(quality + rng.normal(0, verifier_noise), 0.0, 1.0))
            out_tok = int(np.clip(rng.normal(m["out"], m["out"] * 0.25), 20, None))
            records.append({
                "query_id": f"q{q:05d}", "prompt": prompt, "domain": task, "model": mid,
                "sample": 0, "quality": quality, "verifier": round(verifier, 4),
                "in_tokens": in_tok, "out_tokens": out_tok,
                "output": f"answer to {task} #{q}",     # 스텁 (품질·검증기는 직접 제공)
            })
    return records, model_ids


def main(path):
    records, model_ids = generate()
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[make_routerbench_style] {len(records)}건 → {p}")
    print(f"  모델 {len(model_ids)}: {model_ids}")
    print(f"  태스크 {len(TASKS)}: {TASKS}")
    print(f"  포맷: long JSONL (query_id·prompt·domain·model·quality·verifier·tokens), 단일응답(S=1)")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "eval" / "results" / "routerbench_style.jsonl")
    main(out)
