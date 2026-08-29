"""Baseline runner: evaluate Qwen2.5-Coder-7B-Instruct under different prompt strategies.

Two-phase pipeline (decoupled to keep GPU saturated):
  Phase 1 (GPU): generate LLM responses → save raw codes to outputs/raw_<name>.jsonl
  Phase 2 (CPU, parallel): score raw codes via NetArena's BenchmarkEvaluator → outputs/baseline_<name>.jsonl

The five baselines we run (all on the same base model):
  - zeroshot_base
  - zeroshot_cot
  - fewshot_base       (4 fixed examples baked into the prompt)
  - fewshot_semantic   (top-k retrieved per query, k from C.VISTA_RETRIEVAL_K)
  - sft_only           (LoRA-finetuned base, base prompt)
"""
from __future__ import annotations

import argparse, json, os, sys, time
from typing import Sequence

from loguru import logger

from . import config as C
from . import prompts
from .data import read_jsonl


def _build_messages_for_baseline(name: str, question: str, retriever=None, train_pool=None, exclude_ids: set[int] | None = None):
    if name in ("zeroshot_base", "zeroshot_cot", "fewshot_base", "fewshot_cot"):
        return prompts.build_messages(question, prompt_type=name)
    elif name == "fewshot_semantic":
        assert retriever is not None and train_pool is not None
        idxs = retriever.topk(question, k=C.VISTA_RETRIEVAL_K, exclude_ids=exclude_ids)
        examples = [{"question": train_pool[i]["question"],
                     "answer": _gt_to_process_graph(train_pool[i]["answer"])} for i in idxs]
        return prompts.build_messages(question, prompt_type="fewshot_semantic",
                                      fewshot_examples=examples)
    elif name == "sft_only":
        return prompts.build_messages(question, prompt_type="zeroshot_base")
    else:
        raise ValueError(f"unknown baseline name: {name}")


def _gt_to_process_graph(gt_code: str) -> str:
    return gt_code.replace("ground_truth_process_graph", "process_graph")


def _load_train_pool() -> list[dict]:
    return read_jsonl(C.TRAIN_FILE)


def phase1_generate(
    *,
    name: str,
    test: list[dict],
    raw_out_path: str,
    batch_size: int = 16,
    lora_path: str | None = None,
):
    from .llm_engine import LLMEngine, make_lora_request
    retriever = None
    train_pool = None
    if name == "fewshot_semantic":
        from .retrieval import CaseRetriever
        train_pool = _load_train_pool()
        retriever = CaseRetriever.build_or_load(train_pool)

    enable_lora = lora_path is not None
    engine = LLMEngine(enable_lora=enable_lora)
    lora_req = make_lora_request(lora_path) if enable_lora else None

    logger.info(f"[phase1:{name}] generating {len(test)} responses → {raw_out_path}")
    with open(raw_out_path, "w") as fout:
        for bs in range(0, len(test), batch_size):
            batch = test[bs:bs+batch_size]
            messages_batch = [
                _build_messages_for_baseline(
                    name,
                    ex["question"],
                    retriever,
                    train_pool,
                    exclude_ids={ex["id"]},
                )
                for ex in batch
            ]
            t0 = time.perf_counter()
            outs = engine.generate(messages_batch,
                                   temperature=C.GEN_TEMPERATURE_GREEDY,
                                   top_p=C.GEN_TOP_P,
                                   max_tokens=C.GEN_MAX_TOKENS,
                                   n=1, lora_request=lora_req)
            wall = time.perf_counter() - t0
            per_q_lat = wall / max(1, len(batch))

            for ex, samples in zip(batch, outs):
                raw = samples[0]["text"]
                code = prompts.extract_code_output(raw)
                fout.write(json.dumps({
                    "id": ex["id"], "query": ex["question"],
                    "golden": ex["answer"], "code": code, "label": ex["task_label"],
                    "raw_response": raw, "latency": per_q_lat,
                }, ensure_ascii=False) + "\n")
            if (bs // batch_size) % 4 == 0 or (bs+batch_size) >= len(test):
                logger.info(f"[phase1:{name}] {min(bs+batch_size,len(test))}/{len(test)} batches done")
    logger.info(f"[phase1:{name}] generation complete")


def phase2_score(*, raw_path: str, out_path: str, num_workers: int = 12):
    from .parallel_score import score_records_parallel
    logger.info(f"[phase2] scoring {raw_path} with {num_workers} workers → {out_path}")
    records = []
    with open(raw_path) as f:
        for line in f:
            records.append(json.loads(line))
    t0 = time.perf_counter()
    res = score_records_parallel(records, num_workers=num_workers)
    logger.info(f"[phase2] scored {len(res)} records in {time.perf_counter()-t0:.1f}s")

    pass_correct = pass_safety = total = 0
    sum_lat = 0.0
    with open(out_path, "w") as fout:
        for r, raw in zip(res, records):
            r["baseline"] = raw.get("baseline_name", os.path.basename(raw_path).replace("raw_","").replace(".jsonl",""))
            r["id"] = raw.get("id")
            r["raw_response"] = raw.get("raw_response","")
            r["extracted_code"] = raw.get("code","")
            fout.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")
            total += 1
            if r.get("Result-Correctness") == "Pass": pass_correct += 1
            if r.get("Result-Safety")      == "Pass": pass_safety += 1
            sum_lat += float(r.get("Result-Latency", 0.0))

    summary = {
        "baseline": os.path.basename(out_path).replace(".jsonl","").replace("baseline_",""),
        "n": total,
        "correctness": pass_correct/total if total else 0.0,
        "safety": pass_safety/total if total else 0.0,
        "avg_latency_s": sum_lat/total if total else 0.0,
        "out_path": out_path,
    }
    with open(out_path.replace(".jsonl",".summary.json"),"w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"[phase2] DONE — {summary}")
    return summary


def run_baseline(
    *,
    name: str,
    test_file: str = C.TEST_FILE,
    out_path: str | None = None,
    raw_out_path: str | None = None,
    limit: int | None = None,
    batch_size: int = 16,
    lora_path: str | None = None,
    num_workers: int = 12,
):
    if out_path is None:
        out_path = os.path.join(C.OUTPUTS_DIR, f"baseline_{name}.jsonl")
    if raw_out_path is None:
        raw_out_path = os.path.join(C.OUTPUTS_DIR, f"raw_{name}.jsonl")

    test = read_jsonl(test_file)
    if limit is not None:
        test = test[:limit]

    phase1_generate(name=name, test=test, raw_out_path=raw_out_path,
                    batch_size=batch_size, lora_path=lora_path)
    return phase2_score(raw_path=raw_out_path, out_path=out_path, num_workers=num_workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True,
                    choices=["zeroshot_base","zeroshot_cot","fewshot_base","fewshot_cot",
                             "fewshot_semantic","sft_only"])
    ap.add_argument("--test", default=C.TEST_FILE)
    ap.add_argument("--out", default=None)
    ap.add_argument("--raw-out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--phase", choices=["all","gen","score"], default="all")
    args = ap.parse_args()

    if args.phase in ("all", "gen"):
        if args.phase == "gen":
            test = read_jsonl(args.test)
            if args.limit is not None: test = test[:args.limit]
            raw_path = args.raw_out or os.path.join(C.OUTPUTS_DIR, f"raw_{args.name}.jsonl")
            phase1_generate(name=args.name, test=test, raw_out_path=raw_path,
                            batch_size=args.batch_size, lora_path=args.lora)
        else:
            run_baseline(name=args.name, test_file=args.test, out_path=args.out,
                         raw_out_path=args.raw_out, limit=args.limit,
                         batch_size=args.batch_size, lora_path=args.lora,
                         num_workers=args.num_workers)
    elif args.phase == "score":
        raw_path = args.raw_out or os.path.join(C.OUTPUTS_DIR, f"raw_{args.name}.jsonl")
        out_path = args.out or os.path.join(C.OUTPUTS_DIR, f"baseline_{args.name}.jsonl")
        phase2_score(raw_path=raw_path, out_path=out_path, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
