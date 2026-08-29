"""VISTA-Net++ batched runner — two-phase pattern (GPU-only generation, deferred parallel scoring).

Phase 1 (GPU-bound):
  For each query in chunks:
    - intent → typed Goal G
    - belief snapshot (from real MALT graph)
    - retrieve k cases
    - sample N candidates per query (batched across the chunk)
    - verify each candidate locally (fast verifier, NO isomorphism)
    - if any pass → done
    - else repair up to K rounds (each round one greedy gen per failing q)
  Save raw final code per query → outputs/raw_vista[_sft].jsonl

Phase 2 (CPU-bound, parallel):
  Score all raw codes via NetArena's BenchmarkEvaluator → outputs/vista[_sft].jsonl
"""
from __future__ import annotations

import argparse, glob, json, os, time
from typing import Sequence

from loguru import logger

from . import config as C
from . import prompts as P
from .data import read_jsonl
from .bridge import get_malt_graph
from .intent import compile_intent
from .belief import build_belief
from .verifier import verify
from .planner import _build_messages, _build_repair_messages
from .polish import polish_code
from .reasoning import analyze_hypotheses, goal_retrieval_tag


def _process_one_initial(question: str, retriever, train_pool, malt_graph, exclude_ids: set[int] | None = None):
    goal = compile_intent(question)
    belief = build_belief(goal)
    belief_summary = belief.summarize_for_prompt(goal)
    if retriever is not None:
        idxs = retriever.topk(
            question,
            k=C.VISTA_RETRIEVAL_K,
            tag=goal_retrieval_tag(goal),
            exclude_ids=exclude_ids,
        )
        retrieved = [train_pool[i] for i in idxs]
    else:
        retrieved = []
    planning_trace, hypothesis_trace = analyze_hypotheses(goal, belief, retrieved)
    msgs = _build_messages(goal, belief_summary, retrieved, planning_trace)
    return goal, belief_summary, planning_trace, hypothesis_trace, retrieved, msgs


def phase1_vista_generate(
    *,
    test_file: str,
    raw_out_path: str,
    limit: int | None,
    chunk_size: int,
    n_hyp: int,
    max_repair: int,
    lora_path: str | None,
):
    from .llm_engine import LLMEngine, make_lora_request
    from .retrieval import CaseRetriever

    if lora_path is not None and not os.path.isfile(os.path.join(lora_path, "adapter_config.json")):
        parent = os.path.dirname(lora_path.rstrip(os.sep))
        candidates = [
            p for p in glob.glob(os.path.join(parent, "checkpoint-*"))
            if os.path.isfile(os.path.join(p, "adapter_config.json"))
        ]
        if candidates:
            lora_path = sorted(candidates, key=os.path.getmtime, reverse=True)[0]
            logger.warning(f"[phase1] requested LoRA adapter missing; using latest checkpoint {lora_path}")
        else:
            raise FileNotFoundError(
                f"LoRA adapter not found at {lora_path}; expected adapter_config.json. "
                "Run `python -m src.sft_train` first or pass a valid checkpoint directory."
            )

    test = read_jsonl(test_file)
    if limit is not None:
        test = test[:limit]
    train_pool = read_jsonl(C.TRAIN_FILE)
    retriever = CaseRetriever.build_or_load(train_pool)

    enable_lora = lora_path is not None
    engine = LLMEngine(enable_lora=enable_lora)
    lora_req = make_lora_request(lora_path) if enable_lora else None
    malt_graph = get_malt_graph()
    method_name = "vista_sft" if lora_path else "vista_base"

    logger.info(f"[phase1:{method_name}] running on {len(test)} examples → {raw_out_path}; "
                f"n_hyp={n_hyp}, max_repair={max_repair}, lora={lora_path}")

    n_initial_pass_total = 0
    n_repair_pass_total = 0
    n_repair_iters_total = 0

    with open(raw_out_path, "w") as fout:
        for cs in range(0, len(test), chunk_size):
            chunk = test[cs:cs+chunk_size]
            t_chunk = time.perf_counter()

            goals, beliefs, planning_traces, hypothesis_traces, retrieves, msg_batch = [], [], [], [], [], []
            for ex in chunk:
                g, b, pt, ht, r, m = _process_one_initial(
                    ex["question"],
                    retriever,
                    train_pool,
                    malt_graph,
                    exclude_ids={ex["id"]},
                )
                goals.append(g); beliefs.append(b); planning_traces.append(pt)
                hypothesis_traces.append(ht); retrieves.append(r); msg_batch.append(m)

            outs = engine.generate(
                msg_batch,
                temperature=C.GEN_TEMPERATURE_SAMPLE,
                top_p=C.GEN_TOP_P,
                max_tokens=C.GEN_MAX_TOKENS,
                n=n_hyp,
                lora_request=lora_req,
            )

            best_per_q: list[dict] = []
            n_initial_pass = 0
            for ex, goal, samples in zip(chunk, goals, outs):
                cands = []
                for s in samples:
                    code = polish_code(P.extract_code_output(s["text"]), goal)
                    pf = verify(code, goal, malt_graph)
                    cands.append({"text": s["text"], "code": code, "proof": pf})
                cands.sort(key=lambda x:(x["proof"].passed, x["proof"].confidence), reverse=True)
                best_per_q.append(cands[0])
                if cands[0]["proof"].passed: n_initial_pass += 1
            n_initial_pass_total += n_initial_pass

            need_repair_idx = [i for i, b in enumerate(best_per_q) if not b["proof"].passed]
            n_repair = 0
            for r_iter in range(max_repair):
                if not need_repair_idx: break
                rep_msgs = []
                for i in need_repair_idx:
                    rep_msgs.append(_build_repair_messages(goals[i], beliefs[i], retrieves[i],
                                                            best_per_q[i]["code"], best_per_q[i]["proof"],
                                                            planning_traces[i]))
                rep_outs = engine.generate(
                    rep_msgs,
                    temperature=C.GEN_TEMPERATURE_GREEDY,
                    top_p=C.GEN_TOP_P,
                    max_tokens=C.GEN_MAX_TOKENS,
                    n=1, lora_request=lora_req,
                )
                still_fail = []
                for k, i in enumerate(need_repair_idx):
                    rep = rep_outs[k][0]
                    rep_code = polish_code(P.extract_code_output(rep["text"]), goals[i])
                    rep_proof = verify(rep_code, goals[i], malt_graph)
                    n_repair += 1
                    if rep_proof.passed or rep_proof.confidence > best_per_q[i]["proof"].confidence:
                        best_per_q[i] = {"text": rep["text"], "code": rep_code, "proof": rep_proof}
                    if not rep_proof.passed:
                        still_fail.append(i)
                    else:
                        n_repair_pass_total += 1
                need_repair_idx = still_fail
            n_repair_iters_total += n_repair

            chunk_wall = time.perf_counter() - t_chunk
            per_q_lat = chunk_wall / len(chunk)
            for ex, goal, best, b_summary, p_trace, h_trace in zip(
                chunk, goals, best_per_q, beliefs, planning_traces, hypothesis_traces
            ):
                fout.write(json.dumps({
                    "id": ex["id"], "query": ex["question"],
                    "golden": ex["answer"], "code": best["code"],
                    "label": ex["task_label"],
                    "raw_response": best["text"],
                    "latency": per_q_lat,
                    "proof": best["proof"].to_json(),
                    "goal": goal,
                    "belief_summary": b_summary,
                    "planning_trace": p_trace,
                    "hypothesis_trace": h_trace,
                }, default=str, ensure_ascii=False) + "\n")
            logger.info(f"[phase1:{method_name}] chunk {cs//chunk_size+1}/{(len(test)+chunk_size-1)//chunk_size} "
                        f"initial_pass={n_initial_pass}/{len(chunk)} repairs+={n_repair} "
                        f"running pass={n_initial_pass_total}/{cs+len(chunk)}")

    logger.info(f"[phase1:{method_name}] generation complete: initial_pass={n_initial_pass_total}/{len(test)} "
                f"repair_pass={n_repair_pass_total}/{len(test)} total_repair_iters={n_repair_iters_total}")


def phase2_vista_score(*, raw_path: str, out_path: str, num_workers: int = 12, method: str):
    from .parallel_score import score_records_parallel
    logger.info(f"[phase2:{method}] scoring {raw_path} with {num_workers} workers → {out_path}")
    records = []
    with open(raw_path) as f:
        for line in f:
            records.append(json.loads(line))
    t0 = time.perf_counter()
    res = score_records_parallel(records, num_workers=num_workers)
    logger.info(f"[phase2:{method}] scored {len(res)} records in {time.perf_counter()-t0:.1f}s")

    pass_correct = pass_safety = total = 0; sum_lat = 0.0
    with open(out_path, "w") as fout:
        for r, raw in zip(res, records):
            r["baseline"] = method
            r["id"] = raw.get("id")
            r["raw_response"] = raw.get("raw_response","")
            r["extracted_code"] = raw.get("code","")
            r["proof"] = raw.get("proof")
            r["goal"] = raw.get("goal")
            r["belief_summary"] = raw.get("belief_summary","")
            r["planning_trace"] = raw.get("planning_trace","")
            r["hypothesis_trace"] = raw.get("hypothesis_trace")
            fout.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")
            total += 1
            if r.get("Result-Correctness") == "Pass": pass_correct += 1
            if r.get("Result-Safety")      == "Pass": pass_safety += 1
            sum_lat += float(r.get("Result-Latency", 0.0))

    summary = {
        "method": method,
        "n": total,
        "correctness": pass_correct/total if total else 0.0,
        "safety": pass_safety/total if total else 0.0,
        "avg_latency_s": sum_lat/total if total else 0.0,
        "out_path": out_path,
    }
    with open(out_path.replace(".jsonl",".summary.json"),"w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"[phase2:{method}] DONE — {summary}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=C.TEST_FILE)
    ap.add_argument("--out", default=None)
    ap.add_argument("--raw-out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--n-hyp", type=int, default=C.VISTA_NUM_HYPOTHESES)
    ap.add_argument("--max-repair", type=int, default=C.VISTA_MAX_REPAIR_ITER)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--phase", choices=["all","gen","score"], default="all")
    args = ap.parse_args()

    method = "vista_sft" if args.lora else "vista_base"
    raw_path = args.raw_out or os.path.join(C.OUTPUTS_DIR, f"raw_{method}.jsonl")
    out_path = args.out or os.path.join(C.OUTPUTS_DIR, f"{method}.jsonl")

    if args.phase in ("all", "gen"):
        phase1_vista_generate(
            test_file=args.test, raw_out_path=raw_path,
            limit=args.limit, chunk_size=args.chunk_size,
            n_hyp=args.n_hyp, max_repair=args.max_repair, lora_path=args.lora,
        )
    if args.phase in ("all", "score"):
        phase2_vista_score(raw_path=raw_path, out_path=out_path,
                           num_workers=args.num_workers, method=method)


if __name__ == "__main__":
    main()
