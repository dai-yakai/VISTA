"""Unified runner: load vLLM once, run all generation phases sequentially.

This saves 4–6 × ~3 min of vLLM cold-start time across the full pipeline. It
runs *only* phase-1 generation; scoring is done afterwards in a separate
process so vLLM stays in memory the whole time.
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


def _gt_to_process_graph(gt_code: str) -> str:
    return gt_code.replace("ground_truth_process_graph", "process_graph")


def _build_messages_for_baseline(name: str, question: str, retriever=None, train_pool=None):
    if name in ("zeroshot_base", "zeroshot_cot", "fewshot_base", "fewshot_cot"):
        return P.build_messages(question, prompt_type=name)
    elif name == "fewshot_semantic":
        idxs = retriever.topk(question, k=C.VISTA_RETRIEVAL_K)
        examples = [{"question": train_pool[i]["question"],
                     "answer": _gt_to_process_graph(train_pool[i]["answer"])} for i in idxs]
        return P.build_messages(question, prompt_type="fewshot_semantic",
                                fewshot_examples=examples)
    elif name == "sft_only":
        return P.build_messages(question, prompt_type="zeroshot_base")
    else:
        raise ValueError(f"unknown baseline: {name}")


def _gen_baseline(engine, *, name: str, test: list[dict], lora_req,
                   batch_size: int, raw_path: str, retriever, train_pool):
    logger.info(f"[gen:{name}] {len(test)} → {raw_path}")
    with open(raw_path, "w") as f:
        for bs in range(0, len(test), batch_size):
            batch = test[bs:bs+batch_size]
            messages = [_build_messages_for_baseline(name, ex["question"], retriever, train_pool)
                        for ex in batch]
            t0 = time.perf_counter()
            outs = engine.generate(messages, temperature=C.GEN_TEMPERATURE_GREEDY,
                                   top_p=C.GEN_TOP_P, max_tokens=C.GEN_MAX_TOKENS,
                                   n=1, lora_request=lora_req)
            wall = time.perf_counter() - t0
            per_q = wall / max(1, len(batch))
            for ex, samples in zip(batch, outs):
                raw = samples[0]["text"]
                code = P.extract_code_output(raw)
                f.write(json.dumps({
                    "id": ex["id"], "query": ex["question"],
                    "golden": ex["answer"], "code": code,
                    "label": ex["task_label"], "raw_response": raw,
                    "latency": per_q,
                }, ensure_ascii=False) + "\n")
            if (bs // batch_size) % 4 == 0:
                logger.info(f"[gen:{name}] {bs+len(batch)}/{len(test)}")


def _gen_vista(engine, *, method: str, test: list[dict], lora_req,
                chunk_size: int, n_hyp: int, max_repair: int,
                raw_path: str, retriever, train_pool, malt_graph):
    logger.info(f"[gen:{method}] {len(test)} → {raw_path}; n_hyp={n_hyp} max_repair={max_repair}")
    n_initial_pass_total = 0
    n_repair_iters_total = 0
    with open(raw_path, "w") as fout:
        for cs in range(0, len(test), chunk_size):
            chunk = test[cs:cs+chunk_size]
            t_chunk = time.perf_counter()
            goals, beliefs, planning_traces, hypothesis_traces, retrieves, msg_batch = [], [], [], [], [], []
            for ex in chunk:
                g = compile_intent(ex["question"])
                b = build_belief(g)
                bs_ = b.summarize_for_prompt(g)
                idxs = retriever.topk(
                    ex["question"],
                    k=C.VISTA_RETRIEVAL_K,
                    tag=goal_retrieval_tag(g),
                    exclude_ids={ex["id"]},
                )
                rs = [train_pool[i] for i in idxs]
                pt, ht = analyze_hypotheses(g, b, rs)
                m = _build_messages(g, bs_, rs, pt)
                goals.append(g); beliefs.append(bs_); planning_traces.append(pt)
                hypothesis_traces.append(ht); retrieves.append(rs); msg_batch.append(m)

            outs = engine.generate(
                msg_batch, temperature=C.GEN_TEMPERATURE_SAMPLE, top_p=C.GEN_TOP_P,
                max_tokens=C.GEN_MAX_TOKENS, n=n_hyp, lora_request=lora_req,
            )
            best_per_q: list[dict] = []
            n_initial_pass = 0
            for ex, goal, samples in zip(chunk, goals, outs):
                cands = []
                for s in samples:
                    raw_code = P.extract_code_output(s["text"])
                    code = polish_code(raw_code, goal)  # §9 typed-output normalization + typed action repair
                    pf = verify(code, goal, malt_graph)
                    cands.append({"text": s["text"], "code": code, "proof": pf})
                cands.sort(key=lambda x: (x["proof"].passed, x["proof"].confidence), reverse=True)
                best_per_q.append(cands[0])
                if cands[0]["proof"].passed: n_initial_pass += 1
            n_initial_pass_total += n_initial_pass

            need_repair_idx = [i for i, b in enumerate(best_per_q) if not b["proof"].passed]
            n_repair = 0
            for r_iter in range(max_repair):
                if not need_repair_idx: break
                rep_msgs = [_build_repair_messages(goals[i], beliefs[i], retrieves[i],
                                                    best_per_q[i]["code"], best_per_q[i]["proof"],
                                                    planning_traces[i])
                            for i in need_repair_idx]
                rep_outs = engine.generate(
                    rep_msgs, temperature=C.GEN_TEMPERATURE_GREEDY,
                    top_p=C.GEN_TOP_P, max_tokens=C.GEN_MAX_TOKENS,
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
                    if not rep_proof.passed: still_fail.append(i)
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
                    "label": ex["task_label"], "raw_response": best["text"],
                    "latency": per_q_lat, "proof": best["proof"].to_json(),
                    "goal": goal, "belief_summary": b_summary,
                    "planning_trace": p_trace, "hypothesis_trace": h_trace,
                }, default=str, ensure_ascii=False) + "\n")
            logger.info(f"[gen:{method}] chunk {cs//chunk_size+1}/{(len(test)+chunk_size-1)//chunk_size} "
                        f"init_pass={n_initial_pass}/{len(chunk)} +repairs={n_repair} "
                        f"running pass={n_initial_pass_total}/{cs+len(chunk)}")
    logger.info(f"[gen:{method}] DONE  init_pass={n_initial_pass_total}/{len(test)} "
                f"repair_iters={n_repair_iters_total}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--vista-chunk-size", type=int, default=8)
    ap.add_argument("--n-hyp", type=int, default=C.VISTA_NUM_HYPOTHESES)
    ap.add_argument("--max-repair", type=int, default=C.VISTA_MAX_REPAIR_ITER)
    ap.add_argument("--lora", default=None,
                    help="Path to LoRA adapter; if given, runs sft_only/vista_sft, else only base baselines + vista_base.")
    ap.add_argument("--baselines", nargs="*", default=None,
                    help="Subset of baseline names to run; default = all applicable.")
    ap.add_argument("--skip-vista", action="store_true")
    args = ap.parse_args()

    test = read_jsonl(C.TEST_FILE)
    if args.limit is not None: test = test[:args.limit]
    train_pool = read_jsonl(C.TRAIN_FILE)

    from .llm_engine import LLMEngine, make_lora_request
    from .retrieval import CaseRetriever
    retriever = CaseRetriever.build_or_load(train_pool)
    malt_graph = get_malt_graph()

    enable_lora = args.lora is not None
    if enable_lora and not os.path.isfile(os.path.join(args.lora, "adapter_config.json")):
        parent = os.path.dirname(args.lora.rstrip(os.sep))
        candidates = [
            p for p in glob.glob(os.path.join(parent, "checkpoint-*"))
            if os.path.isfile(os.path.join(p, "adapter_config.json"))
        ]
        if candidates:
            args.lora = sorted(candidates, key=os.path.getmtime, reverse=True)[0]
            logger.warning(f"[run_all_gen] requested LoRA adapter missing; using latest checkpoint {args.lora}")
        else:
            raise FileNotFoundError(
                f"LoRA adapter not found at {args.lora}; expected adapter_config.json. "
                "Run `python -m src.sft_train` first or pass a valid checkpoint directory."
            )
    engine = LLMEngine(enable_lora=enable_lora)
    lora_req = make_lora_request(args.lora) if enable_lora else None

    # Decide which baselines/runs to do
    base_pool = ["zeroshot_base","zeroshot_cot","fewshot_base","fewshot_semantic"]
    if args.baselines is not None:
        base_pool = [b for b in base_pool if b in args.baselines]

    # 1) Base prompt baselines (no LoRA)
    for name in base_pool:
        raw_path = os.path.join(C.OUTPUTS_DIR, f"raw_{name}.jsonl")
        _gen_baseline(engine, name=name, test=test, lora_req=None,
                      batch_size=args.batch_size, raw_path=raw_path,
                      retriever=retriever, train_pool=train_pool)

    # 2) sft_only (LoRA + base prompt)
    if enable_lora and (args.baselines is None or "sft_only" in args.baselines):
        raw_path = os.path.join(C.OUTPUTS_DIR, "raw_sft_only.jsonl")
        _gen_baseline(engine, name="sft_only", test=test, lora_req=lora_req,
                      batch_size=args.batch_size, raw_path=raw_path,
                      retriever=retriever, train_pool=train_pool)

    # 3) vista_base
    if not args.skip_vista and (args.baselines is None or "vista_base" in args.baselines):
        raw_path = os.path.join(C.OUTPUTS_DIR, "raw_vista_base.jsonl")
        _gen_vista(engine, method="vista_base", test=test, lora_req=None,
                    chunk_size=args.vista_chunk_size, n_hyp=args.n_hyp,
                    max_repair=args.max_repair, raw_path=raw_path,
                    retriever=retriever, train_pool=train_pool, malt_graph=malt_graph)

    # 4) vista_sft
    if enable_lora and not args.skip_vista and (args.baselines is None or "vista_sft" in args.baselines):
        raw_path = os.path.join(C.OUTPUTS_DIR, "raw_vista_sft.jsonl")
        _gen_vista(engine, method="vista_sft", test=test, lora_req=lora_req,
                    chunk_size=args.vista_chunk_size, n_hyp=args.n_hyp,
                    max_repair=args.max_repair, raw_path=raw_path,
                    retriever=retriever, train_pool=train_pool, malt_graph=malt_graph)


if __name__ == "__main__":
    main()
