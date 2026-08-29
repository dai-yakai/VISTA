"""Parallel scoring driver — runs NetArena's BenchmarkEvaluator across many
records concurrently. The bottleneck is networkx's `is_isomorphic` on the
~5500-node MALT graph (≈ 30 s when graphs ARE isomorphic). Parallelising over
CPU cores brings 1000 examples from ~9 h down to ~30–60 min on a 16-core box.

Two entry points:
  - `score_records_parallel(records, num_workers)`
        records: list of dict with keys {Query, Label, golden, code, id}
        returns: list of dict scoring results (NetArena format) in same order
  - `score_jsonl_parallel(in_path, out_path, num_workers)`
"""
from __future__ import annotations

import contextlib, io, json, os, sys, time
from typing import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

# Each worker imports NetArena once at startup.
_EVALUATOR = None


def _worker_init():
    global _EVALUATOR
    # Make NetArena imports importable in this worker.
    sys.path.insert(0, "/home/ubuntu/NetArena/app-malt")
    sys.path.insert(0, "/home/ubuntu/NetArena/src")
    from malt_env import BenchmarkEvaluator
    from solid_step_helper import getGraphData
    _, G = getGraphData()
    _EVALUATOR = BenchmarkEvaluator(graph_data=G)


def _score_one_in_worker(args: dict) -> dict:
    global _EVALUATOR
    if _EVALUATOR is None:
        _worker_init()
    ev = _EVALUATOR
    query = args["query"]; golden = args["golden"]; code = args["code"]; label = args["label"]
    latency = args.get("latency", 0.0)
    import time as _t, traceback
    t0 = _t.perf_counter()
    try:
        # NetArena's evaluator prints every query and large graph/list payloads.
        # Suppress that chatter here; structured fields below still preserve the
        # actual correctness and safety errors for experiment logs.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ret, gt_ret, ver, ver_err, gt_ver, gt_ver_err, ret_graph_copy = ev.run_agent_output(
                query, golden, llm_answer=code or "")
        elapsed = latency if latency else (_t.perf_counter() - t0)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            res = ev.ground_truth_check(query, label, ret, gt_ret, ret_graph_copy,
                                        ver, ver_err, gt_ver, gt_ver_err, elapsed)
    except Exception:
        res = {
            "Query": query, "Label": label,
            "Result-Correctness": "Fail", "Result-Safety": "Fail",
            "Result-Latency": latency, "Error": traceback.format_exc(),
        }
    res["__id"] = args.get("id")
    res["__extracted_code"] = code
    res["__raw_response"] = args.get("raw_response", "")
    return res


def score_records_parallel(records: Sequence[dict], num_workers: int = 8,
                            timeout_per_task: float = 120.0) -> list[dict]:
    """records: list of {query, golden, code, label, latency, id, raw_response}."""
    ctx = get_context("spawn")
    results: dict[int, dict] = {}
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx,
                              initializer=_worker_init) as ex:
        futs = {ex.submit(_score_one_in_worker, dict(r)): i for i, r in enumerate(records)}
        for fut in as_completed(futs):
            idx = futs[fut]
            try:
                results[idx] = fut.result(timeout=timeout_per_task)
            except Exception as e:
                results[idx] = {
                    "Query": records[idx]["query"], "Label": records[idx]["label"],
                    "Result-Correctness": "Fail", "Result-Safety": "Fail",
                    "Result-Latency": records[idx].get("latency", 0.0),
                    "Error": f"worker error: {e!r}",
                    "__id": records[idx].get("id"),
                    "__extracted_code": records[idx].get("code", ""),
                }
    return [results[i] for i in range(len(records))]


def main():
    # CLI:   python -m src.parallel_score in.jsonl out.jsonl [num_workers]
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    num_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    records = []
    with open(in_path) as f:
        for line in f:
            o = json.loads(line)
            records.append({
                "query": o["query"], "golden": o["golden"], "code": o["code"],
                "label": o["label"], "latency": o.get("latency", 0.0),
                "id": o.get("id"), "raw_response": o.get("raw_response", ""),
            })
    print(f"[parallel_score] {len(records)} records → {out_path} with {num_workers} workers")
    t0 = time.perf_counter()
    res = score_records_parallel(records, num_workers=num_workers)
    print(f"[parallel_score] done in {time.perf_counter()-t0:.1f}s")

    with open(out_path, "w") as f:
        for r in res:
            f.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
