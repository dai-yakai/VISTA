"""Bridge to NetArena's official MALT scoring code.

We run *exactly* the same evaluator NetArena uses (`malt_env.BenchmarkEvaluator`),
just bypassing the A2A HTTP layer so we can drive it from a single process and
score thousands of examples efficiently.

Key invariants we preserve (= bit-for-bit alignment with NetArena's published numbers):
- The MALT real graph G is built by `getGraphData()` (same prototxt parser).
- LLM-generated code is executed by `BenchmarkEvaluator.run_agent_output(...)`.
- Correctness/Safety/Latency are produced by `BenchmarkEvaluator.ground_truth_check(...)`.
- The SafetyChecker (6 graph invariants) and solid_step* helpers are imported from NetArena.
"""
from __future__ import annotations

import os, sys, time, json, copy, traceback, contextlib, signal
from typing import Any
from . import config as C

# Make NetArena's app-malt importable.
sys.path.insert(0, C.NETARENA_MALT_DIR)
sys.path.insert(0, C.NETARENA_SRC_DIR)

# These imports come from NetArena. We deliberately *do not* shadow them.
import networkx as nx                                        # noqa: E402
from solid_step_helper import getGraphData, validate_llm_output, clean_up_updated_graph_data  # noqa: E402
from solid_step_helper import (                              # noqa: E402
    solid_step_add_node_to_graph,
    solid_step_counting_query,
    solid_step_remove_node_from_graph,
    solid_step_list_child_nodes,
    solid_step_update_node_value,
    solid_step_rank_child_nodes,
)
from error_check import SafetyChecker                        # noqa: E402
from malt_env import BenchmarkEvaluator                      # noqa: E402


# Singleton: load the real MALT graph once.
_GRAPH_TUPLE: tuple[Any, nx.DiGraph] | None = None


def get_malt_graph() -> nx.DiGraph:
    global _GRAPH_TUPLE
    if _GRAPH_TUPLE is None:
        _GRAPH_TUPLE = getGraphData()
    return _GRAPH_TUPLE[1]


def make_evaluator() -> BenchmarkEvaluator:
    return BenchmarkEvaluator(graph_data=get_malt_graph())


def score_one(query: str, golden_answer: str, llm_code: str,
              task_label: str, run_latency: float | None = None,
              evaluator: BenchmarkEvaluator | None = None) -> dict:
    """Score a single (query, golden, llm_code) triplet using NetArena's evaluator.

    Returns a dict with at least:
      - "Result-Correctness": "Pass" | "Fail"
      - "Result-Safety":      "Pass" | "Fail"
      - "Result-Latency":     float seconds
      - "Label":              task_label
      - "Query":              query
      - "Error" (optional)
    """
    ev = evaluator or make_evaluator()
    t0 = time.perf_counter()
    # NetArena's run_agent_output expects (current_query, golden_answer, llm_answer=code)
    # We pass our (already-extracted) `def process_graph(...)` snippet directly as `llm_answer`.
    try:
        ret, gt_ret, ver, ver_err, gt_ver, gt_ver_err, ret_graph_copy = ev.run_agent_output(
            query, golden_answer, llm_answer=llm_code or "")
    except Exception:  # belt-and-suspenders; NetArena's evaluator already wraps exec errors
        return {
            "Query": query,
            "Label": task_label,
            "Result-Correctness": "Fail",
            "Result-Safety": "Fail",
            "Result-Latency": run_latency if run_latency is not None else 0.0,
            "Error": traceback.format_exc(),
            "LLM code": llm_code,
        }
    elapsed = run_latency if run_latency is not None else (time.perf_counter() - t0)
    res = ev.ground_truth_check(query, task_label, ret, gt_ret, ret_graph_copy,
                                ver, ver_err, gt_ver, gt_ver_err, elapsed)
    return res


# Tiny smoke test — run with `python -m src.bridge`.
def _smoke():
    # First training example we can introspect.
    from .data import read_jsonl
    train = read_jsonl(C.TRAIN_FILE)[:1]
    ex = train[0]
    # Run the *ground-truth* code as if it were our LLM output: should always be a Pass.
    gt_code = ex["answer"].replace("ground_truth_process_graph", "process_graph")
    res = score_one(ex["question"], ex["answer"], gt_code, ex["task_label"])
    print(json.dumps({k: v for k, v in res.items() if k not in ("Ground truth code","LLM code","Ground truth exec","LLM code exec","Error")}, indent=2))
    assert res["Result-Correctness"] == "Pass", "smoke test FAILED — bridge is broken"
    print("OK: bridge smoke test passed.")


if __name__ == "__main__":
    _smoke()
