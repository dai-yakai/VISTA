"""Case-grounded hypothesis analysis for VISTA planning.

This module keeps the reasoning layer explicit: it combines the compiled goal,
the current belief snapshot, and retrieved historical cases to produce a compact
planning trace. The trace is injected into VISTA prompts and stored in raw
outputs so that every candidate program can be audited against the same
assumptions, risks, and feasible action branches.
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

from .belief import BeliefGraph, HIERARCHY


def goal_retrieval_tag(goal: dict) -> str:
    """Return the procedural tag used by the case-memory index."""
    level = goal.get("level", "unknown")
    ops = [s.get("op", "unknown") for s in goal.get("operations", [])]
    if not ops:
        op_tag = "unknown"
    elif len(ops) == 1:
        op_tag = ops[0]
    else:
        op_tag = f"{ops[0]}-{ops[-1]}"
    return f"{level}|{op_tag}"


def _case_memory_summary(retrieved: Sequence[dict]) -> str:
    if not retrieved:
        return "no retrieved cases"
    labels = Counter(r.get("task_label", "unknown") for r in retrieved)
    ops = Counter(r.get("op", "unknown") for r in retrieved)
    helper_hits = Counter()
    for r in retrieved:
        answer = r.get("answer", "")
        for helper in (
            "solid_step_add_node_to_graph",
            "solid_step_remove_node_from_graph",
            "solid_step_counting_query",
            "solid_step_list_child_nodes",
            "solid_step_rank_child_nodes",
            "solid_step_update_node_value",
        ):
            if helper in answer:
                helper_hits[helper] += 1
    top_labels = ", ".join(f"{k}:{v}" for k, v in labels.most_common(3))
    top_ops = ", ".join(f"{k}:{v}" for k, v in ops.most_common(3))
    helpers = ", ".join(f"{k}:{v}" for k, v in helper_hits.most_common(4))
    return f"labels=[{top_labels}], ops=[{top_ops}], helpers=[{helpers}]"


def _risk_label(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def analyze_hypotheses(goal: dict, belief: BeliefGraph, retrieved: Sequence[dict]) -> tuple[str, dict]:
    """Build an auditable hypothesis/risk trace for the current query.

    The numeric scores are deterministic and deliberately simple:
    - evidence uncertainty is one minus the parser/evidence confidence;
    - risk is dominated by known invariant threats;
    - information gain is high when an operation targets an unresolved node.
    """
    parse_conf = float((goal.get("uncertainty") or {}).get("parse_confidence", 0.0))
    case_memory = _case_memory_summary(retrieved)
    hypotheses: list[dict] = []

    for st in goal.get("operations", []):
        op = st.get("op")
        tgt = st.get("target")
        params = st.get("params") or {}
        risk = 0.05
        info_gain = 0.05
        feasible = True
        rationale = "observed graph evidence supports the literal operation"

        if op == "add":
            parent = params.get("parent")
            new_type = params.get("type")
            parent_type = belief.kind_of(parent) if parent else None
            allowed = HIERARCHY.get(parent_type or "", [])
            if parent and not belief.has_node(parent):
                feasible = False
                risk = 0.9
                info_gain = 0.8
                rationale = f"parent `{parent}` is unresolved"
            elif new_type and parent_type and new_type not in allowed:
                feasible = False
                risk = 1.0
                rationale = (
                    f"direct containment {parent_type}->{new_type} violates the MALT hierarchy; "
                    "a proof-carrying plan must reject or repair this before execution"
                )
            elif not new_type or not parent:
                risk = 0.45
                info_gain = 0.55
                rationale = "add operation is underspecified"
            else:
                rationale = f"direct containment {parent_type}->{new_type} is allowed"

        elif op == "remove" and tgt:
            descendants = belief.contains_descendants(tgt)
            if not belief.has_node(tgt):
                feasible = False
                risk = 0.85
                info_gain = 0.8
                rationale = f"target `{tgt}` is unresolved"
            elif descendants:
                if goal.get("return_shape") == "graph":
                    feasible = False
                    risk = 1.0
                    rationale = (
                        f"literal graph removal leaves {len(descendants)} contained descendants isolated; "
                        "safe subtree removal changes the requested graph"
                    )
                else:
                    risk = 0.25
                    rationale = (
                        f"target contains {len(descendants)} descendants; use safe subtree removal "
                        "before the downstream query"
                    )
            else:
                rationale = "target is a leaf node, so direct removal has low structural risk"

        elif op in {"count", "list", "rank"}:
            scope = params.get("scope")
            if scope and not belief.has_node(scope):
                feasible = False
                risk = 0.75
                info_gain = 0.8
                rationale = f"query scope `{scope}` is unresolved"
            else:
                rationale = "query is read-only; risk comes from whether prior mutation is safe"

        hypotheses.append({
            "op": op,
            "target": tgt,
            "feasible": feasible,
            "risk": risk,
            "risk_label": _risk_label(risk),
            "information_gain": info_gain,
            "uncertainty": max(0.0, 1.0 - parse_conf),
            "rationale": rationale,
        })

    feasible_all = all(h["feasible"] for h in hypotheses)
    max_risk = max((h["risk"] for h in hypotheses), default=0.0)
    trace = {
        "retrieval_tag": goal_retrieval_tag(goal),
        "case_memory": case_memory,
        "feasible": feasible_all,
        "max_risk": max_risk,
        "hypotheses": hypotheses,
    }

    lines = ["# Hypothesis and planning trace:"]
    lines.append(f"#   - retrieval_tag={trace['retrieval_tag']}; case_memory={case_memory}")
    lines.append(
        f"#   - global_feasible={feasible_all}; max_risk={max_risk:.2f} "
        f"({_risk_label(max_risk)}); parse_confidence={parse_conf:.2f}"
    )
    for i, h in enumerate(hypotheses, 1):
        lines.append(
            f"#   - h{i}: op={h['op']}, target={h.get('target')}, "
            f"feasible={h['feasible']}, risk={h['risk']:.2f}, "
            f"info_gain={h['information_gain']:.2f}, reason={h['rationale']}"
        )
    if not feasible_all:
        lines.append(
            "#   - planning_rule: do not execute a literal action that violates an invariant; "
            "repair the action or let verification fail."
        )
    return "\n".join(lines), trace
