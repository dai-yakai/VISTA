"""§3 Intent Compiler — turn a natural-language MALT query into a typed Goal G.

We parse with regex templates that mirror NetArena's `dy_query_generation.py`.
A small LLM-based fallback is provided for queries that don't match.

The Goal G is a Python dict, intentionally keyed exactly like methodology.md §3:
  - task_type: "capacity_planning"
  - level: "level-1" | "level-2" | "level-3"
  - operations: list of typed steps in order
       each step: {op, target, params}
  - return_shape: "graph" | "text" | "list" | "table"
  - invariants: list[str]   (network invariants we must preserve)
  - allowed_actions: list[str]
  - protected_scope: list[str] (nodes that must not be touched)
  - stop_condition: { "predicate": str }
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any

from . import config as C


# -------- regex templates aligned with dy_query_generation --------

# Identifier-friendly token: matches "ju1.a1.m4.s3c6.p1" but stops at trailing punctuation.
ID = r"[A-Za-z0-9_.\-]+"
# Optional single quote around an identifier
QID = rf"'?(?P<name>{ID})'?"
NODE_TYPES = [
    "EK_CONTROL_DOMAIN",
    "EK_CONTROL_POINT",
    "EK_PACKET_SWITCH",
    "EK_SUPERBLOCK",
    "EK_SPINEBLOCK",
    "EK_AGG_BLOCK",
    "EK_JUPITER",
    "EK_CHASSIS",
    "EK_RACK",
    "EK_PORT",
]

# Level-1
ADD_PAT = re.compile(
    rf"^Add new node with name\s+(?P<name>{ID})\s+type\s+(?P<type>{ID}),\s+to\s+(?P<parent>{ID})\.\s+Return a graph\.\s*$"
)
REMOVE_PAT = re.compile(
    rf"^Remove\s+(?P<name>{ID})\s+from the graph\.\s+Return a graph\.\s*$"
)
COUNT_PAT = re.compile(
    rf"^Count the\s+(?P<type>{ID})\s+in the\s+(?P<scope>{ID})\.\s+Return (?:only the count number|the count number as text)\.\s*$"
)
LIST_PAT = re.compile(
    rf"^List(?: all)? the (?:direct )?child nodes of\s+(?P<scope>{ID})\.\s+Return a list of child node(?:s)? name(?:s)?\.\s*$"
)
# RANK accepts "Rank all child nodes of <SCOPE>" or "Rank all child nodes of <TYPE> type <SCOPE>"
RANK_PAT = re.compile(
    rf"^Rank (?:all|direct) child nodes of\s+(?:(?P<ntype>{ID})\s+type\s+)?(?P<scope>{ID})(?:\s+in the updated graph)?\s+based on physical_capacity_bps attribute\.\s+Return a list of tuple, each tuple has (?:child )?node name and its total physical capacity\.\s*$"
)
UPDATE_PAT = re.compile(
    rf"^Update the physical capacity value of\s+(?P<name>{ID})\s+to\s+(?P<value>\d+)\.\s+Return a graph\.\s*$"
)

# Level-2/3 prefixes (compound)
ADD_THEN_PREFIX_PAT = re.compile(
    rf"^Add(?: new node with name| node with name)?\s+'?(?P<name>{ID})'?"
    rf"(?:\s+type\s+(?P<type>{ID}))?(?:,)?(?:\s+to\s+(?P<parent>{ID}))?\.\s+(?P<rest>.+)$"
)
REMOVE_THEN_PREFIX_PAT = re.compile(
    rf"^Remove\s+(?P<name>{ID})\s+from the graph\.\s+(?P<rest>.+)$"
)
COUNT_AFTER_PAT = re.compile(
    rf"^Count the\s+(?P<type>{ID})\s+in\s+(?P<scope>{ID})\s+in the updated graph\.\s+Return (?:the count number as text|only the counts)\.\s*$"
)
LIST_AFTER_PAT = re.compile(
    rf"^List direct child nodes of\s+(?P<scope>{ID})\s+in the updated graph\.\s+Return a list of child nodes? name\.\s*$"
)
RANK_AFTER_PAT = re.compile(
    rf"^Rank direct child nodes of\s+(?P<scope>{ID})\s+in the updated graph based on physical_capacity_bps attribute\.\s+Return a list of tuple, each tuple has (?:child )?node name and its total physical capacity\.\s*$"
)


def infer_node_type(name: str | None) -> str | None:
    """Infer a MALT node type from generated benchmark names like new_EK_PORT_46."""
    if not name:
        return None
    for t in sorted(NODE_TYPES, key=len, reverse=True):
        if t in name:
            return t
    return None


def _parse_step(text: str) -> dict | None:
    text = text.strip()
    if (m := ADD_PAT.match(text)):
        return {"op": "add",   "target": m.group("name"),
                "params": {"type": m.group("type"), "parent": m.group("parent").strip()}}
    if (m := REMOVE_PAT.match(text)):
        return {"op": "remove", "target": m.group("name"), "params": {}}
    if (m := COUNT_PAT.match(text)):
        return {"op": "count",  "target": None,
                "params": {"node_type": m.group("type"), "scope": m.group("scope")}}
    if (m := LIST_PAT.match(text)):
        return {"op": "list",   "target": None,
                "params": {"scope": m.group("scope")}}
    if (m := RANK_PAT.match(text)):
        return {"op": "rank",   "target": None,
                "params": {"node_type": m.group("ntype"), "scope": m.group("scope"),
                           "key": "physical_capacity_bps"}}
    if (m := UPDATE_PAT.match(text)):
        return {"op": "update", "target": m.group("name"),
                "params": {"attr": "physical_capacity_bps", "value": int(m.group("value"))}}
    if (m := COUNT_AFTER_PAT.match(text)):
        return {"op": "count", "target": None,
                "params": {"node_type": m.group("type"), "scope": m.group("scope")}}
    if (m := LIST_AFTER_PAT.match(text)):
        return {"op": "list", "target": None, "params": {"scope": m.group("scope")}}
    if (m := RANK_AFTER_PAT.match(text)):
        return {"op": "rank", "target": None,
                "params": {"scope": m.group("scope"), "key": "physical_capacity_bps"}}
    return None


def _split_compound(query: str) -> list[str]:
    """Split a compound query at sentence boundaries — preserves order."""
    raw_parts = [p.strip() for p in re.split(r"(?<=\.)\s+(?=[A-Z])", query.strip()) if p.strip()]
    parts: list[str] = []
    for part in raw_parts:
        # "Return ..." is the tail of the preceding operation template, not a
        # standalone step. Keeping it attached makes the regex templates match
        # the exact NetArena query strings.
        if part.startswith("Return ") and parts:
            parts[-1] = parts[-1] + " " + part
        else:
            parts.append(part)
    return parts


def compile_intent(query: str) -> dict:
    """Compile the natural-language query into a typed Goal."""
    query_s = query.strip()
    steps: list[dict] = []

    # 1) Try parsing the whole query as a single L1 step first (covers list/count/rank/update/add/remove L1).
    single = _parse_step(query_s)
    if single is not None:
        steps.append(single)
    else:
        # 2) Try compound L2/L3: head op, then one or more "after-update" sub-queries.
        if (m := ADD_THEN_PREFIX_PAT.match(query_s)):
            head_target = m.group("name")
            head_parent = (m.group("parent") or "").strip()
            head_type = (m.group("type") or infer_node_type(head_target) or "").strip()
            params = {}
            if head_type:
                params["type"] = head_type
            if head_parent:
                params["parent"] = head_parent
            steps.append({"op": "add", "target": head_target, "params": params})
            rest = m.group("rest")
            for sub in _split_compound(rest):
                s2 = _parse_step(sub)
                if s2 is not None:
                    steps.append(s2)
        elif (m := REMOVE_THEN_PREFIX_PAT.match(query_s)):
            steps.append({"op": "remove", "target": m.group("name"), "params": {}})
            for sub in _split_compound(m.group("rest")):
                s2 = _parse_step(sub)
                if s2 is not None:
                    steps.append(s2)
        else:
            # 3) Fallback: split and try each part.
            for p in _split_compound(query_s):
                step = _parse_step(p)
                if step is not None:
                    steps.append(step)

    # Coarse return shape
    return_shape = "graph" if "Return a graph" in query else (
        "list" if " list" in query.lower() or "List " in query else
        "text" if "count" in query.lower() else "graph"
    )

    # Determine level
    ops = [s.get("op") for s in steps]
    if len(steps) >= 3 or "For each" in query:
        level = "level-3"
    elif len(steps) == 2 and ops[0] == "add" and return_shape != "graph":
        # NetArena labels add+query tasks as level-3 because the add action
        # creates a changed graph and the downstream query reasons on it.
        level = "level-3"
    elif len(steps) == 2:
        level = "level-2"
    else:
        level = "level-1"

    ambiguities: list[str] = []
    if not steps:
        ambiguities.append("no_operation_parsed")
    for st in steps:
        if st.get("op") == "add":
            params = st.get("params") or {}
            if not params.get("type"):
                ambiguities.append(f"add_type_unknown:{st.get('target')}")
            if not params.get("parent"):
                ambiguities.append(f"add_parent_unknown:{st.get('target')}")
        if st.get("op") in {"count", "list", "rank"}:
            params = st.get("params") or {}
            if not params.get("scope"):
                ambiguities.append(f"{st.get('op')}_scope_unknown")
    parse_confidence = 1.0 if steps and not ambiguities else (0.65 if steps else 0.0)

    # Universal MALT invariants (taken from SafetyChecker in error_check.py).
    invariants = [
        "valid_node_types",        # nodes have type in EK_* whitelist
        "valid_edge_types",        # edges have type in {RK_CONTAINS, RK_CONTROLS}
        "valid_node_hierarchy",    # parent-child kinds match allowed pairs
        "no_isolated_nodes",       # every node connected
        "bandwidth_consistent",    # parent capacity = sum of children
        "ports_exist_under_switch" # PACKET_SWITCH must own ≥1 PORT
    ]
    allowed_actions = sorted({s["op"] for s in steps}) or ["query"]
    protected_scope: list[str] = []  # MALT doesn't have explicit protected scope

    goal = {
        "task_type": "capacity_planning",
        "level": level,
        "operations": steps,
        "return_shape": return_shape,
        "invariants": invariants,
        "allowed_actions": allowed_actions,
        "protected_scope": protected_scope,
        "risk_budget": {
            "max_invariant_violations": 0,
            "policy": "reject_or_repair_before_execution",
        },
        "uncertainty": {
            "parse_confidence": parse_confidence,
            "ambiguities": ambiguities,
        },
        "stop_condition": {
            "predicate": "verified_pass and matches_return_shape",
        },
        "raw_query": query,
        "parsed_steps": len(steps),
    }
    return goal


# Used by the agent to render the parsed Goal into a human/LLM-readable string
def goal_to_hint(goal: dict) -> str:
    lines = [f"# Compiled goal — task={goal['task_type']}, level={goal['level']}, return_shape={goal['return_shape']}"]
    for i, st in enumerate(goal["operations"]):
        params = ", ".join(f"{k}={v}" for k, v in (st.get("params") or {}).items())
        tgt = f" target={st.get('target')}" if st.get("target") else ""
        lines.append(f"#   step {i+1}: op={st['op']}{tgt}{(' — ' + params) if params else ''}")
    lines.append(f"# Invariants to preserve: {', '.join(goal['invariants'])}")
    return "\n".join(lines)


def _smoke():
    cases = [
        "Add new node with name new_EK_PORT_82 type EK_PORT, to ju1.a2.m4.s3c6. Return a graph.",
        "Remove ju1.a1.m4.s3c6.p1 from the graph. Return a graph.",
        "Count the EK_PACKET_SWITCH in the ju1.a2.dom. Return only the count number.",
        "List all the child nodes of ju1.s3.dom. Return a list of child node names.",
        "Update the physical capacity value of ju1.a3.m2.s2c4.p10 to 72. Return a graph.",
        "Add new_EK_PORT_3 to ju1.a1.m1.s2c1. Count the EK_PORT in ju1.a1 in the updated graph. Return the count number as text.",
        "Remove ju1.a2.m1.s3c1.p4 from the graph. List direct child nodes of ju1.a2 in the updated graph. Return a list of child nodes name.",
    ]
    for q in cases:
        g = compile_intent(q)
        print("Q:", q)
        print(goal_to_hint(g))
        print()


if __name__ == "__main__":
    _smoke()
