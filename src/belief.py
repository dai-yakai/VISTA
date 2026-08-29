"""§4 Time-typed Belief Graph B_t over the MALT topology.

In MALT the "topology" is already a `nx.DiGraph`. The "belief" object adds:
  - evidence_set: nodes/edges we've explicitly inspected (their attrs at the time)
  - assumptions: what we are presuming (e.g. "parent ju1.s1.dom exists")
  - touched_nodes: nodes we've modified (or plan to modify)
  - confidence: per-evidence confidence in [0,1]

Concretely, the belief is consumed by the LLM in two ways:
  - As a short factual snippet inside the prompt (children counts, parent kind, etc.)
  - As an oracle for evidence-grounded retries (we tell the LLM if its target exists)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import copy as _copy
import networkx as nx


HIERARCHY = {
    "EK_JUPITER": ["EK_SPINEBLOCK", "EK_SUPERBLOCK"],
    "EK_SPINEBLOCK": ["EK_PACKET_SWITCH"],
    "EK_SUPERBLOCK": ["EK_AGG_BLOCK"],
    "EK_AGG_BLOCK": ["EK_PACKET_SWITCH"],
    "EK_CHASSIS": ["EK_CONTROL_POINT", "EK_PACKET_SWITCH"],
    "EK_CONTROL_POINT": ["EK_PACKET_SWITCH"],
    "EK_RACK": ["EK_CHASSIS"],
    "EK_PACKET_SWITCH": ["EK_PORT"],
    "EK_CONTROL_DOMAIN": ["EK_CONTROL_POINT", "EK_PACKET_SWITCH"],
}


@dataclass
class BeliefGraph:
    """Belief state at time t built on top of the MALT real graph."""
    G: nx.DiGraph                    # full MALT graph (read-only reference)
    evidence: dict[str, dict] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    touched: set[str] = field(default_factory=set)

    def has_node(self, name: str) -> bool:
        for n, d in self.G.nodes(data=True):
            if d.get("name") == name or n == name:
                return True
        return False

    def find_node(self, name: str) -> tuple[str, dict] | None:
        for n, d in self.G.nodes(data=True):
            if d.get("name") == name or n == name:
                return n, d
        return None

    def kind_of(self, name: str) -> str | None:
        f = self.find_node(name)
        if not f:
            return None
        types = f[1].get("type") or []
        return types[0] if isinstance(types, list) and types else (
            types if isinstance(types, str) else None
        )

    def direct_children(self, name: str) -> list[tuple[str, dict]]:
        f = self.find_node(name)
        if not f:
            return []
        node = f[0]
        out = []
        for u, v, d in self.G.out_edges(node, data=True):
            t = d.get("type") or []
            if (isinstance(t, list) and "RK_CONTAINS" in t) or t == "RK_CONTAINS":
                out.append((v, dict(self.G.nodes[v])))
        return out

    def direct_parents(self, name: str) -> list[tuple[str, dict]]:
        f = self.find_node(name)
        if not f:
            return []
        node = f[0]
        out = []
        for u, _, d in self.G.in_edges(node, data=True):
            t = d.get("type") or []
            if (isinstance(t, list) and "RK_CONTAINS" in t) or t == "RK_CONTAINS":
                out.append((u, dict(self.G.nodes[u])))
        return out

    def contains_descendants(self, name: str) -> list[tuple[str, dict]]:
        f = self.find_node(name)
        if not f:
            return []
        out: list[tuple[str, dict]] = []
        stack = [f[0]]
        seen = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            for _, child, d in self.G.out_edges(node, data=True):
                t = d.get("type") or []
                if (isinstance(t, list) and "RK_CONTAINS" in t) or t == "RK_CONTAINS":
                    out.append((child, dict(self.G.nodes[child])))
                    stack.append(child)
        return out

    def add_evidence(self, key: str, payload: dict, conf: float = 1.0):
        self.evidence[key] = {"payload": payload, "confidence": conf}

    def assume(self, statement: str):
        self.assumptions.append(statement)

    # ------------------------------------------------------------------
    def summarize_for_prompt(self, goal: dict) -> str:
        """Render a compact factual snippet to inject into the LLM prompt.

        For each operation in the parsed goal, we add concrete facts about the
        named target nodes — kind, existence, direct child count, sample child
        names. Keeps the LLM grounded even before any tool call.
        """
        lines = ["# Belief snapshot (queried evidence — fresh from the graph):"]
        for st in goal.get("operations", []):
            tgt = st.get("target")
            params = st.get("params") or {}
            scope = params.get("scope") or params.get("parent")

            for who in (tgt, scope):
                if not who or who in self.evidence:
                    continue
                f = self.find_node(who)
                if not f:
                    lines.append(f"#   - node `{who}`: NOT FOUND in graph")
                    self.add_evidence(who, {"exists": False}, conf=1.0)
                else:
                    n, d = f
                    types = d.get("type")
                    children = self.direct_children(who)
                    parents = self.direct_parents(who)
                    descendants = self.contains_descendants(who)
                    sample = ", ".join(c[1].get("name", str(c[0])) for c in children[:5])
                    parent_sample = ", ".join(p[1].get("name", str(p[0])) for p in parents[:3])
                    line = (f"#   - node `{who}`: exists, type={types}, "
                            f"#children={len(children)}"
                            + (f", sample_children=[{sample}{'...' if len(children)>5 else ''}]"
                               if children else "")
                            + (f", parents=[{parent_sample}]" if parents else "")
                            + (f", contains_descendants={len(descendants)}" if descendants else ""))
                    lines.append(line)
                    payload = {"exists": True, "types": types,
                               "n_children": len(children),
                               "sample_children": [c[1].get("name", c[0]) for c in children[:5]],
                               "n_descendants": len(descendants),
                               "parents": [p[1].get("name", p[0]) for p in parents[:3]]}
                    self.add_evidence(who, payload, conf=1.0)

            if st.get("op") == "add":
                target_type = (params.get("type") or "").strip()
                parent = (params.get("parent") or "").strip()
                if target_type and parent:
                    parent_type = self.kind_of(parent)
                    allowed = HIERARCHY.get(parent_type or "", [])
                    verdict = "allowed" if target_type in allowed else "not in direct hierarchy"
                    lines.append(
                        f"#   - add-constraint: parent `{parent}` type={parent_type}, "
                        f"new_type={target_type}, allowed_children={allowed}, verdict={verdict}"
                    )

            if st.get("op") == "remove" and tgt:
                descendants = self.contains_descendants(tgt)
                if descendants:
                    sample_desc = ", ".join(d[1].get("name", str(d[0])) for d in descendants[:5])
                    lines.append(
                        f"#   - removal-risk: `{tgt}` contains {len(descendants)} descendant nodes "
                        f"({sample_desc}{'...' if len(descendants) > 5 else ''}); removing only the parent "
                        "will leave isolated descendants. For non-graph query outputs, remove the contained "
                        "subtree before answering to preserve safety."
                    )

            if st.get("op") == "rank" or st.get("op") == "list":
                # also list scope's children types distribution
                if scope:
                    children = self.direct_children(scope)
                    type_count: dict[str, int] = {}
                    for _, cd in children:
                        ct = cd.get("type")
                        ct0 = ct[0] if isinstance(ct, list) and ct else str(ct)
                        type_count[ct0] = type_count.get(ct0, 0) + 1
                    lines.append(f"#   - children-type distribution under `{scope}`: {dict(type_count)}")
        if self.assumptions:
            lines.append("# Assumptions:")
            for a in self.assumptions:
                lines.append(f"#   - {a}")
        return "\n".join(lines)


def build_belief(goal: dict) -> BeliefGraph:
    """Build a fresh BeliefGraph from the cached MALT graph."""
    from .bridge import get_malt_graph
    G = get_malt_graph()
    bg = BeliefGraph(G=G)
    return bg
