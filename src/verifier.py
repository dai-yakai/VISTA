"""§10 Proof-carrying Verifier.

Turns a candidate `def process_graph(...)` snippet into a `Proof` object that
records:
  - SyntaxOK: parses as Python via ast.parse
  - TypeOK:   defines `process_graph(graph_data)` with one positional arg
  - LocalInvariantOK / GlobalInvariantOK: SafetyChecker (NetArena's 6 checks)
                                          on the resulting graph
  - SemanticEqOK: matches the typed Goal (op type matches what the code does)
  - SimSafeOK:    counterfactual execution against the live MALT graph;
                   also re-checks invariants on the simulated output graph

The final boolean `proof.passed` is the conjunction of all individual checks.
We also produce a per-check confidence score in [0,1] for the calibration loss.
"""
from __future__ import annotations

import ast, copy, json, time, traceback
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any
import networkx as nx

from . import config as C


@dataclass
class Proof:
    syntax_ok: bool = False
    type_ok: bool = False
    local_invariant_ok: bool = False
    global_invariant_ok: bool = False  # alias for safety on returned graph
    semantic_eq_ok: bool = False
    sim_safe_ok: bool = False
    sim_executed: bool = False

    # diagnostics
    syntax_error: str = ""
    type_error: str = ""
    invariant_error: str = ""
    semantic_error: str = ""
    sim_error: str = ""

    # outputs from simulation (copy of `ret` and updated graph)
    sim_ret_type: str = ""
    sim_ret_data_kind: str = ""
    sim_ret_data_summary: str = ""
    sim_updated_graph_present: bool = False

    # final
    passed: bool = False
    confidence: float = 0.0
    repair_hint: str = ""

    # provenance
    elapsed_s: float = 0.0

    def to_json(self) -> dict:
        return asdict(self)


# ---------- helpers ----------

def _check_syntax(code: str) -> tuple[bool, str]:
    if not code.strip():
        return False, "empty code"
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {e.lineno}"


def _check_signature(code: str) -> tuple[bool, str]:
    """Confirm code defines `def process_graph(graph_data, ...)`."""
    try:
        tree = ast.parse(code)
    except Exception as e:
        return False, f"unparseable: {e}"
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "process_graph":
            found = True
            args = node.args.args
            if not args:
                return False, "process_graph defined without positional argument"
            return True, ""
    return False, "no `def process_graph(...)` definition found"


def _summary_of_data(data: Any, max_len: int = 120) -> str:
    try:
        if isinstance(data, (dict, list)):
            s = json.dumps(data, default=str)[:max_len]
        else:
            s = str(data)[:max_len]
        return s
    except Exception:
        return f"<unrepresentable: {type(data).__name__}>"


def _safety_check(graph: nx.Graph) -> tuple[bool, str]:
    """Reuse NetArena's SafetyChecker (6 invariants on the graph)."""
    from error_check import SafetyChecker
    checker = SafetyChecker(ret_graph=graph, ret_list=None)
    try:
        ok, err = checker.evaluate_all()
        return bool(ok), str(err) if not ok else ""
    except Exception:
        return False, traceback.format_exc(limit=2)


def _graph_from_payload(payload: Any) -> tuple[nx.Graph | None, str]:
    if isinstance(payload, nx.Graph):
        return payload, ""
    if isinstance(payload, dict):
        try:
            return nx.readwrite.json_graph.node_link_graph(payload), ""
        except Exception:
            return None, f"graph payload not deserialisable: {type(payload).__name__}"
    return None, f"graph payload has unexpected type {type(payload).__name__}"


def _graph_snapshots_equal(left: nx.Graph, right: nx.Graph) -> bool:
    """Fast exact comparison for two snapshots emitted by the same program.

    The local verifier must stay cheap because it runs over many candidates.
    Full graph isomorphism on MALT's ~5500-node graph is reserved for the
    official scorer; here we only require that `data` and `updated_graph` are
    the same serialized state with the same node identifiers and attributes.
    """
    if left.is_directed() != right.is_directed():
        return False
    if left.number_of_nodes() != right.number_of_nodes() or left.number_of_edges() != right.number_of_edges():
        return False
    if set(left.nodes()) != set(right.nodes()):
        return False
    for n, attrs in left.nodes(data=True):
        if dict(attrs) != dict(right.nodes[n]):
            return False
    for u, v, attrs in left.edges(data=True):
        if not right.has_edge(u, v):
            return False
        if dict(attrs) != dict(right.edges[u, v]):
            return False
    return True


def _node_name_map(G: nx.Graph) -> dict[str, Any]:
    return {d.get("name", n): n for n, d in G.nodes(data=True)}


def _node_has_type(attrs: dict, wanted: str | None) -> bool:
    if not wanted:
        return True
    t = attrs.get("type")
    if isinstance(t, list):
        return wanted in t
    return t == wanted


def _contains_edge(G: nx.Graph, parent_name: str, child_name: str) -> bool:
    names = _node_name_map(G)
    p = names.get(parent_name)
    c = names.get(child_name)
    if p is None or c is None:
        return False
    if not G.has_edge(p, c):
        return False
    typ = G.edges[p, c].get("type")
    return typ == "RK_CONTAINS" or (isinstance(typ, list) and "RK_CONTAINS" in typ)


def _direct_children_names(G: nx.Graph, scope: str) -> list[str]:
    names = _node_name_map(G)
    parent = names.get(scope)
    if parent is None:
        return []
    out = []
    for _, child, ed in G.out_edges(parent, data=True):
        typ = ed.get("type")
        if typ == "RK_CONTAINS" or (isinstance(typ, list) and "RK_CONTAINS" in typ):
            out.append(G.nodes[child].get("name", child))
    return out


def _count_type_under_scope(G: nx.Graph, scope: str, node_type: str) -> int:
    names = _node_name_map(G)
    root = names.get(scope)
    if root is None:
        return 0
    count = 0
    queue = [root]
    seen = set()
    while queue:
        node = queue.pop(0)
        if node in seen:
            continue
        seen.add(node)
        for _, child, ed in G.out_edges(node, data=True):
            typ = ed.get("type")
            if typ == "RK_CONTAINS" or (isinstance(typ, list) and "RK_CONTAINS" in typ):
                if _node_has_type(G.nodes[child], node_type):
                    count += 1
                queue.append(child)
    return count


def _rank_direct_children(G: nx.Graph, scope: str) -> list[tuple[str, int]]:
    names = _node_name_map(G)
    parent = names.get(scope)
    if parent is None:
        return []
    ranked: list[tuple[str, int]] = []
    for _, child, ed in G.out_edges(parent, data=True):
        typ = ed.get("type")
        if not (typ == "RK_CONTAINS" or (isinstance(typ, list) and "RK_CONTAINS" in typ)):
            continue
        capacity = 0
        if _node_has_type(G.nodes[child], "EK_PORT"):
            capacity += int(G.nodes[child].get("physical_capacity_bps", 0) or 0)
        for _, grandchild, ged in G.out_edges(child, data=True):
            gtyp = ged.get("type")
            if gtyp == "RK_CONTAINS" or (isinstance(gtyp, list) and "RK_CONTAINS" in gtyp):
                if _node_has_type(G.nodes[grandchild], "EK_PORT"):
                    capacity += int(G.nodes[grandchild].get("physical_capacity_bps", 0) or 0)
        ranked.append((G.nodes[child].get("name", child), capacity))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def _normalise_sequence(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return [value]
    out = []
    for item in value:
        if isinstance(item, list):
            out.append(tuple(item))
        else:
            out.append(item)
    return out


def _data_matches(expected: Any, got: Any, shape: str) -> bool:
    if shape == "text":
        return str(got) == str(expected)
    if shape == "list":
        return Counter(_normalise_sequence(got)) == Counter(_normalise_sequence(expected))
    return got == expected


def _semantic_check(ret: dict, goal: dict, updated_graph: nx.Graph | None) -> tuple[bool, str]:
    """Approximate semantic equivalence: check type/return-shape and that the
    operation in the code matches the operation in the typed Goal.

    For MALT, the strongest check we can cheaply do is:
      - return_shape matches goal.return_shape (graph/text/list/table)
      - if goal includes an 'add' step with target X, check X is in updated_graph
      - if goal includes a 'remove' step with target X, check X is NOT in updated_graph
      - if goal includes an 'update' step with target X attr=A val=V, check that
        the node exists in updated_graph and has X[A] == V
    """
    if not isinstance(ret, dict):
        return False, "ret is not a dict"
    if "type" not in ret or "data" not in ret:
        return False, "ret missing 'type' or 'data'"

    expected_shape = goal.get("return_shape", "graph")
    got_shape = ret.get("type")
    # Allow text↔count interchange (NetArena tolerates this)
    shape_ok = (expected_shape == got_shape) or (
        expected_shape == "text" and got_shape in ("text",)) or (
        expected_shape == "list" and got_shape in ("list",)) or (
        expected_shape == "graph" and got_shape == "graph")
    if not shape_ok:
        return False, f"return shape mismatch: expected={expected_shape} got={got_shape}"

    if updated_graph is None:
        return False, "updated_graph is required for semantic and safety validation"
    G = updated_graph

    if got_shape == "graph":
        data_graph, data_err = _graph_from_payload(ret.get("data"))
        if data_graph is None:
            return False, f"graph return data invalid: {data_err}"
        if not _graph_snapshots_equal(data_graph, G):
            return False, "graph return data and updated_graph describe different network states"

    # Check semantics step-by-step
    name_set = {d.get("name", n): n for n, d in G.nodes(data=True)}
    for st in goal.get("operations", []):
        op = st["op"]; tgt = st.get("target"); params = st.get("params") or {}
        if op == "add" and tgt:
            if tgt not in name_set:
                return False, f"add: target `{tgt}` not in updated_graph"
            target_type = params.get("type")
            parent = params.get("parent")
            if target_type and not _node_has_type(G.nodes[name_set[tgt]], target_type):
                return False, f"add: target `{tgt}` has type {G.nodes[name_set[tgt]].get('type')}, expected {target_type}"
            if parent and not _contains_edge(G, parent, tgt):
                return False, f"add: target `{tgt}` is not directly contained by `{parent}`"
        elif op == "remove" and tgt:
            if tgt in name_set:
                return False, f"remove: target `{tgt}` still in updated_graph"
        elif op == "update" and tgt:
            n_id = name_set.get(tgt)
            if n_id is None:
                return False, f"update: target `{tgt}` not in updated_graph"
            attr = params.get("attr"); val = params.get("value")
            if attr and (G.nodes[n_id].get(attr) != val):
                return False, (f"update: attr `{attr}` of `{tgt}` is "
                               f"{G.nodes[n_id].get(attr)}, expected {val}")
        elif op == "count":
            node_type = params.get("node_type")
            scope = params.get("scope")
            if node_type and scope:
                expected = _count_type_under_scope(G, scope, node_type)
                if not _data_matches(expected, ret.get("data"), "text"):
                    return False, f"count: data={ret.get('data')} expected={expected} under `{scope}`"
        elif op == "list":
            scope = params.get("scope")
            if scope:
                expected = _direct_children_names(G, scope)
                if not _data_matches(expected, ret.get("data"), "list"):
                    return False, f"list: data does not match direct children under `{scope}`"
        elif op == "rank":
            scope = params.get("scope")
            if scope:
                expected = _rank_direct_children(G, scope)
                if not _data_matches(expected, ret.get("data"), "list"):
                    return False, f"rank: data does not match ranked direct children under `{scope}`"
    return True, ""


# ---------- top-level ----------

# Same exec namespace items NetArena uses, so the LLM can call `solid_step_*`
# helpers if it wants to — keeps semantics aligned.
def _exec_namespace_for(graph_data: nx.Graph) -> dict:
    import networkx as nx_, json as json_
    # Hot-import the helpers (require app-malt to be on sys.path)
    from solid_step_helper import (
        solid_step_add_node_to_graph, solid_step_counting_query,
        solid_step_remove_node_from_graph, solid_step_list_child_nodes,
        solid_step_update_node_value, solid_step_rank_child_nodes,
    )
    return {
        "G": graph_data,
        "copy": copy, "nx": nx_, "json": json_,
        "solid_step_add_node_to_graph": solid_step_add_node_to_graph,
        "solid_step_counting_query": solid_step_counting_query,
        "solid_step_remove_node_from_graph": solid_step_remove_node_from_graph,
        "solid_step_list_child_nodes": solid_step_list_child_nodes,
        "solid_step_update_node_value": solid_step_update_node_value,
        "solid_step_rank_child_nodes": solid_step_rank_child_nodes,
    }


def verify(code: str, goal: dict, malt_graph: nx.DiGraph) -> Proof:
    """Run all checks on a candidate solution. Always returns a Proof."""
    p = Proof()
    t0 = time.perf_counter()

    # 1. Syntax
    p.syntax_ok, p.syntax_error = _check_syntax(code)
    if not p.syntax_ok:
        p.repair_hint = f"The code does not parse: {p.syntax_error}. Fix the Python syntax."
        p.elapsed_s = time.perf_counter() - t0
        return p

    # 2. Type/structure (signature)
    p.type_ok, p.type_error = _check_signature(code)
    if not p.type_ok:
        p.repair_hint = (f"The code is missing a valid `def process_graph(graph_data)` definition: "
                         f"{p.type_error}.")
        p.elapsed_s = time.perf_counter() - t0
        return p

    # 3. Counterfactual execution against a fresh copy of the graph.
    ns = _exec_namespace_for(copy.deepcopy(malt_graph))
    ret: Any = None
    try:
        exec(code, ns)
        ret = eval("process_graph(copy.deepcopy(G))", ns)
        p.sim_executed = True
    except Exception:
        p.sim_error = traceback.format_exc(limit=4)
        p.repair_hint = ("Execution failed in sandbox:\n" + p.sim_error.splitlines()[-1] +
                         "\nFix the implementation so process_graph(graph_data) runs without error.")
        p.elapsed_s = time.perf_counter() - t0
        return p

    if not isinstance(ret, dict):
        p.sim_error = f"process_graph returned a {type(ret).__name__}, expected dict"
        p.repair_hint = p.sim_error + " — return a dict with keys 'type','data','updated_graph'."
        p.elapsed_s = time.perf_counter() - t0
        return p
    p.sim_ret_type = str(ret.get("type"))
    p.sim_ret_data_kind = type(ret.get("data")).__name__
    p.sim_ret_data_summary = _summary_of_data(ret.get("data"))
    p.sim_updated_graph_present = "updated_graph" in ret

    # 4. Local invariants on updated graph. NetArena's official evaluator applies
    # safety to `updated_graph` whenever the output is valid, including text/list
    # query tasks, so VISTA must require the same proof obligation here.
    ug = ret.get("updated_graph")
    Gnew: nx.Graph | None = None
    if ug is None:
        p.invariant_error = "updated_graph missing; official safety checker cannot validate the proposed network state"
    else:
        Gnew, graph_err = _graph_from_payload(ug)
        if Gnew is None:
            p.invariant_error = graph_err
        else:
            ok, err = _safety_check(Gnew)
            p.local_invariant_ok = ok
            p.global_invariant_ok = ok
            if not ok:
                p.invariant_error = err

    # 5. Semantic equivalence to typed goal
    sem_ok, sem_err = _semantic_check(ret, goal, Gnew)
    p.semantic_eq_ok = sem_ok
    p.semantic_error = sem_err if not sem_ok else ""

    # 6. SimSafeOK: re-execution OK + invariants OK + semantic OK
    p.sim_safe_ok = p.sim_executed and p.local_invariant_ok and p.semantic_eq_ok

    # Compose
    p.passed = (p.syntax_ok and p.type_ok and p.sim_executed and
                p.semantic_eq_ok and p.local_invariant_ok)
    # Confidence: weighted Bernoulli average
    weights = {"syntax_ok":0.10,"type_ok":0.10,"sim_executed":0.20,
               "local_invariant_ok":0.20,"semantic_eq_ok":0.30,"sim_safe_ok":0.10}
    p.confidence = sum(weights[k]*float(getattr(p,k)) for k in weights)

    # Repair hints
    if not p.passed:
        bits = []
        if not p.semantic_eq_ok:
            bits.append("Semantic mismatch with the typed goal: " + p.semantic_error)
        if not p.local_invariant_ok:
            bits.append("Updated graph violates an invariant: " + p.invariant_error)
        if not p.sim_updated_graph_present:
            bits.append("Missing 'updated_graph' key in return_object.")
        if any(st.get("op") == "remove" for st in goal.get("operations", [])) and "isolated" in p.invariant_error.lower():
            bits.append(
                "If the removed node has contained descendants, remove the contained subtree as well "
                "before answering so the updated_graph has no isolated child nodes."
            )
        p.repair_hint = "\n".join(bits) if bits else "Failure with unknown cause; reread the goal."

    p.elapsed_s = time.perf_counter() - t0
    return p
