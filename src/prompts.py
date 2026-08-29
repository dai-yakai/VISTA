"""Baseline prompts for the MALT task.

We reuse NetArena's text content verbatim (so any baseline-vs-baseline comparison
is bit-equivalent in instruction wording), but format them as ChatML messages so
they slot into Qwen2.5-Coder-7B-Instruct's chat template. We expose builders that
match NetArena's four prompt strategies:

  - zeroshot_base
  - zeroshot_cot
  - fewshot_base
  - fewshot_cot

Plus a "fewshot_semantic" mode (top-k retrieved few-shot, used by both a baseline
and the VISTA agent).
"""
from __future__ import annotations

from typing import Sequence
from . import config as C

# ---------- NetArena-equivalent text blocks ----------

BASE_PROMPT = """You need to behave like a network engineer who processes graph data to answer user queries about capacity planning.

Your task is to generate the Python code needed to process the network graph to answer the user question or request. The code should take the form of a function named process_graph that accepts a single input argument graph_data and returns a single object return_object.

You have access (already imported in the execution scope) to these helper functions, and you SHOULD prefer them when applicable — they implement the canonical/expected semantics:
  - solid_step_add_node_to_graph(graph_data, new_node, parent_node_name=None)
  - solid_step_remove_node_from_graph(graph_data, target_node_name)
  - solid_step_counting_query(graph_data, parent_node, child_node_type)
  - solid_step_list_child_nodes(graph_data, parent_node_name)
  - solid_step_update_node_value(graph_data, target_node_name, new_value)
  - solid_step_rank_child_nodes(graph_data, parent_node_name)
where new_node is a dict like {'name': str, 'type': str}. Use STRING type names like 'EK_PORT' (not lists) when constructing new_node for solid_step_add_node_to_graph; the helper handles the rest, including default capacities and child ports for PACKET_SWITCH.

Graph Structure:
- The input graph_data is a networkx DiGraph object with nodes and edges
- Nodes carry attributes 'name' (str) and 'type' (list[str] or str). Allowed kinds: ['EK_SUPERBLOCK', 'EK_CHASSIS', 'EK_RACK', 'EK_AGG_BLOCK', 'EK_JUPITER', 'EK_PORT', 'EK_SPINEBLOCK', 'EK_PACKET_SWITCH', 'EK_CONTROL_POINT', 'EK_CONTROL_DOMAIN']
- Edges carry a 'type' attribute in {'RK_CONTAINS', 'RK_CONTROLS'}
- Hierarchy: JUPITER → SUPERBLOCK → AGG_BLOCK → PACKET_SWITCH → PORT (and CHASSIS → PACKET_SWITCH → PORT)

Important Guidelines:
- When updating a graph, always work on a deepcopy: `graph_copy = copy.deepcopy(graph_data)` then call the relevant helper on `graph_copy`.
- Identify nodes by checking their 'name' attribute (do NOT rely on node IDs); to find by name use a small helper or scan with `for n,d in graph_copy.nodes(data=True): if d.get('name') == ...`.
- For counting, listing, ranking: prefer the corresponding `solid_step_*` helper.
- Safety is checked on 'updated_graph' for EVERY answer, including text/list answers. Always return an updated_graph that satisfies the graph invariants.
- For remove followed by count/list/rank, if the removed node contains child nodes, remove the contained subtree as well before answering; otherwise the child ports become isolated and the safety check fails.
- For graph-returning remove tasks, keep 'data' and 'updated_graph' consistent with each other.

Output Format:
- Return ONE dict named `return_object` with keys 'type', 'data', 'updated_graph':
  * 'type'  ∈ {'text','list','table','graph'} per the question
  * 'data'  carries the answer in the requested form (string for 'text'/count, list of names for 'list', list of [name, capacity] tuples for 'rank', a graph JSON for 'graph')
  * 'updated_graph' = nx.readwrite.json_graph.node_link_data(graph_copy)  — REQUIRED in every return; even for read-only queries pass through the unchanged copy.

Response Format:
- Output exactly ONE fenced ```python``` code block defining `def process_graph(graph_data):` and nothing else.
- No imports.
"""

COT_HINT = "Think step by step before writing the function. Briefly state (1) the operation type, (2) the target node(s), and (3) any safety/structural concerns; then output ONLY the final ```python``` block.\n"

# Same 4 examples as NetArena's `EXAMPLE_LIST`, but rewritten so they invoke the
# `solid_step_*` helpers (which match the canonical/expected semantics used by
# the ground-truth code). This is the single most important correctness lever
# for prompt-only baselines.
EXAMPLE_LIST: list[dict] = [
    {
        "question": "Update the physical capacity value of ju1.a3.m2.s2c4.p10 to 72. Return a graph.",
        "answer": (
            "def process_graph(graph_data):\n"
            "    graph_copy = copy.deepcopy(graph_data)\n"
            "    graph_copy = solid_step_update_node_value(graph_copy, 'ju1.a3.m2.s2c4.p10', 72)\n"
            "    graph_json = nx.readwrite.json_graph.node_link_data(graph_copy)\n"
            "    return_object = {'type': 'graph', 'data': graph_copy, 'updated_graph': graph_json}\n"
            "    return return_object"
        ),
    },
    {
        "question": "Add new node with name new_EK_PORT_82 type EK_PORT, to ju1.a2.m4.s3c6. Return a graph.",
        "answer": (
            "def process_graph(graph_data):\n"
            "    graph_copy = copy.deepcopy(graph_data)\n"
            "    new_node = {'name': 'new_EK_PORT_82', 'type': 'EK_PORT'}\n"
            "    graph_copy = solid_step_add_node_to_graph(graph_copy, new_node, 'ju1.a2.m4.s3c6')\n"
            "    graph_json = nx.readwrite.json_graph.node_link_data(graph_copy)\n"
            "    return_object = {'type': 'graph', 'data': graph_copy, 'updated_graph': graph_json}\n"
            "    return return_object"
        ),
    },
    {
        "question": "Count the EK_PACKET_SWITCH in the ju1.a2.dom. Return only the count number.",
        "answer": (
            "def process_graph(graph_data):\n"
            "    graph_copy = copy.deepcopy(graph_data)\n"
            "    count = solid_step_counting_query(graph_copy, {'name': 'ju1.a2.dom'}, {'type': 'EK_PACKET_SWITCH'})\n"
            "    graph_json = nx.readwrite.json_graph.node_link_data(graph_copy)\n"
            "    return_object = {'type': 'text', 'data': str(count), 'updated_graph': graph_json}\n"
            "    return return_object"
        ),
    },
    {
        "question": "Remove ju1.a1.m4.s3c6.p1 from the graph. Return a graph.",
        "answer": (
            "def process_graph(graph_data):\n"
            "    graph_copy = copy.deepcopy(graph_data)\n"
            "    graph_copy = solid_step_remove_node_from_graph(graph_copy, 'ju1.a1.m4.s3c6.p1')\n"
            "    graph_json = nx.readwrite.json_graph.node_link_data(graph_copy)\n"
            "    return_object = {'type': 'graph', 'data': graph_copy, 'updated_graph': graph_json}\n"
            "    return return_object"
        ),
    },
    {
        "question": "List all the child nodes of ju1.s3.dom. Return a list of child node names.",
        "answer": (
            "def process_graph(graph_data):\n"
            "    graph_copy = copy.deepcopy(graph_data)\n"
            "    parent = {'name': 'ju1.s3.dom'}\n"
            "    children = solid_step_list_child_nodes(graph_copy, parent)\n"
            "    graph_json = nx.readwrite.json_graph.node_link_data(graph_copy)\n"
            "    return_object = {'type': 'list', 'data': children, 'updated_graph': graph_json}\n"
            "    return return_object"
        ),
    },
    {
        "question": "Rank all child nodes of EK_AGG_BLOCK type ju1.a3.m2 based on physical_capacity_bps attribute. Return a list of tuple, each tuple has child node name and its total physical capacity.",
        "answer": (
            "def process_graph(graph_data):\n"
            "    graph_copy = copy.deepcopy(graph_data)\n"
            "    ranked = solid_step_rank_child_nodes(graph_copy, 'ju1.a3.m2')\n"
            "    graph_json = nx.readwrite.json_graph.node_link_data(graph_copy)\n"
            "    return_object = {'type': 'list', 'data': ranked, 'updated_graph': graph_json}\n"
            "    return return_object"
        ),
    },
]

USER_QUERY_TEMPLATE = (
    "Question: {question}\n\n"
    "Answer with only one Python code block (```python ... ```) defining `def process_graph(graph_data)`. "
    "The function must `return return_object` where `return_object` is a dict with keys 'type', 'data', 'updated_graph'."
)

# ---------- ChatML message builders ----------


def _format_examples(exs: Sequence[dict]) -> str:
    chunks = []
    for ex in exs:
        chunks.append(
            f"Question: {ex['question']}\n\nAnswer:\n```python\n{ex['answer']}\n```"
        )
    return "\n\n".join(chunks)


def build_messages(
    question: str,
    *,
    prompt_type: str = "zeroshot_base",
    fewshot_examples: Sequence[dict] | None = None,
) -> list[dict]:
    """Build a ChatML-style list[message] for the given question.

    prompt_type ∈ {zeroshot_base, zeroshot_cot, fewshot_base, fewshot_cot, fewshot_semantic}
    fewshot_examples: list of {"question","answer"} dicts. Required for fewshot_semantic.
    """
    sys_parts = [BASE_PROMPT.strip()]
    if prompt_type in ("zeroshot_cot", "fewshot_cot"):
        sys_parts.append(COT_HINT.strip())

    if prompt_type in ("fewshot_base", "fewshot_cot"):
        sys_parts.append("Here are a few examples of questions and their corresponding answers:")
        sys_parts.append(_format_examples(EXAMPLE_LIST))
    elif prompt_type == "fewshot_semantic":
        if not fewshot_examples:
            raise ValueError("fewshot_semantic requires `fewshot_examples`")
        sys_parts.append("Here are a few examples of questions and their corresponding answers:")
        sys_parts.append(_format_examples(fewshot_examples))

    system = "\n\n".join(sys_parts)
    user = USER_QUERY_TEMPLATE.format(question=question)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ---------- Programmatic code-extraction (mirrors NetArena's extract_code_output) ----------
import re

CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
PROCESS_GRAPH_DEF_RE = re.compile(r"def\s+([a-zA-Z_0-9]*process_graph[a-zA-Z_0-9]*)")


def extract_code_output(answer: str) -> str:
    """Robustly extract `def process_graph(...)` body from an LLM response.

    Compatible with NetArena's `extract_code_output`. Tries fenced code first,
    then falls back to substring extraction. Also strips HTML-style comments
    (e.g. SFT-style `<!-- intent: ... -->` headers that aren't valid Python).
    """
    if not answer:
        return ""
    # Try fenced code blocks first; pick the longest one that contains process_graph.
    best = None
    for m in CODE_FENCE_RE.finditer(answer):
        block = m.group(1)
        if "process_graph" in block:
            if best is None or len(block) > len(best):
                best = block
    if best is None:
        # Fallback: substring after `def *process_graph*`
        ans = PROCESS_GRAPH_DEF_RE.sub("def process_graph", answer)
        start = ans.find("def process_graph")
        if start == -1:
            return ""
        best = ans[start:].strip()

    code = PROCESS_GRAPH_DEF_RE.sub("def process_graph", best).strip()
    # Strip HTML-style comments that the SFT model emits as an intent header.
    code = re.sub(r"<!--.*?-->", "", code, flags=re.DOTALL).strip()
    # Also strip lines that are only an unparseable hint marker.
    lines = []
    for ln in code.splitlines():
        s = ln.strip()
        if s.startswith("import "):
            continue  # NetArena rule: no imports
        lines.append(ln)
    return "\n".join(lines).strip()
