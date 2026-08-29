"""§7-§12 VISTA-Net++ planner: multi-hypothesis sampling + proof-guided repair.

Top-level loop (per query):

    Goal G   ← intent.compile_intent(q)
    B_t      ← belief.build_belief(G); summary = B_t.summarize_for_prompt(G)
    R_t      ← retrieval.topk(q, k=K_RET)
    H_t      ← LLM.sample(prompt(G, B_t, R_t), n=N_HYP, T=0.7)

    For each h in H_t:
        proof_h ← verifier.verify(extract_code(h), G, MALT_graph)

    If any proof.passed: return argmax(proof.confidence) sample
    Else: pick the failing sample with the highest confidence, build a repair
    prompt from its proof.repair_hint, and ask the LLM (greedy) for a fixed
    version. Repeat for up to K_REPAIR rounds.

We also implement a "ladder" — try the SFT LoRA first; if it can't pass, fall
back to the base model with a richer few-shot context. This stops VISTA from
being just an ensemble over noise.
"""
from __future__ import annotations

import json, time
from dataclasses import dataclass, field
from typing import Any, Sequence

from loguru import logger

from . import config as C
from . import prompts as P
from .intent import compile_intent, goal_to_hint
from .belief import build_belief, BeliefGraph
from .verifier import verify, Proof
from .polish import polish_code
from .reasoning import analyze_hypotheses, goal_retrieval_tag


@dataclass
class VistaResult:
    final_code: str = ""
    final_response: str = ""
    proof: Proof | None = None
    proof_passed: bool = False
    n_hypotheses_tried: int = 0
    n_repair_iterations: int = 0
    goal: dict | None = None
    belief_summary: str = ""
    planning_trace: str = ""
    hypothesis_trace: dict | None = None
    repair_history: list[dict] = field(default_factory=list)
    total_latency_s: float = 0.0


def _fewshot_from_records(records: Sequence[dict]) -> list[dict]:
    """Convert retrieval records (with 'answer' as ground_truth_process_graph) into fewshot examples."""
    out = []
    for r in records:
        out.append({
            "question": r["question"],
            "answer": r["answer"].replace("ground_truth_process_graph", "process_graph").strip(),
        })
    return out


def _build_messages(goal: dict, belief_summary: str, retrieved: list[dict],
                    planning_trace: str | None = None,
                    extra_hint: str | None = None) -> list[dict]:
    fewshot = _fewshot_from_records(retrieved)
    msgs = P.build_messages(goal["raw_query"], prompt_type="fewshot_semantic",
                            fewshot_examples=fewshot)
    # Inject the belief summary + the typed-goal hint as an additional system message.
    extra = "\n\n".join([
        goal_to_hint(goal),
        belief_summary,
        planning_trace or "",
        ("# Hint:\n" + extra_hint) if extra_hint else "",
    ]).strip()
    msgs[0]["content"] = msgs[0]["content"] + "\n\n" + extra
    return msgs


def _build_repair_messages(goal: dict, belief_summary: str, retrieved: list[dict],
                            prev_code: str, prev_proof: Proof,
                            planning_trace: str | None = None) -> list[dict]:
    hint = (
        "Your previous attempt failed verification. Concrete diagnostics:\n"
        f"- syntax_ok={prev_proof.syntax_ok}\n"
        f"- type_ok={prev_proof.type_ok}\n"
        f"- semantic_eq_ok={prev_proof.semantic_eq_ok}\n"
        f"- local_invariant_ok={prev_proof.local_invariant_ok}\n"
        f"- ret_type={prev_proof.sim_ret_type}, updated_graph_present={prev_proof.sim_updated_graph_present}\n"
        f"\nReason:\n{prev_proof.repair_hint}\n"
        f"\nPrevious attempt:\n```python\n{prev_code}\n```\n"
        "Return a corrected `def process_graph(graph_data)` ONLY (one fenced code block)."
    )
    return _build_messages(goal, belief_summary, retrieved, planning_trace, extra_hint=hint)


def run_vista_one(
    *,
    question: str,
    engine,
    retriever,
    train_pool: list[dict],
    malt_graph,
    n_hyp: int = C.VISTA_NUM_HYPOTHESES,
    n_retrieve: int = C.VISTA_RETRIEVAL_K,
    max_repair: int = C.VISTA_MAX_REPAIR_ITER,
    temperature_sample: float = C.GEN_TEMPERATURE_SAMPLE,
    lora_request=None,
    exclude_ids: set[int] | None = None,
) -> VistaResult:
    """Run VISTA-Net++ for a single query, returning code + proof."""
    t0 = time.perf_counter()

    # 1) Intent
    goal = compile_intent(question)

    # 2) Belief snapshot
    belief = build_belief(goal)
    belief_summary = belief.summarize_for_prompt(goal)

    # 3) Retrieval (semantic)
    if retriever is not None:
        idxs = retriever.topk(
            question,
            k=n_retrieve,
            tag=goal_retrieval_tag(goal),
            exclude_ids=exclude_ids,
        )
        retrieved = [train_pool[i] for i in idxs]
    else:
        retrieved = []
    planning_trace, hypothesis_trace = analyze_hypotheses(goal, belief, retrieved)

    # 4) Hypothesis voting (multi-sample)
    msgs = _build_messages(goal, belief_summary, retrieved, planning_trace)
    sampled = engine.generate(
        [msgs], temperature=temperature_sample, top_p=C.GEN_TOP_P,
        max_tokens=C.GEN_MAX_TOKENS, n=n_hyp, lora_request=lora_request,
    )[0]

    # 5) Verify each candidate
    candidates: list[dict] = []
    for s in sampled:
        code = polish_code(P.extract_code_output(s["text"]), goal)
        proof = verify(code, goal, malt_graph)
        candidates.append({"text": s["text"], "code": code, "proof": proof})

    candidates.sort(key=lambda x: (x["proof"].passed, x["proof"].confidence), reverse=True)
    best = candidates[0]

    # 6) If best passes, done.
    if best["proof"].passed:
        return VistaResult(
            final_code=best["code"], final_response=best["text"],
            proof=best["proof"], proof_passed=True,
            n_hypotheses_tried=len(candidates), n_repair_iterations=0,
            goal=goal, belief_summary=belief_summary,
            planning_trace=planning_trace, hypothesis_trace=hypothesis_trace,
            total_latency_s=time.perf_counter() - t0,
        )

    # 7) Repair loop
    history: list[dict] = []
    for it in range(max_repair):
        repair_msgs = _build_repair_messages(goal, belief_summary, retrieved,
                                              best["code"], best["proof"], planning_trace)
        rep_out = engine.generate(
            [repair_msgs], temperature=C.GEN_TEMPERATURE_GREEDY, top_p=C.GEN_TOP_P,
            max_tokens=C.GEN_MAX_TOKENS, n=1, lora_request=lora_request,
        )[0][0]
        rep_code = polish_code(P.extract_code_output(rep_out["text"]), goal)
        rep_proof = verify(rep_code, goal, malt_graph)
        history.append({
            "iter": it,
            "passed": rep_proof.passed,
            "confidence": rep_proof.confidence,
            "repair_hint": best["proof"].repair_hint,
        })
        if rep_proof.passed or rep_proof.confidence > best["proof"].confidence:
            best = {"text": rep_out["text"], "code": rep_code, "proof": rep_proof}
        if rep_proof.passed:
            break

    return VistaResult(
        final_code=best["code"], final_response=best["text"],
        proof=best["proof"], proof_passed=best["proof"].passed,
        n_hypotheses_tried=len(candidates), n_repair_iterations=len(history),
        goal=goal, belief_summary=belief_summary,
        planning_trace=planning_trace, hypothesis_trace=hypothesis_trace,
        repair_history=history,
        total_latency_s=time.perf_counter() - t0,
    )
