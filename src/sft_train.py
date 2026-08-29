"""SFT (LoRA) training for Qwen2.5-Coder-7B-Instruct on the MALT train split.

This implements `L_intent + L_plan` from §11 of methodology.md:
  - L_plan: cross-entropy on the (cleaned) ground-truth code as the assistant
    response, conditioned on a chat-formatted (system, user) prompt.
  - L_intent: implicit — task-type, op, and goal-shape are reflected in the
    response prefix produced by the model. We add a small "intent header"
    instruction (one tag) to make this explicit.

We deliberately train on only the train split (4000 examples) and never look at
the test split during training, so all evaluation numbers are out-of-sample.
"""
from __future__ import annotations

import argparse, json, os, sys
from dataclasses import asdict
from typing import Any, Sequence

import torch
from loguru import logger

from . import config as C
from . import prompts as P
from .data import read_jsonl


def _build_chat_dataset(records: Sequence[dict], tokenizer):
    """Render each (q, gt-code) into a ChatML-formatted training example."""
    out = []
    for r in records:
        # Strip "ground_truth_" prefix → use a clean `def process_graph(...)`.
        code = r["answer"].replace("ground_truth_process_graph", "process_graph").strip()
        # Wrap into the assistant's expected fenced-code answer with the intent header.
        intent_header = (
            f"<!-- intent: task_type={r['complexity']}; op={r['op']} -->\n"
        )
        assistant = f"```python\n{intent_header}{code}\n```"
        msgs = P.build_messages(r["question"], prompt_type="zeroshot_base") + [
            {"role": "assistant", "content": assistant}
        ]
        # Render with the chat template (ground-truth shape) and SFT-train on it.
        text = tokenizer.apply_chat_template(msgs, tokenize=False)
        out.append({"text": text})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.BASE_MODEL)
    ap.add_argument("--out", default=C.SFT_OUTPUT_DIR)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)  # for debugging
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    os.makedirs(args.out, exist_ok=True)
    logger.add(os.path.join(args.out, "train.log"), level="INFO")

    logger.info(f"Loading tokenizer & model from {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    train_records = read_jsonl(C.TRAIN_FILE)
    if args.limit is not None:
        train_records = train_records[: args.limit]
    logger.info(f"train records: {len(train_records)}")

    rendered = _build_chat_dataset(train_records, tok)
    ds = Dataset.from_list(rendered)
    logger.info(f"Sample[0] (truncated):\n{rendered[0]['text'][:500]}\n... ({len(rendered[0]['text'])} chars)")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    # gradient checkpointing requires use_cache=False
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
    )

    sft_args = SFTConfig(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.0,
        bf16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        seed=args.seed,
        max_seq_length=args.max_seq_len,
        packing=False,
        dataset_text_field="text",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
        save_strategy="epoch",
        # Help avoid OOM on 24GB
        dataloader_num_workers=2,
        optim="adamw_torch_fused",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=ds,
        peft_config=peft_config,
        args=sft_args,
    )
    trainer.train()

    # Save final adapter
    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    logger.info(f"Saved final adapter to {final}")
    print(json.dumps({"status":"ok","out":final}))


if __name__ == "__main__":
    main()
