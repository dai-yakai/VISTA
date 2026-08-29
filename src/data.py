"""Data preparation: loading and stratified splitting of the MALT benchmark.

Source format (one obj per line):
  {"messages": [
      {"question": "..."},
      {"answer": "def ground_truth_process_graph(graph_data): ..."},
      {"task_label": "capacity planning, level-1, add"}
  ]}

Output (per-line records):
  {"id": int, "question": str, "answer": str, "task_label": str, "complexity": "level-1|2|3", "op": "add|...", "split": "train|test"}
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Iterator

from . import config as C


@dataclass
class Example:
    id: int
    question: str
    answer: str            # ground-truth code (defines `ground_truth_process_graph`)
    task_label: str        # e.g. "capacity planning, level-1, add"
    complexity: str        # level-1 | level-2 | level-3
    op: str                # add | rank | remove | list | remove-list | ...
    split: str = ""        # "train" | "test"

    def to_json(self) -> dict:
        return asdict(self)


def _parse_label(task_label: str) -> tuple[str, str]:
    parts = [p.strip() for p in task_label.split(",")]
    if len(parts) >= 3:
        return parts[1], parts[2]
    return parts[1] if len(parts) > 1 else "unknown", parts[-1]


def load_raw(path: str = C.MALT_RAW_DATA) -> list[Example]:
    out: list[Example] = []
    with open(path) as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            q = a = lab = None
            for m in obj["messages"]:
                if "question" in m:   q = m["question"]
                elif "answer" in m:   a = m["answer"]
                elif "task_label" in m: lab = m["task_label"]
            assert q and a and lab, f"line {i} missing fields"
            comp, op = _parse_label(lab)
            out.append(Example(id=i, question=q, answer=a, task_label=lab,
                               complexity=comp, op=op))
    return out


def stratified_split(
    examples: list[Example],
    test_per_label: int = C.TEST_PER_LABEL,
    train_per_label: int = C.TRAIN_PER_LABEL,
    seed: int = C.SEED,
) -> tuple[list[Example], list[Example]]:
    by_label: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        by_label[ex.task_label].append(ex)

    rng = random.Random(seed)
    train, test = [], []
    for label, items in sorted(by_label.items()):
        rng.shuffle(items)
        # Take last test_per_label as test (deterministic given seed)
        cut = len(items) - test_per_label
        train_part = items[:cut][:train_per_label]
        test_part = items[cut:]
        for x in train_part:
            x.split = "train"
        for x in test_part:
            x.split = "test"
        train.extend(train_part)
        test.extend(test_part)
    return train, test


def write_jsonl(records: list[Example], path: str) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
    return out


def main():
    examples = load_raw()
    print(f"Loaded {len(examples)} examples")

    label_counts = defaultdict(int)
    for ex in examples:
        label_counts[ex.task_label] += 1
    print(f"Labels: {len(label_counts)}; min={min(label_counts.values())}, max={max(label_counts.values())}")

    train, test = stratified_split(examples)
    print(f"Split: train={len(train)} test={len(test)}")

    write_jsonl(train, C.TRAIN_FILE)
    write_jsonl(test, C.TEST_FILE)
    print(f"Wrote: {C.TRAIN_FILE} {C.TEST_FILE}")


if __name__ == "__main__":
    main()
