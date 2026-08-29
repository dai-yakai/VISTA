"""Score all `outputs/raw_*.jsonl` files in parallel — runs after `run_all_gen`.

For each raw_<name>.jsonl, produces:
  - outputs/<name>.jsonl (or outputs/baseline_<name>.jsonl for prompt baselines)
  - outputs/<name>.summary.json
"""
from __future__ import annotations

import argparse, glob, json, os, sys, time

from loguru import logger

from . import config as C
from .parallel_score import score_records_parallel


def _is_baseline_prompt(name: str) -> bool:
    return name in {"zeroshot_base","zeroshot_cot","fewshot_base","fewshot_cot",
                    "fewshot_semantic","sft_only"}


def _out_path_for(name: str) -> str:
    if _is_baseline_prompt(name):
        return os.path.join(C.OUTPUTS_DIR, f"baseline_{name}.jsonl")
    else:
        return os.path.join(C.OUTPUTS_DIR, f"{name}.jsonl")


def _score_one(raw_path: str, num_workers: int):
    fname = os.path.basename(raw_path)  # raw_<name>.jsonl
    name = fname[len("raw_"):-len(".jsonl")]
    out_path = _out_path_for(name)
    if os.path.isfile(out_path) and os.path.isfile(out_path.replace(".jsonl",".summary.json")):
        # Skip if already scored AND up to date.
        if os.path.getmtime(out_path) >= os.path.getmtime(raw_path):
            logger.info(f"[score:{name}] up-to-date, skipping")
            return

    logger.info(f"[score:{name}] {raw_path} → {out_path} ({num_workers} workers)")
    records = []
    with open(raw_path) as f:
        for line in f:
            records.append(json.loads(line))
    t0 = time.perf_counter()
    res = score_records_parallel(records, num_workers=num_workers)
    elapsed = time.perf_counter() - t0
    logger.info(f"[score:{name}] {len(res)} records in {elapsed:.1f}s = {elapsed/max(1,len(res)):.2f}s/q")

    pass_correct = pass_safety = total = 0; sum_lat = 0.0
    with open(out_path, "w") as fout:
        for r, raw in zip(res, records):
            r["baseline"] = name
            r["id"] = raw.get("id")
            r["raw_response"] = raw.get("raw_response", "")
            r["extracted_code"] = raw.get("code", "")
            for k in ("proof", "goal", "belief_summary"):
                if k in raw: r[k] = raw[k]
            fout.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")
            total += 1
            if r.get("Result-Correctness") == "Pass": pass_correct += 1
            if r.get("Result-Safety")      == "Pass": pass_safety += 1
            sum_lat += float(r.get("Result-Latency", 0.0))

    summary = {
        "method": name,
        "n": total,
        "correctness": pass_correct/total if total else 0.0,
        "safety": pass_safety/total if total else 0.0,
        "avg_latency_s": sum_lat/total if total else 0.0,
        "out_path": out_path,
    }
    with open(out_path.replace(".jsonl",".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"[score:{name}] {summary}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--names", nargs="*", default=None,
                    help="Subset of names to score (without raw_ prefix)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(C.OUTPUTS_DIR, "raw_*.jsonl")))
    if args.names is not None:
        paths = [p for p in paths
                 if os.path.basename(p)[len("raw_"):-len(".jsonl")] in args.names]
    logger.info(f"will score {len(paths)} raw files")

    for p in paths:
        try:
            _score_one(p, args.num_workers)
        except Exception:
            logger.exception(f"failed scoring {p}")


if __name__ == "__main__":
    main()
