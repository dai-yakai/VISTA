"""Final aggregation: combine all *.summary.json results in outputs/ into a
single comparison table (CSV + Markdown) and produce per-task-label breakdowns.
"""
from __future__ import annotations

import argparse, glob, json, os
from collections import defaultdict

from . import config as C


METHOD_ORDER = [
    "baseline_zeroshot_base",
    "baseline_zeroshot_cot",
    "baseline_fewshot_base",
    "baseline_fewshot_cot",
    "baseline_fewshot_semantic",
    "baseline_sft_only",
    "vista_base",
    "vista_sft",
]


def _load_results_jsonl(p: str) -> list[dict]:
    out = []
    with open(p) as f:
        for line in f:
            out.append(json.loads(line))
    return out


def aggregate(out_dir: str = C.OUTPUTS_DIR):
    summary_paths = sorted(glob.glob(os.path.join(out_dir, "*.summary.json")))
    summaries = {}
    for sp in summary_paths:
        with open(sp) as f:
            s = json.load(f)
        method = s.get("baseline") or s.get("method") or os.path.basename(sp).replace(".summary.json","")
        # Normalise method name
        if method.startswith("baseline_"):
            mname = method
        elif method.startswith("vista"):
            mname = method
        else:
            mname = "baseline_" + method
        summaries[mname] = {**s, "method": mname}

    # Per-method by-label breakdown
    rows_overall = []
    rows_by_label: dict[str, list[dict]] = defaultdict(list)
    for sp in summary_paths:
        # Find the matching .jsonl
        p = sp.replace(".summary.json", ".jsonl")
        if not os.path.isfile(p):
            continue
        recs = _load_results_jsonl(p)
        s = summaries[next(k for k in summaries if k.endswith(os.path.basename(p).replace(".jsonl","")))]
        method = s["method"]
        # by label
        by_label = defaultdict(lambda: {"n":0,"c":0,"s":0,"lat":0.0})
        for r in recs:
            lab = r.get("Label", "?")
            by_label[lab]["n"] += 1
            by_label[lab]["c"] += int(r.get("Result-Correctness")=="Pass")
            by_label[lab]["s"] += int(r.get("Result-Safety")=="Pass")
            by_label[lab]["lat"] += float(r.get("Result-Latency", 0.0))
        for lab, v in by_label.items():
            n = max(1, v["n"])
            rows_by_label[lab].append({
                "method": method, "n": v["n"],
                "correctness": v["c"]/n, "safety": v["s"]/n,
                "avg_latency_s": v["lat"]/n,
            })

        rows_overall.append({
            "method": method,
            "n": s.get("n", len(recs)),
            "correctness": s.get("correctness", 0.0),
            "safety": s.get("safety", 0.0),
            "avg_latency_s": s.get("avg_latency_s", 0.0),
        })

    rows_overall.sort(key=lambda r: METHOD_ORDER.index(r["method"]) if r["method"] in METHOD_ORDER else 999)

    # Markdown table
    def fmt(v, n=4):
        return f"{v:.{n}f}" if isinstance(v, float) else str(v)
    md = ["# VISTA-Net++ vs Baselines (MALT, Qwen2.5-Coder-7B-Instruct)\n",
          "## Overall (n=1000 held-out test)\n",
          "| Method | n | Correctness | Safety | Avg Latency (s) |",
          "|---|---:|---:|---:|---:|"]
    for r in rows_overall:
        md.append(f"| {r['method']} | {r['n']} | {fmt(r['correctness'])} "
                  f"| {fmt(r['safety'])} | {fmt(r['avg_latency_s'])} |")

    # By-label
    for lab in sorted(rows_by_label.keys()):
        md += [f"\n### {lab}\n",
               "| Method | n | Correctness | Safety | Avg Latency (s) |",
               "|---|---:|---:|---:|---:|"]
        rows_by_label[lab].sort(key=lambda r: METHOD_ORDER.index(r["method"]) if r["method"] in METHOD_ORDER else 999)
        for r in rows_by_label[lab]:
            md.append(f"| {r['method']} | {r['n']} | {fmt(r['correctness'])} "
                      f"| {fmt(r['safety'])} | {fmt(r['avg_latency_s'])} |")

    out_md = os.path.join(out_dir, "RESULTS.md")
    with open(out_md, "w") as f:
        f.write("\n".join(md) + "\n")

    # CSV (simple)
    out_csv = os.path.join(out_dir, "RESULTS.csv")
    with open(out_csv, "w") as f:
        f.write("scope,method,n,correctness,safety,avg_latency_s\n")
        for r in rows_overall:
            f.write(f"overall,{r['method']},{r['n']},{r['correctness']:.6f},{r['safety']:.6f},{r['avg_latency_s']:.6f}\n")
        for lab, rs in rows_by_label.items():
            for r in rs:
                f.write(f'"{lab}",{r["method"]},{r["n"]},{r["correctness"]:.6f},{r["safety"]:.6f},{r["avg_latency_s"]:.6f}\n')

    print(f"Wrote: {out_md}\n      {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=C.OUTPUTS_DIR)
    args = ap.parse_args()
    aggregate(args.out_dir)


if __name__ == "__main__":
    main()
