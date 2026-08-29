"""Re-extract `code` field in raw_*.jsonl using the current `extract_code_output`.
Useful when the extractor logic changes (e.g., new HTML-comment stripping).

Usage: python -m src.reextract raw_<name>.jsonl [...]
"""
from __future__ import annotations
import json, sys
from src.prompts import extract_code_output


def main():
    paths = sys.argv[1:]
    for p in paths:
        recs = []
        with open(p) as f:
            for line in f:
                recs.append(json.loads(line))
        n_changed = 0
        for r in recs:
            new_code = extract_code_output(r.get("raw_response", ""))
            if new_code != r.get("code", ""):
                n_changed += 1
            r["code"] = new_code
        with open(p, "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{p}: {n_changed}/{len(recs)} re-extracted")


if __name__ == "__main__":
    main()
