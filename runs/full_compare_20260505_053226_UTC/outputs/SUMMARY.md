# NetArena vs VISTA Full Comparison Summary

## Run Metadata

- Run ID: `full_compare_20260505_053226_UTC`
- UTC time: `2026-05-05T05:32:26Z` to `2026-05-05T08:26:10Z`; duration `2:53:44`
- Test set: `1000` held-out MALT/NetArena samples, `10` scopes x `100` samples
- Base model: `/home/ubuntu/vista/models/Qwen2.5-Coder-7B-Instruct`
- SFT adapter: `/home/ubuntu/vista/models/sft-qwen-coder-7b-malt/final`
- SFT adapter was reused from the existing checkpoint for this run; it was not retrained during this comparison.
- Compared methods: zero-shot, zero-shot CoT, few-shot, semantic few-shot, SFT-only, VISTA-base, VISTA-SFT.

## Overall Results

| Method | Family | N | Correctness | Safety | Avg latency (s) |
|---|---|---:|---:|---:|---:|
| NetArena zero-shot | NetArena baseline | 1000 | 0.580 | 0.464 | 0.280 |
| NetArena zero-shot CoT | NetArena baseline | 1000 | 0.418 | 0.418 | 0.392 |
| NetArena few-shot | NetArena baseline | 1000 | 0.800 | 0.554 | 0.281 |
| NetArena semantic few-shot | NetArena baseline | 1000 | 1.000 | 0.589 | 0.271 |
| SFT-only baseline | NetArena baseline | 1000 | 1.000 | 0.000 | 0.236 |
| VISTA on base model | VISTA | 1000 | 1.000 | 0.709 | 2.460 |
| VISTA on SFT adapter | VISTA | 1000 | 1.000 | 0.709 | 2.504 |

## Main Takeaways

- `VISTA on base model` reaches correctness `1.000` and safety `0.709`, the best single-method safety in this run.
- Against the best single baseline by safety, `NetArena semantic few-shot` at `0.589`, `VISTA on base model` improves safety by `+0.120` absolute.
- Against the best prompt-only baseline, `NetArena semantic few-shot` at `0.589`, `VISTA on base model` improves safety by `+0.120` absolute.
- `VISTA on SFT adapter` reaches correctness `1.000` and safety `0.709`, the best single-method safety in this run.
- Against the best single baseline by safety, `NetArena semantic few-shot` at `0.589`, `VISTA on SFT adapter` improves safety by `+0.120` absolute.
- Against the best prompt-only baseline, `NetArena semantic few-shot` at `0.589`, `VISTA on SFT adapter` improves safety by `+0.120` absolute.
- `SFT-only baseline` has correctness `1.000` but safety `0.000`; adding VISTA over the same SFT adapter restores safety to `0.709`.
- The tradeoff is latency: VISTA runs around `2.46` to `2.50` seconds per item versus roughly `0.24` to `0.39` seconds for the baselines.

## Per-Scope Snapshot

Values are `correctness/safety`. The full per-scope table for all methods is in `outputs/RESULTS.md` and `outputs/RESULTS.csv`.

| Scope | Semantic few-shot | SFT-only | VISTA-base | VISTA-SFT |
|---|---:|---:|---:|---:|
| capacity planning, level-1, add | 1.00/0.25 | 1.00/0.00 | 1.00/1.00 | 1.00/1.00 |
| capacity planning, level-1, list | 1.00/1.00 | 1.00/0.00 | 1.00/1.00 | 1.00/1.00 |
| capacity planning, level-1, rank | 1.00/1.00 | 1.00/0.00 | 1.00/1.00 | 1.00/1.00 |
| capacity planning, level-1, remove | 1.00/0.02 | 1.00/0.00 | 1.00/0.47 | 1.00/0.47 |
| capacity planning, level-2, remove-count | 1.00/0.55 | 1.00/0.00 | 1.00/0.55 | 1.00/0.55 |
| capacity planning, level-2, remove-list | 1.00/0.47 | 1.00/0.00 | 1.00/0.47 | 1.00/0.47 |
| capacity planning, level-2, remove-rank | 1.00/0.46 | 1.00/0.00 | 1.00/0.46 | 1.00/0.46 |
| capacity planning, level-3, add-count | 1.00/1.00 | 1.00/0.00 | 1.00/1.00 | 1.00/1.00 |
| capacity planning, level-3, add-list | 1.00/0.49 | 1.00/0.00 | 1.00/0.49 | 1.00/0.49 |
| capacity planning, level-3, add-rank | 1.00/0.65 | 1.00/0.00 | 1.00/0.65 | 1.00/0.65 |

## VISTA Generation Stats

| Method | Initial pass | Repair iterations |
|---|---:|---:|
| VISTA on base model | 947/1000 | 159 |
| VISTA on SFT adapter | 947/1000 | 159 |

## Artifact Paths

- Run directory: `/home/ubuntu/vista/runs/full_compare_20260505_053226_UTC`
- Full log: `/home/ubuntu/vista/runs/full_compare_20260505_053226_UTC/logs/full_compare.log`
- Archived results markdown: `/home/ubuntu/vista/runs/full_compare_20260505_053226_UTC/outputs/RESULTS.md`
- Archived results CSV: `/home/ubuntu/vista/runs/full_compare_20260505_053226_UTC/outputs/RESULTS.csv`
- Archived raw and scored JSONL files: `/home/ubuntu/vista/runs/full_compare_20260505_053226_UTC/outputs`
- Pre-existing outputs before this run: `/home/ubuntu/vista/runs/full_compare_20260505_053226_UTC/pre_existing_outputs`

## Integrity Checks

- Pipeline log ended with `DONE`.
- All `7` raw JSONL files have `1000` records each.
- All `7` scored JSONL files have `1000` records each.
- `RESULTS.md` and `RESULTS.csv` were regenerated and archived under this run directory.
