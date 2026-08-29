#!/usr/bin/env bash
#
# End-to-end orchestration for VISTA-Net++ on MALT.
# Each step writes its own results & summary; aggregation is rerunnable.
#
# Usage: bash scripts/run_all.sh [LIMIT]
#   LIMIT (optional): if set, only the first LIMIT test examples are used. Use
#                     for fast smoke tests (e.g. LIMIT=50).
set -euo pipefail
cd /home/ubuntu/vista

LIMIT=${1:-}
LIMIT_FLAG=""
[ -n "$LIMIT" ] && LIMIT_FLAG="--limit $LIMIT"

log() { echo "[run_all $(date +%H:%M:%S)] $*"; }

# 0) data prep (idempotent)
log "0/8 prep data"
python -m src.data

# 1-4) prompting baselines (no LoRA)
for name in zeroshot_base zeroshot_cot fewshot_base fewshot_semantic; do
  log "running baseline: $name"
  python -m src.run_baseline --name "$name" $LIMIT_FLAG --batch-size 16
done

# 5) SFT training
log "5/8 SFT training"
python -m src.sft_train

# 6) sft_only baseline (LoRA + base prompt)
SFT_ADAPTER="/home/ubuntu/vista/models/sft-qwen-coder-7b-malt/final"
log "running baseline: sft_only with adapter=$SFT_ADAPTER"
python -m src.run_baseline --name sft_only --lora "$SFT_ADAPTER" $LIMIT_FLAG --batch-size 16

# 7) VISTA-Net++ (no LoRA = on top of plain base model)
log "7/8 VISTA-Net++ on base"
python -m src.run_vista $LIMIT_FLAG --chunk-size 8 --n-hyp 5 --max-repair 3

# 7b) VISTA-Net++ on SFT
log "7b/8 VISTA-Net++ on SFT"
python -m src.run_vista --lora "$SFT_ADAPTER" $LIMIT_FLAG --chunk-size 8 --n-hyp 5 --max-repair 3

# 8) Aggregate everything
log "8/8 aggregate"
python -m src.aggregate

log "ALL DONE — see outputs/RESULTS.md"
