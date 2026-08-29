#!/usr/bin/env bash
# Full pipeline using single vLLM session for generation + parallel scoring.
# Logs to logs/run_pipeline.log. Resumable (Phase 1 writes raw_*.jsonl, Phase 2
# writes *.jsonl/.summary.json — re-runs are idempotent).
set -euo pipefail
cd /home/ubuntu/vista

LOG=/home/ubuntu/vista/logs/run_pipeline.log
mkdir -p logs outputs models
exec > >(tee -a "$LOG") 2>&1

LIMIT=${1:-}
LIMIT_FLAG=""
[ -n "$LIMIT" ] && LIMIT_FLAG="--limit $LIMIT"

step() { echo; echo "=================================================================="; echo "[$(date +%H:%M:%S)] $*"; echo "=================================================================="; }

step "0/4 sanity: data split"
python -m src.data

# ---------- Phase A: generation passes that DON'T need a LoRA adapter ----------
step "A. base+vista generation (single vLLM session)"
python -m src.run_all_gen \
  $LIMIT_FLAG \
  --batch-size 16 \
  --vista-chunk-size 8 \
  --n-hyp 5 \
  --max-repair 3

# ---------- Score what we have so far ----------
step "score Phase-A outputs"
python -m src.score_all --num-workers 12 || true

# ---------- Phase B: train SFT ----------
step "B. SFT LoRA training"
python -m src.sft_train

SFT_ADAPTER=/home/ubuntu/vista/models/sft-qwen-coder-7b-malt/final
if [ ! -d "$SFT_ADAPTER" ]; then
  CKPT=$(ls -td /home/ubuntu/vista/models/sft-qwen-coder-7b-malt/checkpoint-* 2>/dev/null | head -1 || true)
  [ -n "$CKPT" ] && SFT_ADAPTER=$CKPT
fi
echo "SFT adapter: $SFT_ADAPTER"

# ---------- Phase C: generation passes that need LoRA ----------
if [ -d "$SFT_ADAPTER" ]; then
  step "C. sft_only + vista_sft generation (LoRA-enabled vLLM session)"
  python -m src.run_all_gen \
    --lora "$SFT_ADAPTER" \
    --baselines sft_only vista_sft \
    $LIMIT_FLAG \
    --batch-size 16 \
    --vista-chunk-size 8 \
    --n-hyp 5 \
    --max-repair 3
else
  echo "WARNING: SFT adapter missing, skipping LoRA-dependent runs"
fi

# ---------- Score final outputs ----------
step "score Phase-C outputs"
python -m src.score_all --num-workers 12 || true

# ---------- Aggregate ----------
step "aggregate"
python -m src.aggregate

echo
echo "ALL DONE — $(date)"
echo "Results: /home/ubuntu/vista/outputs/RESULTS.md"
