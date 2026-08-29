#!/usr/bin/env bash
# Resume from SFT training onwards.
set -euo pipefail
cd /home/ubuntu/vista

LOG=/home/ubuntu/vista/logs/run_pipeline.log
exec > >(tee -a "$LOG") 2>&1

step() { echo; echo "=================================================================="; echo "[$(date +%H:%M:%S)] $*"; echo "=================================================================="; }

step "B (resume). SFT LoRA training"
python -m src.sft_train

SFT_ADAPTER=/home/ubuntu/vista/models/sft-qwen-coder-7b-malt/final
if [ ! -d "$SFT_ADAPTER" ]; then
  CKPT=$(ls -td /home/ubuntu/vista/models/sft-qwen-coder-7b-malt/checkpoint-* 2>/dev/null | head -1 || true)
  [ -n "$CKPT" ] && SFT_ADAPTER=$CKPT
fi
echo "SFT adapter: $SFT_ADAPTER"

step "C. sft_only + vista_sft generation"
python -m src.run_all_gen \
  --lora "$SFT_ADAPTER" \
  --baselines sft_only vista_sft \
  --batch-size 16 \
  --vista-chunk-size 8 \
  --n-hyp 5 \
  --max-repair 3

step "score Phase-C outputs"
python -m src.score_all --num-workers 12

step "aggregate"
python -m src.aggregate

echo "ALL DONE — $(date)"
