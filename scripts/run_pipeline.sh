#!/usr/bin/env bash
# Sequentially run all baselines, then SFT, then vista_base, vista_sft.
# Each step writes to outputs/raw_*.jsonl (raw codes) and outputs/baseline_*.jsonl
# (final scored). Aggregation runs at the very end.
#
# Tail logs/ for live progress: `tail -F /home/ubuntu/vista/logs/run_pipeline.log`
set -euo pipefail
cd /home/ubuntu/vista

LOG=/home/ubuntu/vista/logs/run_pipeline.log
LIMIT=${1:-}
LIMIT_FLAG=""
[ -n "$LIMIT" ] && LIMIT_FLAG="--limit $LIMIT"

mkdir -p logs outputs models
exec > >(tee -a "$LOG") 2>&1

step() { echo; echo "=================================================================="; echo "[$(date +%H:%M:%S)] $*"; echo "=================================================================="; }

step "0/9 sanity: data split"
python -m src.data

# ---------- Prompting baselines (no LoRA) ----------
for name in zeroshot_base zeroshot_cot fewshot_base fewshot_semantic; do
  step "running baseline: $name"
  python -m src.run_baseline --name "$name" $LIMIT_FLAG --batch-size 16 --num-workers 12
done

# ---------- SFT training ----------
step "5/9 SFT LoRA training (Qwen2.5-Coder-7B-Instruct on train split)"
python -m src.sft_train

SFT_ADAPTER=/home/ubuntu/vista/models/sft-qwen-coder-7b-malt/final
if [ ! -d "$SFT_ADAPTER" ]; then
  echo "WARNING: SFT adapter not at $SFT_ADAPTER — checking last checkpoint…"
  SFT_ADAPTER=$(ls -td /home/ubuntu/vista/models/sft-qwen-coder-7b-malt/checkpoint-* 2>/dev/null | head -1)
  if [ -z "$SFT_ADAPTER" ]; then
    echo "ERROR: no SFT adapter available, skipping LoRA-dependent baselines"
    SFT_ADAPTER=""
  fi
fi

if [ -n "$SFT_ADAPTER" ]; then
  step "6/9 baseline: sft_only (LoRA + base prompt) using $SFT_ADAPTER"
  python -m src.run_baseline --name sft_only --lora "$SFT_ADAPTER" $LIMIT_FLAG --batch-size 16 --num-workers 12
fi

# ---------- VISTA-Net++ on plain base ----------
step "7/9 VISTA-Net++ on base"
python -m src.run_vista $LIMIT_FLAG --chunk-size 8 --n-hyp 5 --max-repair 3 --num-workers 12

# ---------- VISTA-Net++ on SFT ----------
if [ -n "$SFT_ADAPTER" ]; then
  step "8/9 VISTA-Net++ on SFT adapter"
  python -m src.run_vista --lora "$SFT_ADAPTER" $LIMIT_FLAG --chunk-size 8 --n-hyp 5 --max-repair 3 --num-workers 12
fi

# ---------- Aggregation ----------
step "9/9 aggregate results"
python -m src.aggregate
echo "ALL DONE — see /home/ubuntu/vista/outputs/RESULTS.md"
