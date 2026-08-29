#!/bin/bash
set -e
cd /home/ubuntu/vista

echo "[$(date)] Waiting for vista_sft to finish..."
while pgrep -f "run_all_gen.*vista_sft" > /dev/null 2>&1; do
    sleep 30
done
echo "[$(date)] vista_sft done. Starting vista_base..."

python -m src.run_all_gen \
  --baselines vista_base \
  --vista-chunk-size 8 --n-hyp 5 --max-repair 3 \
  > logs/vista_base_full.out 2>&1

echo "[$(date)] vista_base done. Scoring all..."
python -m src.score_all --num-workers 12 --names vista_sft vista_base \
  > logs/score_vista.out 2>&1

echo "[$(date)] Scoring done. Aggregating..."
python -m src.aggregate > logs/aggregate.out 2>&1
echo "[$(date)] All done!"
