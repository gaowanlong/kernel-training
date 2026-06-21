#!/bin/bash
# v1.1 Full Pipeline: Train → Evaluate → Upload
# Run: bash scripts/pipeline_v11.sh
# This script runs everything sequentially without interruption.

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

LOG="pipeline_v11.log"
echo "[$(date)] v1.1 Pipeline started" | tee -a "$LOG"

# Step 1: Wait for training to complete (if already running)
echo "" | tee -a "$LOG"
echo "=== Step 1: Training ===" | tee -a "$LOG"
if ps aux | grep -q "[t]rain_lora_v11"; then
    echo "Training already running, waiting for it to finish..." | tee -a "$LOG"
    while ps aux | grep -q "[t]rain_lora_v11"; do
        sleep 60
    done
    echo "Training completed!" | tee -a "$LOG"
else
    echo "Starting training..." | tee -a "$LOG"
    python scripts/train_lora_v11.py \
        --model models/qwen2.5-7b \
        --data data/processed \
        --output lora_adapters/kernel-lora-v1.1 \
        --iters 300 --rank 16 --lr 2e-5 2>&1 | tee -a "$LOG"
fi

# Step 2: Evaluate
echo "" | tee -a "$LOG"
echo "=== Step 2: Evaluation ===" | tee -a "$LOG"
python scripts/evaluate.py \
    --model models/qwen2.5-7b \
    --adapter lora_adapters/kernel-lora-v1.1 \
    --output results 2>&1 | tee -a "$LOG"

# Step 3: Find latest eval report
LATEST_REPORT=$(ls -t results/eval_report_*.json | head -1)
LATEST_SUMMARY=$(ls -t results/eval_summary_*.txt | head -1)
echo "Latest report: $LATEST_REPORT" | tee -a "$LOG"
cat "$LATEST_SUMMARY" | tee -a "$LOG"

# Step 4: Fuse and upload to HuggingFace
echo "" | tee -a "$LOG"
echo "=== Step 3: Upload to HuggingFace ===" | tee -a "$LOG"
python -m mlx_lm fuse \
    --model models/qwen2.5-7b \
    --adapter-path lora_adapters/kernel-lora-v1.1 \
    --save-path models/qwen2.5-7b-fused-v1.1 \
    --upload-repo gaowanlong/kernel-lora-v1.1 2>&1 | tee -a "$LOG"

# Step 5: Git commit and push
echo "" | tee -a "$LOG"
echo "=== Step 4: Upload to GitHub ===" | tee -a "$LOG"
git add CHANGELOG.md results/ scripts/train_lora_v11.py scripts/build_multiturn_data.py
git commit -m "v1.1: Multi-turn conversation + curriculum learning" || true
git tag -a v1.1 -m "v1.1: Multi-turn conversation + curriculum learning"
git push origin main
git push origin v1.1

# Step 6: Create GitHub Release
echo "" | tee -a "$LOG"
echo "=== Step 5: GitHub Release ===" | tee -a "$LOG"
gh release create v1.1 \
    --title "v1.1 — Multi-Turn Conversation + Curriculum Learning" \
    --notes "See CHANGELOG.md for details" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[$(date)] v1.1 Pipeline COMPLETE!" | tee -a "$LOG"
