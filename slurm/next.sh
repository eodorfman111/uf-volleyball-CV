#!/usr/bin/env bash
cd /blue/uf-sal/leodorfman/uf-volleyball-CV
squeue -u "$USER" -o "%.10i %.10j %.2t %.9M" | head -6
if [ ! -s runs/ufuk_full/detect/ball.csv ]; then echo "ball.csv not ready yet"; ls -la runs/ufuk_full/detect 2>/dev/null; exit 0; fi
module load conda
conda activate /blue/uf-sal/leodorfman/conda/envs/vbcv
python scripts/rally_ball_classifier.py configs/ufuk_full.yaml /blue/uf-sal/leodorfman/dvw/ufuk --baseline
