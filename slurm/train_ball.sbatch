#!/usr/bin/env bash
#SBATCH --job-name=vbcv-train-ball
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source slurm/common.sh
python scripts/train_ball_heatmap.py --data "${DATA:?set DATA=path to YOLO-format ball dataset}" --out "${OUT:-runs/train_ball}" --device cuda ${EXTRA:-}
