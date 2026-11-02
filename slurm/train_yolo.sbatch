#!/usr/bin/env bash
#SBATCH --job-name=vbcv-train-yolo
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source slurm/common.sh
python scripts/train_ball_yolo.py --data "${DATA:?set DATA=path to an exported UF dataset with images/ and labels/}" --out "${OUT:-runs/train_ball_yolo}" ${EXTRA:-}
