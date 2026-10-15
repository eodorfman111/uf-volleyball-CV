#!/usr/bin/env bash
#SBATCH --job-name=vbcv-matches
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --array=1-1
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
source slurm/common.sh
LIST=${LIST:?set LIST=file with one config path per line}
CONFIG=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
python -m vbcv all -c "$CONFIG" --continue-on-fail --set detect.players.device=cuda ${EXTRA:-}
