#!/usr/bin/env bash
#SBATCH --job-name=vbcv-stage
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source slurm/common.sh
STAGE=${STAGE:?set STAGE=ingest|calibrate|detect|track|events|stats|all}
CONFIG=${CONFIG:?set CONFIG=configs/xxx.yaml}
python -m vbcv "$STAGE" -c "$CONFIG" --set detect.players.device=cuda ${EXTRA:-}
