#!/usr/bin/env bash
#SBATCH --job-name=vbcv-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source slurm/common.sh
STAGES=${STAGES:?set STAGES="ingest calibrate" or "track events stats"}
CONFIG=${CONFIG:?set CONFIG=configs/xxx.yaml}
python -m vbcv $STAGES -c "$CONFIG" ${EXTRA:-}
