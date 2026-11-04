#!/bin/bash
#SBATCH --job-name=scorewin
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source slurm/common.sh
python scripts/score_windows.py "${LABELS:?set LABELS}" "${WINDOWS:?set WINDOWS}" ${CKPTS:?set CKPTS} --out "${OUT:-runs/score_windows}" ${EXTRA:-}
