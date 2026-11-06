#!/usr/bin/env bash
#SBATCH --job-name=rallycls
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source slurm/common.sh
python scripts/rally_ball_classifier.py "${CONFIG:-configs/ufuk_full.yaml}" "${DVW:-/blue/uf-sal/leodorfman/dvw/ufuk}" --baseline ${EXTRA:-}
