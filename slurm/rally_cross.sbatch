#!/usr/bin/env bash
#SBATCH --job-name=crossmatch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source slurm/common.sh
python scripts/rally_cross_match.py runs/ufuk_full/detect/ball.csv /blue/uf-sal/leodorfman/dvw/ufuk "runs/${TEST:?set TEST=run name}/detect/ball.csv" "/blue/uf-sal/leodorfman/dvw/${DVWNAME:?set DVWNAME=folder name}" ${EXTRA:-}
