#!/bin/bash
#SBATCH --job-name=export
#SBATCH --cpus-per-task=2
#SBATCH --mem=24gb
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source slurm/common.sh
python scripts/export_ball_labels.py "${VIDEO:?set VIDEO}" "${LABELS:?set LABELS}" "${OUT:?set OUT}" --gap "${GAP:-1}" ${EXTRA:-}
if [ -n "${CHECK_DIR:-}" ]; then python scripts/check_export.py "$LABELS" "$OUT" "$CHECK_DIR"; fi
