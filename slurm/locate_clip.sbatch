#!/bin/bash
#SBATCH --job-name=locate
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source slurm/common.sh
FULL=${FULL:-/blue/uf-sal/leodorfman/film/UFvsUKFull.mp4}
CLIP=${CLIP:-/blue/uf-sal/leodorfman/film/UFvsUKSnippet.mp4}
python scripts/locate_clip.py "$FULL" "$CLIP"
