#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source slurm/common.sh
VIDEO=${VIDEO:-/blue/uf-sal/leodorfman/film/UFvsUKFull.mp4}
OUT=${OUT:-/blue/uf-sal/leodorfman/film/frames_full}
STEP=${STEP:-558}
COUNT=${COUNT:-401}
mkdir -p "$OUT"
run() { ffmpeg -nostdin -y -threads 4 -i "$VIDEO" -vf "select='not(mod(n,$STEP))'" "$@" -q:v 2 -frames:v "$COUNT" "$OUT/%05d.jpg"; }
run -fps_mode passthrough || run -vsync 0
chmod -R g+rX "$OUT"
