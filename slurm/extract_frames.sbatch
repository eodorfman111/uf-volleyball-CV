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
if ffmpeg -hide_banner -h full 2>&1 | grep -q -- "-fps_mode"; then PASS="-fps_mode passthrough"; else PASS="-vsync 0"; fi
ffmpeg -nostdin -y -threads 4 -i "$VIDEO" -vf "select='not(mod(n,$STEP))'" $PASS -q:v 2 -frames:v "$COUNT" "$OUT/%05d.jpg"
chmod -R g+rX "$OUT"
