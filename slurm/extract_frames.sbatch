#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/common.sh"
VIDEO=${VIDEO:-/blue/uf-sal/leodorfman/film/UFvsUKFull.mp4}
OUT=${OUT:-/blue/uf-sal/leodorfman/film/frames_full}
STEP=${STEP:-558}
COUNT=${COUNT:-401}
mkdir -p "$OUT"
ffmpeg -y -threads 4 -i "$VIDEO" -vf "select='not(mod(n,$STEP))'" -vsync 0 -q:v 2 -frames:v "$COUNT" "$OUT/%05d.jpg"
chmod -R g+rX "$OUT"
