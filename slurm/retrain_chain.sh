#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=/blue/uf-sal/leodorfman
python scripts/merge_labels.py $ROOT/labels_merged/train.csv $ROOT/labels/burst_leo.csv $ROOT/labels/burst_trent.csv --exclude 141260:146040
EXP=$(sbatch --parsable --account=uf-sal --export=ALL,VIDEO=$ROOT/film/UFvsUKFull.mp4,LABELS=$ROOT/labels_merged/train.csv,OUT=$ROOT/uf_train_full,CHECK_DIR=$ROOT/film/frames_burst slurm/export_labels.sbatch)
TRN=$(DATA=$ROOT/data/ball_ds OUT=$ROOT/runs/train_ball_full EXTRA="--steps 6000 --temporal-gap 1 --init $ROOT/runs/train_ball_nat/last.pt --extra $ROOT/uf_train3 $ROOT/uf_train_full --extra-repeat 10 --extra-scale 2" sbatch --parsable --account=uf-sal --dependency=afterok:$EXP --time=02:00:00 slurm/train_ball.sbatch)
echo "export job $EXP, then train job $TRN (runs only if the export and its frame check pass)"
echo "watch: squeue -u $USER   logs: logs/export-$EXP.out  logs/vbcv-train-ball-$TRN.out"
