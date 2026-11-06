#!/usr/bin/env bash
set -euo pipefail
NAME=${1:?usage: slurm/match_ball.sh NAME /path/to/video.mp4}
VIDEO=${2:?usage: slurm/match_ball.sh NAME /path/to/video.mp4}
cd /blue/uf-sal/leodorfman/uf-volleyball-CV
module load conda
conda activate /blue/uf-sal/leodorfman/conda/envs/vbcv
export SBATCH_ACCOUNT=uf-sal
python scripts/make_config.py configs/ufuk_full.yaml "configs/$NAME.yaml" --set "name=$NAME" "video=$VIDEO"
a=$(CONFIG="configs/$NAME.yaml" STAGES="ingest" sbatch --parsable slurm/stage.sbatch)
b=$(CONFIG="configs/$NAME.yaml" sbatch --parsable --dependency=afterok:"$a" slurm/stage_gpu.sbatch)
echo "$NAME ball-only: ingest $a -> detect $b"
