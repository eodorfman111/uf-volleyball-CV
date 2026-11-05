#!/usr/bin/env bash
cd /blue/uf-sal/leodorfman/uf-volleyball-CV
module load conda
conda activate /blue/uf-sal/leodorfman/conda/envs/vbcv
export SBATCH_ACCOUNT=uf-sal
python scripts/make_config.py configs/ufuk_snippet.yaml configs/ufuk_full.yaml --set name=ufuk_full video=/blue/uf-sal/leodorfman/film/UFvsUKFull.mp4 ingest.start_s=0 ingest.end_s=null detect.ball.backend=heatmap detect.ball.model=/blue/uf-sal/leodorfman/runs/train_ball_full/last.pt "detect.ball.input_size=[1280,720]" detect.players.backend=none
a=$(CONFIG=configs/ufuk_full.yaml STAGES="ingest calibrate" sbatch --parsable slurm/stage.sbatch)
b=$(CONFIG=configs/ufuk_full.yaml sbatch --parsable --dependency=afterok:$a slurm/stage_gpu.sbatch)
echo "full game ball-only: ingest+calibrate $a -> detect $b"
