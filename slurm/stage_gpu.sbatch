#!/usr/bin/env bash
#SBATCH --job-name=vbcv-detect
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source slurm/common.sh
CONFIG=${CONFIG:?set CONFIG=configs/xxx.yaml}
python -c "import onnxruntime, torch; print('ort', onnxruntime.get_available_providers(), 'torch cuda', torch.cuda.is_available())"
python -m vbcv detect -c "$CONFIG" --set detect.players.device=cuda detect.ball.device=cuda ${EXTRA:-}
