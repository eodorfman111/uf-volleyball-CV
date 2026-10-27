#!/usr/bin/env bash
set -euo pipefail
module load conda
ENV_DIR=${VBCV_ENV:-/blue/$(id -gn)/$USER/conda/envs/vbcv}
cd "$(dirname "$0")/.."
if [ ! -d "$ENV_DIR" ]; then
  conda env create -p "$ENV_DIR" -f environment.yml
fi
set +u
conda activate "$ENV_DIR"
set -u
pip install -r requirements-optional.txt
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
bash scripts/fetch_models.sh
python -c "import torch, onnxruntime; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('ort', onnxruntime.get_available_providers())"
