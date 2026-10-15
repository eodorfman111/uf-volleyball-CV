module load conda
ENV_DIR=${VBCV_ENV:-/blue/$(id -gn)/$USER/conda/envs/vbcv}
conda activate "$ENV_DIR"
export HF_HOME=${HF_HOME:-/blue/$(id -gn)/$USER/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
