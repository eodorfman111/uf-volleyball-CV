#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:?usage: slurm/submit_match.sh configs/<match>.yaml}
cd "$(dirname "$0")/.."
a=$(CONFIG="$CONFIG" STAGES="ingest calibrate" sbatch --parsable slurm/stage.sbatch)
b=$(CONFIG="$CONFIG" sbatch --parsable --dependency=afterok:"$a" slurm/stage_gpu.sbatch)
c=$(CONFIG="$CONFIG" STAGES="track events stats" sbatch --parsable --dependency=afterok:"$b" slurm/stage.sbatch)
echo "ingest+calibrate $a -> detect $b -> track+events+stats $c"
