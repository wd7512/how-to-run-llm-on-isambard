#!/bin/bash
# Submit the serve job with the nodes, GPUs and time cap the chosen model needs (billing is per GPU).
# Usage: MODEL=glm53 bash scripts/03_serve.sh          (set DEPENDENCY=afterok:<ids> to chain; prints the job id)
set -eu
cd "$(dirname "$0")/.."
source scripts/config.sh
mkdir -p logs
sbatch --parsable --nodes="$NODES" --gpus-per-node="$GPUS" --time="$TIME_LIMIT" \
       ${DEPENDENCY:+--dependency="$DEPENDENCY"} --job-name="serve-$MODEL" scripts/03_serve.sbatch
