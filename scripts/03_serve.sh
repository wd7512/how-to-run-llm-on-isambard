#!/bin/bash
# Submit the serve job with the nodes and GPUs the chosen model needs (billing is per GPU).
# Usage: MODEL=glm53 bash scripts/03_serve.sh
set -eu
cd "$(dirname "$0")/.."
source scripts/config.sh
mkdir -p logs
sbatch --nodes="$NODES" --gpus-per-node="$GPUS" --job-name="serve-$MODEL" scripts/03_serve.sbatch
