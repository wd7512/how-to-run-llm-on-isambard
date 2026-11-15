#!/bin/bash
# Submit the serve job with the node count the chosen model needs.  Usage: MODEL=glm53 bash scripts/03_serve.sh
set -eu
cd "$(dirname "$0")/.."
source scripts/config.sh
mkdir -p logs
sbatch --nodes="$NODES" --job-name="serve-$MODEL" scripts/03_serve.sbatch
