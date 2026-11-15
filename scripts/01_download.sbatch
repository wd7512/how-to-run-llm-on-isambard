#!/bin/bash
# Download the checkpoint (MODEL=dsv41: 510 GB; glm53: 328 GB). Must be a Slurm job: login nodes are limited to
# ~1 core / 4 GiB and the OOM killer ends large downloads (exit 137).
# Takes ~10 minutes (about 1 GB/s). Resumable: just resubmit if it is interrupted.
#
# Submit from the repo root:  mkdir -p logs && sbatch scripts/01_download.sbatch
#SBATCH --job-name=download
#SBATCH --output=logs/%x_%j.out
#SBATCH --gpus=1            # no CPU-only partition exists; 1 GPU = 1 superchip
#SBATCH --time=01:00:00
set -eu
source "$SLURM_SUBMIT_DIR/scripts/config.sh"
module load cray-python/3.11.7

mkdir -p "$MODEL_DIR"
"$HF_VENV/bin/hf" download "$MODEL_REPO" --local-dir "$MODEL_DIR" --max-workers 16

# Remove fragments left by any earlier interrupted attempts.
rm -f "$MODEL_DIR"/.cache/huggingface/download/*.incomplete
echo "DONE"; du -sh "$MODEL_DIR"
