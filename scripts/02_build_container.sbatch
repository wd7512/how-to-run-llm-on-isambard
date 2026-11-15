#!/bin/bash
# Build the Apptainer image and check that vLLM knows the model architecture.
# Done in a job (not on the login node) because the build needs more than 4 GiB.
# Takes ~20 minutes; the .sif is ~9.4 GB and is reused afterwards.
#
# Submit from the repo root:  sbatch scripts/02_build_container.sbatch
#SBATCH --job-name=vllm-build
#SBATCH --output=logs/%x_%j.out
#SBATCH --gpus=1
#SBATCH --time=01:00:00
set -eu
source "$SLURM_SUBMIT_DIR/scripts/config.sh"

export APPTAINER_CACHEDIR="$SCRATCHDIR/tmp/apptainer-cache"
export APPTAINER_TMPDIR="$SCRATCHDIR/tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$(dirname "$SIF")"

echo "=== node ==="; uname -m
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

if [ ! -f "$SIF" ]; then
  echo "=== building $IMAGE_REF ==="; date
  apptainer build "$SIF" "$IMAGE_REF"
  date
fi
ls -la "$SIF"

echo "=== check ==="
apptainer exec --nv "$SIF" python3 - <<'PY'
import torch, vllm
print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda,
      torch.cuda.is_available(), torch.cuda.get_device_name(0))
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
import os; a = os.environ["ARCH"]
print(a, "supported:", a in archs)
PY
echo DONE
