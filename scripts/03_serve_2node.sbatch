#!/bin/bash
# Serve the chosen model (MODEL=dsv41|glm53, default dsv41) on 2 nodes (8 x GH200), run a smoke test and benchmarks.
# Layout: tensor-parallel 4 inside each node x data-parallel 2 across nodes + expert parallel.
# Timeline: ~2.5 min weights, ~1 min kernels/graphs, then the benchmarks (~7-8 min total).
#
# Submit from the repo root:  sbatch scripts/03_serve_2node.sbatch
#SBATCH --job-name=serve-2node
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
set -u
source "$SLURM_SUBMIT_DIR/scripts/config.sh"

module load brics/nccl brics/apptainer-multi-node   # Slingshot RDMA env + /host/adapt.sh
export HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export REPO_DIR="$SLURM_SUBMIT_DIR"

# --kill-on-bad-exit: if the head node's server dies, tear the whole job down instead of idling.
srun --kill-on-bad-exit=1 --export=ALL \
  apptainer exec --nv --bind "$SCRATCHDIR" --bind "$REPO_DIR" "$SIF" \
  "$REPO_DIR/scripts/node_entry.sh"
