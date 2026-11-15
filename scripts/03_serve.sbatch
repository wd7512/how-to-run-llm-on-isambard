#!/bin/bash
# Serve the chosen model on NODES nodes (set per model), run a smoke test and benchmarks.
# Layout (TP / DP / expert-parallel) comes from scripts/models/<MODEL>.env, see node_entry.sh.
# Timeline for the big MoE models: ~2.5 min weights, ~1-4 min startup, then benchmarks.
#
# Submit with the wrapper (it sets --nodes from the model file):  bash scripts/03_serve.sh
#SBATCH --job-name=serve
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=2          # nodes, GPUs and CPUs below are overridden by 03_serve.sh
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
