#!/bin/bash
# Runs INSIDE the container, once per node (launched by 03_serve_2node.sbatch).
# Node 0 = API server + benchmarks. Node 1 = headless worker (data-parallel rank 1).
set -u
source "$REPO_DIR/scripts/config.sh"
source /host/adapt.sh              # expose host MPI/NCCL (Slingshot) inside the container

export HF_HUB_OFFLINE=1
export VLLM_HOST_IP=$(hostname -i | awk '{print $1}')

# --- Fix 1: Triton JIT-compiles a helper with gcc; its default libcuda lookup finds a
# host directory with a glibc linker script and the link fails. Point it at the
# nvidia libs that apptainer --nv mounts.
export TRITON_LIBCUDA_PATH=/.singularity.d/libs

# --- Fix 2: all runtime kernel caches go to NODE-LOCAL storage. The defaults live in
# $HOME on Lustre; 4 ranks per node race on them ("missing CUDA CUBIN",
# "Stale file handle"). /local is RAM-backed and gone when the job ends.
LC=/local/user/$(id -u)/cache_${SLURM_JOB_ID}
export DG_JIT_CACHE_DIR=$LC/deep_gemm TRITON_CACHE_DIR=$LC/triton \
       VLLM_CACHE_ROOT=$LC/vllm TORCHINDUCTOR_CACHE_DIR=$LC/inductor XDG_CACHE_HOME=$LC/xdg
mkdir -p "$DG_JIT_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" \
         "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"

# --- Fix 3: tensor-parallel 4 x data-parallel 2 + expert parallel. Pipeline parallel
# across nodes fails in vLLM 0.33 for this model (AssertionError in _run_layers).
COMMON="$MODEL_DIR --served-model-name $SERVED_NAME \
  --tensor-parallel-size 4 --data-parallel-size 2 --data-parallel-size-local 1 \
  --data-parallel-address ${HEAD_NODE} --data-parallel-rpc-port 13345 \
  --enable-expert-parallel \
  --max-model-len $MAX_LEN --max-num-seqs 64 --gpu-memory-utilization 0.90 $VLLM_EXTRA_ARGS"

mkdir -p "$REPO_DIR/logs"
echo "node $SLURM_NODEID host $(hostname) ip $VLLM_HOST_IP head $HEAD_NODE"

if [ "$SLURM_NODEID" = "0" ]; then
  vllm serve $COMMON --port 8000 > "$REPO_DIR/logs/vllm_serve_$SLURM_JOB_ID.log" 2>&1 &
  SP=$!
  # Wait for the server, but stop at once if it crashed (otherwise the job idles).
  for i in $(seq 1 240); do
    curl -sf localhost:8000/health >/dev/null 2>&1 && break
    kill -0 $SP 2>/dev/null || { echo "SERVER DIED"; exit 1; }
    sleep 10
  done
  echo "=== server up $(date) ==="

  curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model":"'$SERVED_NAME'","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64}'
  echo

  for C in 1 16 64; do
    echo "=== bench concurrency $C ==="
    vllm bench serve --model $SERVED_NAME --tokenizer "$MODEL_DIR" --port 8000 \
      --dataset-name random --random-input-len 1024 --random-output-len 256 \
      --num-prompts $((C*4)) --max-concurrency $C 2>&1 \
      | grep -E "Output token throughput|Mean TPOT|Mean TTFT|Successful|Failed"
  done

  # To keep the endpoint up instead, delete the next two lines and `wait $SP`.
  pkill -f "vllm serve"
  # The worker node waits forever once the head node is done. In the original run this was
  # cancelled by hand (`scancel <jobid>`); this line tries to do it automatically but is
  # UNTESTED (scancel may not exist inside the container).
  scancel "$SLURM_JOB_ID" 2>/dev/null || echo "scancel unavailable here - run: scancel $SLURM_JOB_ID"
else
  vllm serve $COMMON --headless --data-parallel-start-rank 1 \
    > "$REPO_DIR/logs/vllm_worker_$SLURM_JOB_ID.log" 2>&1
fi
