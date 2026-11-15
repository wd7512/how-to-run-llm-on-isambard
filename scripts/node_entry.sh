#!/bin/bash
# Runs INSIDE the container, once per node (launched by 03_serve_2node.sbatch).
# Node 0 = API server + benchmarks. Other nodes (NODES>1) = headless data-parallel workers.
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

# --- Fix 3: some models JIT-compile attention kernels on first start (FlashInfer, several
# minutes per rank). vLLM gives up after 600 s by default; the GLM-5.3 vLLM recipe uses 3600.
export VLLM_ENGINE_READY_TIMEOUT_S=3600

# --- Fix 4: layout comes from the model file. For the big MoE models it is tensor-parallel 4 per
# node x data-parallel across nodes + expert parallel; pipeline parallel across nodes fails in
# vLLM 0.33 for these models (AssertionError in _run_layers).
DP=$((NODES * DP_PER_NODE))
LAYOUT="--tensor-parallel-size $TP"
[ "$DP" -gt 1 ] && LAYOUT="$LAYOUT --data-parallel-size $DP --data-parallel-size-local $DP_PER_NODE \
  --data-parallel-address ${HEAD_NODE} --data-parallel-rpc-port 13345"
[ "$EP" = "1" ] && LAYOUT="$LAYOUT --enable-expert-parallel"
COMMON="$MODEL_DIR --served-model-name $SERVED_NAME $LAYOUT \
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
  # In the dsv41 run the worker waited forever and the job was cancelled by hand; in the glm53 run
  # the job ended by itself. scancel is NOT available inside the container, so this line only
  # prints the command to run.
  scancel "$SLURM_JOB_ID" 2>/dev/null || echo "scancel unavailable here - run: scancel $SLURM_JOB_ID"
else
  vllm serve $COMMON --headless --data-parallel-start-rank $((SLURM_NODEID * DP_PER_NODE)) \
    > "$REPO_DIR/logs/vllm_worker_$SLURM_JOB_ID.log" 2>&1
fi
