#!/bin/bash
# Runs INSIDE the container, once per node (launched by 03_serve.sbatch).
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

# Per-model environment variables from the model's file (e.g. EXTRA_ENV="VLLM_FLOAT32_MATMUL_PRECISION=high").
[ -n "${EXTRA_ENV:-}" ] && export $EXTRA_ENV

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

# The exact commands this run uses; recorded in meta.json so the guide shows what was really executed.
export SERVE_CMD="vllm serve $COMMON --port 8000"
export WORKER_CMD="vllm serve $COMMON --headless --data-parallel-start-rank <node-rank x DP_PER_NODE>"
RUN_DIR="$REPO_DIR/logs/run_$SLURM_JOB_ID"      # machine-readable output of this run, read by collect_results.py
mkdir -p "$RUN_DIR"
echo "node $SLURM_NODEID host $(hostname) ip $VLLM_HOST_IP head $HEAD_NODE"

if [ "$SLURM_NODEID" = "0" ]; then
  vllm serve $COMMON --port 8000 > "$REPO_DIR/logs/vllm_serve_$SLURM_JOB_ID.log" 2>&1 &
  SP=$!
  # Wait for the server. Stop at once if it crashed, and give up at STARTUP_DEADLINE_S (a few times the measured
  # start-up time) instead of idling on every GPU until the job's time cap.
  DEADLINE=${STARTUP_DEADLINE_S:-1200}; T0=$SECONDS
  until curl -sf localhost:8000/health >/dev/null 2>&1; do
    kill -0 $SP 2>/dev/null || { echo "SERVER DIED"; touch "$RUN_DIR/DONE"; exit 1; }
    if [ $((SECONDS - T0)) -ge "$DEADLINE" ]; then
      echo "STARTUP DEADLINE (${DEADLINE}s) EXCEEDED - aborting to save GPU-hours"
      touch "$RUN_DIR/DONE"; pkill -f "vllm serve"; exit 1
    fi
    sleep 5
  done
  export STARTUP_S=$((SECONDS - T0))
  echo "=== server up $(date) (start-up ${STARTUP_S}s) ==="
  export CONFIG_HASH=$(python3 "$REPO_DIR/scripts/confighash.py" "$MODEL")

  python3 - > "$RUN_DIR/meta.json" <<PY
import json, os, re, vllm, torch
def clean(t):  # drop machine-specific names so the recorded command is portable
    t = t.replace(os.environ["MODEL_DIR"], "<model-dir>").replace(os.environ["HEAD_NODE"], "<head-node>")
    return " ".join(t.split())
ENV = ["HF_HUB_OFFLINE", "TRITON_LIBCUDA_PATH", "VLLM_ENGINE_READY_TIMEOUT_S", "DG_JIT_CACHE_DIR", "TRITON_CACHE_DIR",
       "VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR", "XDG_CACHE_HOME"]
env = {kv.split("=")[0]: kv.split("=", 1)[1] for kv in os.environ.get("EXTRA_ENV", "").split()}
env.update({k: re.sub(r"/local/user/\\d+/cache_\\d+", "/local/<node-local-cache>", os.environ[k]) for k in ENV if k in os.environ})
print(json.dumps({"serve_command": clean(os.environ["SERVE_CMD"]), "worker_command": clean(os.environ["WORKER_CMD"]) if int(os.environ["NODES"]) > 1 else None, "env": env,"config_hash": os.environ.get("CONFIG_HASH"), "startup_s": int(os.environ.get("STARTUP_S", 0)), "model": os.environ["MODEL"], "repo": os.environ["MODEL_REPO"], "revision": os.environ["MODEL_REVISION"],
  "served_name": os.environ["SERVED_NAME"], "vllm": vllm.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
  "image": os.environ["IMAGE_REF"], "job_id": os.environ["SLURM_JOB_ID"], "nodes": int(os.environ["NODES"]),
  "gpus_per_node": int(os.environ["GPUS"]), "tp": int(os.environ["TP"]), "dp_per_node": int(os.environ["DP_PER_NODE"]),
  "ep": os.environ["EP"] == "1", "max_model_len": int(os.environ["MAX_LEN"]), "extra_args": os.environ["VLLM_EXTRA_ARGS"]}, indent=2))
PY

  curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model":"'$SERVED_NAME'","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64}' \
    | tee "$RUN_DIR/smoke.json"
  echo
  # Base models have no chat template: the chat call errors, so also try plain completions (informational).
  grep -q '"error"' "$RUN_DIR/smoke.json" && curl -s localhost:8000/v1/completions -H "Content-Type: application/json" \
    -d '{"model":"'$SERVED_NAME'","prompt":"Hello, my name is","max_tokens":32}' | tee "$RUN_DIR/smoke_completions.json" && echo

  # The benchmark. Fixed workload: random prompts of exactly 1024 input tokens, 256 output tokens, seed 0.
  # Results are saved as JSON (cN.json) with the measured token counts, not scraped from text.
  for C in 1 16 64; do
    echo "=== bench concurrency $C ==="
    vllm bench serve --model $SERVED_NAME --tokenizer "$MODEL_DIR" --port 8000 \
      --dataset-name random --random-input-len 1024 --random-output-len 256 --seed 0 \
      --num-prompts $((C*4)) --max-concurrency $C \
      --save-result --result-dir "$RUN_DIR" --result-filename "c$C.json" 2>&1 \
      | grep -E "Output token throughput|Mean TPOT|Mean TTFT|Successful|Failed"
  done

  # Tell the other nodes to stop, then stop our server. (scancel does not exist inside the container.)
  touch "$RUN_DIR/DONE"
  pkill -f "vllm serve"
else
  vllm serve $COMMON --headless --data-parallel-start-rank $((SLURM_NODEID * DP_PER_NODE)) \
    > "$REPO_DIR/logs/vllm_worker_$SLURM_JOB_ID.log" 2>&1 &
  WP=$!
  # Exit when the head node signals DONE (or the worker dies), so the job ends instead of idling.
  while kill -0 $WP 2>/dev/null && [ ! -f "$RUN_DIR/DONE" ]; do sleep 5; done
  pkill -f "vllm serve" || true
fi
