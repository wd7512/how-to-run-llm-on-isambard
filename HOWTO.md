# Running a large LLM on Isambard-AI (2 nodes, vLLM)

Step-by-step, with the scripts in [`scripts/`](scripts/). Pick a model with `MODEL=dsv41` (default) or
`MODEL=glm53`; measured results are on the [stats page](https://wd7512.github.io/how-to-run-llm-on-isambard/).

> **Status.** `glm53` was run end to end from a fresh clone. For `dsv41` the numbers come from the
> original absolute-path scripts; the current model-switch version has not been re-run for it.

## What you get

| | `dsv41` | `glm53` |
|---|---|---|
| Model | DeepSeek-V4.1-Flash-1017, 552B MoE | GLM-5.3-Flash, MoE, 288 experts |
| Weights | 510 GB, FP8 + FP4 | 328 GB, FP8 |
| Layout | TP 4 x DP 2 + expert-parallel on 2 nodes (8 x GH200) | same, plus `--enforce-eager` |
| Output tok/s, 1 / 16 / 64 users | 47 / 491 / 1,422 | 6.9 / 92 / 365 |
| GPU-hours to reproduce | ~1.5 | ~2.3 |

Software: vLLM 0.33.0, torch 2.14, CUDA 13.0, Apptainer, driver 580.178.04. After setup, serving on 2
nodes costs 8 GPU-hours per hour the job runs.

## Prerequisites

- An Isambard-AI Phase 2 account and project, and an SSH certificate from `clifton auth`.
- ~550 GB free in `$SCRATCHDIR` per model (the default quota is 5 TiB).
- GPU allocation: ~0.4 GPU-hours for download + build, then 1.0 (`dsv41`) to 2.0 (`glm53`) GPU-hours for
  the serve-and-benchmark run. Total to reproduce: 1.5 to 2.3 GPU-hours. Check your project's balance first.

## Steps

```bash
ssh <project>.aip2.isambard
git clone https://github.com/wd7512/how-to-run-llm-on-isambard.git
cd how-to-run-llm-on-isambard     # keep submitting from here: logs/ is relative to it
export MODEL=glm53                # or dsv41 (default); sbatch passes it to the job
```

### 0. One-off login-node setup (~1 minute)

```bash
bash scripts/00_setup_login_node.sh
```

Makes a small Python venv with `huggingface_hub` and prints the repo's file count and size.
Only the `pip install` runs here; **do not download the weights on the login node**
(see [Troubleshooting](#troubleshooting), item 1).

### 1. Download the weights (6-10 minutes, 1 GPU)

```bash
sbatch scripts/01_download.sbatch
```

Runs at about 1 GB/s. Resumable: resubmit if it is interrupted. Success looks like `DONE` and a
`du` size close to the repo size (476 GiB for `dsv41`, 306 GiB for `glm53`) in `$SCRATCHDIR/models/<repo name>`.

### 2. Build the container (~20 minutes, 1 GPU)

```bash
sbatch scripts/02_build_container.sbatch
```

Pulls the pinned arm64 vLLM image (~11 GB) into a 9.4 GB `.sif`. The log must end with (the class name is your model's `ARCH`):

```
vllm 0.33.0 torch 2.14.0+cu130 cuda 13.0 True NVIDIA GH200 120GB
Glm5NextForConditionalGeneration supported: True
```

Notes: Isambard-AI is aarch64, so only `linux/arm64` images work. The log is full of
`xattr ... ENOTSUP` warnings from Lustre; they are harmless. Build in a job, not on the login node.

### 3. Serve and benchmark on 2 nodes (8-15 minutes, 8 GPUs)

```bash
sbatch scripts/03_serve_2node.sbatch
tail -f logs/serve-2node_<jobid>.out
```

Approximate timeline of a healthy `dsv41` run (figures from the logs; clock times rounded). `glm53`
loads its weights in ~45 s, reports 42 GiB per GPU and a 3.46M-token KV cache, and is ready after ~4 minutes:

| Time | Log line |
|---|---|
| ~0:30 | `Resolved architecture: DeepseekV41ForCausalLM`, Engram tables offloaded to pinned host RAM (23.6 GiB per rank) |
| ~2:30 | `Model loading took 39.01 GiB memory and ~128 seconds` |
| ~4:30 | `Available KV cache memory: 40.8 GiB`, `GPU KV cache size: 12,724,074 tokens`, `Maximum concurrency for 262,144 tokens per request: 48.54x` |
| ~5:00 | `Application startup complete`, then `=== server up ===` and a hello-world reply |
| ~5:00-7:30 | benchmarks at concurrency 1, 16, 64 |

The script then stops the server. `scancel` is not available inside the container. On `glm53` the job
ended by itself; on `dsv41` the worker waited, so run `scancel <jobid>` if the job is still going.

To keep the endpoint up instead, remove the `pkill`/`scancel` lines at the end of
`scripts/node_entry.sh` and `wait` on the server. The API is OpenAI-compatible on port 8000
of the first node (`$HEAD_NODE`); tunnel to it with `ssh -L`.

## Why this configuration

- **2 nodes are the minimum.** Weights are 510 GB; one node has ~384 GB of HBM.
- **DP 2 x TP 4 + expert parallel**, not pipeline parallel. The model's own code guards for
  exactly this multi-node layout (pipeline size 1, expert parallel on, TP > 1, DP > 1).
  MoE experts are sharded across all 8 GPUs, attention weights are replicated per node,
  and the per-layer tensor-parallel all-reduce stays inside one node rather than crossing the network.
- **Per-model flags** (parsers, eager mode) live in `scripts/models/<name>.env`.
- **`--max-model-len 262144`** because 250k context was the target; the KV cache holds 48
  such sequences at once, so memory is not the limit. The model itself supports 1M tokens.
- **Engram offload**: vLLM puts the model's ~196B Engram embedding tables in pinned host RAM
  by itself (23.6 GiB per rank, ~94 GiB per node; vLLM reported ~340 GiB of RAM available per node).

## Troubleshooting

The scripts already include these fixes. If you change the setup, these are the errors to expect. Symptom -> cause -> fix.

1. **Download dies with exit 137 about 20 seconds in.** The login node allows about 1 core and
   4 GiB; the OOM killer ends big transfers. Also, background processes there are reaped when
   your SSH session ends, even with `setsid`/`nohup`. -> Download in a Slurm job (`01_download.sbatch`).
2. **`ray: command not found` / "ray GPUs: 0".** The vLLM image does not contain Ray. -> Use
   vLLM's built-in multi-node mode (`--data-parallel-*`, `--headless`), no Ray needed.
3. **`/usr/bin/ld: cannot find -l:libcuda.so.1` / `cannot find /lib64/libc.so.6`** during the
   first forward pass, from Triton compiling `cuda_utils`. Triton finds libcuda via `ldconfig`,
   which inside the container resolves to a host directory holding a glibc linker script. ->
   `export TRITON_LIBCUDA_PATH=/.singularity.d/libs` (where `apptainer --nv` mounts libcuda).
   A first attempt with a symlink to `ldconfig`'s answer failed because `ldconfig -p` listed nothing.
4. **`AssertionError` at `assert residual is not None and post_mix is not None`** (in
   `vllm/models/deepseek_v41/nvidia/model.py`, `_run_layers`) on the second node when using
   `--pipeline-parallel-size 2`. The non-first pipeline stage does not restore the
   Hyper-Connection state. -> Do not use pipeline parallel for this model in vLLM 0.33;
   use data-parallel + expert-parallel.
5. **`RuntimeError: ... missing CUDA CUBIN: ~/.cache/vllm/deep_gemm/...`.** Eight ranks compile
   and read the same DeepGEMM kernel cache on shared Lustre; one reads the file before another
   has finished writing it. -> `DG_JIT_CACHE_DIR` on node-local `/local`.
6. **`OSError: [Errno 116] Stale file handle`** from Triton's cache, same class of race. ->
   `TRITON_CACHE_DIR`, `VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`, `XDG_CACHE_HOME` all on `/local`.
   `node_entry.sh` sets all five in one place.
7. **A crashed server keeps burning allocation.** A script that waits for `/health` forever holds
   both nodes after the server has died. -> the wait loop checks `kill -0 $SP`, and
   `srun --kill-on-bad-exit=1` tears the job down. Keep both if you edit the scripts.

8. **`TimeoutError: Timed out waiting for engine core processes to start ... 600s`** after the weights
   have loaded (`glm53`). FlashInfer JIT-compiles an attention kernel on first use and vLLM gives up
   after 600 s. -> `VLLM_ENGINE_READY_TIMEOUT_S=3600` (already set in `node_entry.sh`; the vLLM recipe
   for the model uses the same value).
9. **Hangs at `Capturing CUDA graphs (PIECEWISE) 12/19` with no error** (`glm53`, DP 2 x TP 4 + EP).
   No progress for 22 minutes, so the run was cancelled. -> `--enforce-eager` (in `glm53.env`). It works
   but decode is about 7x slower than `dsv41`. Open item: find a graph mode that does not hang, or try
   TP 8 across both nodes as the vLLM recipe does on one 8-GPU node.

Tip: when something fails, the useful error is rarely at the bottom of the log. Search
`vllm_serve_<jobid>.log` and `vllm_worker_<jobid>.log` for the first `ERROR`, not the last.

## Known issues and caveats

- **Stray `</think>`** at the start of `dsv41` smoke-test replies; not investigated. `glm53` runs with
  `--reasoning-parser glm47` and its reply is clean, so a reasoning parser is the first thing to try for `dsv41`.
- **`glm53` numbers are eager-mode.** They are a floor, not what the hardware can do.
- **Short benchmarks only.** The numbers use 1,024 input / 256 output tokens and 4 x concurrency
  requests per level. They say nothing about 250k-token prompts (prefill time, long-context decode).
- **Single run.** One working run, no repeats, so no error bars.
- **Node-local caches mean kernels recompile every job** (~1 minute of start-up). A warm cache
  would need a per-node, per-job copy, not a shared directory.
- **`latest` image moves.** Use the digest in `scripts/config.sh`.
- **Driver and image.** The cluster driver (580.178.04) was new enough for CUDA 13 images. If
  yours is older, use a CUDA 12 image tag.

## Checking the cost yourself

```bash
sacct -u $USER -S <date> -X -P -o JobID,JobName,State,NNodes,AllocTRES,ElapsedRaw
```

GPU-hours = GPUs in `AllocTRES` x `ElapsedRaw` / 3600. One node = 4 GPUs. See
[`results/gpu_hours.json`](results/gpu_hours.json) for the per-job breakdown.
