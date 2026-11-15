# How we ran DeepSeek-V4.1-Flash-1017 on Isambard-AI (2 nodes, vLLM)

Step-by-step, with the scripts in [`scripts/`](scripts/) and the errors you may hit.
Measured results are on the [stats page](https://wd7512.github.io/how-to-run-llm-on-isambard/).

> **Status of these scripts.** They are the scripts that produced the results, with paths
> parameterised (`$SCRATCHDIR`, `$SLURM_SUBMIT_DIR`) so no account or project names are
> hard-coded. The container is now pinned by digest, and `node_entry.sh` gained an
> automatic `scancel` at the end. **That parameterised version has not been re-run
> end-to-end**; the original ran with absolute paths and the `latest` tag. If something
> differs, compare with the fixes in [Troubleshooting](#troubleshooting).

## What you get

| | |
|---|---|
| Model | `deepseek-ai/DeepSeek-V4.1-Flash-1017` (552B-parameter MoE, ~16B active when decoding, 1M-token context, MIT) |
| Weights | 510.3 GB, 84 files (FP8 with FP4 experts) |
| Hardware | Isambard-AI Phase 2: 2 nodes x 4 GH200 (96 GiB HBM each) = 8 GPUs |
| Software | vLLM 0.33.0, torch 2.14, CUDA 13.0, Apptainer, driver 580.178.04 |
| Layout | tensor-parallel 4 x data-parallel 2 + expert-parallel |
| Result | ~47 tok/s for one user, ~1,420 tok/s aggregate at 64 users |
| Cost | ~1.5 GPU-hours to reproduce (download 0.16 + build 0.35 + serve run 1.02); 8 GPU-hours per hour of serving afterwards |

## Prerequisites

- An Isambard-AI Phase 2 account and project, and an SSH certificate from `clifton auth`.
- ~600 GB free in `$SCRATCHDIR` (the default quota is 5 TiB).
- GPU allocation: about 0.5 GPU-hours once for download + build, then ~1 GPU-hour (8 GPUs for ~8 minutes)
  per serve-and-benchmark run. Total to reproduce: ~1.5 GPU-hours. Check your project's balance first.

## Steps

```bash
ssh <project>.aip2.isambard
git clone https://github.com/wd7512/how-to-run-llm-on-isambard.git
cd how-to-run-llm-on-isambard     # keep submitting from here: logs/ is relative to it
```

### 0. One-off login-node setup (~1 minute)

```bash
bash scripts/00_setup_login_node.sh
```

Makes a small Python venv with `huggingface_hub` and confirms the repo is 84 files / 510.3 GB.
Only the `pip install` runs here; **do not download the weights on the login node**
(see [Troubleshooting](#troubleshooting), item 1).

### 1. Download the weights (~10 minutes, 1 GPU)

```bash
sbatch scripts/01_download.sbatch
```

Runs at about 1 GB/s. Resumable: resubmit if it is interrupted. Success looks like
`DONE` and ~476 GiB (= 510.3 GB) in `$SCRATCHDIR/models/DeepSeek-V4.1-Flash-1017`.

### 2. Build the container (~20 minutes, 1 GPU)

```bash
sbatch scripts/02_build_container.sbatch
```

Pulls the pinned arm64 vLLM image (~11 GB) into a 9.4 GB `.sif`. The log must end with:

```
vllm 0.33.0 torch 2.14.0+cu130 cuda 13.0 True NVIDIA GH200 120GB
DeepseekV41ForCausalLM supported: True
```

Notes: Isambard-AI is aarch64, so only `linux/arm64` images work. The log is full of
`xattr ... ENOTSUP` warnings from Lustre; they are harmless. Build in a job, not on the login node.

### 3. Serve and benchmark on 2 nodes (~8 minutes, 8 GPUs)

```bash
sbatch scripts/03_serve_2node.sbatch
tail -f logs/dsv41-2node_<jobid>.out
```

Approximate timeline of a healthy run (weight-load and KV-cache figures are from the logs; the
clock times are rounded):

| Time | Log line |
|---|---|
| ~0:30 | `Resolved architecture: DeepseekV41ForCausalLM`, Engram tables offloaded to pinned host RAM (23.6 GiB per rank) |
| ~2:30 | `Model loading took 39.01 GiB memory and ~128 seconds` |
| ~4:30 | `Available KV cache memory: 40.8 GiB`, `GPU KV cache size: 12,724,074 tokens`, `Maximum concurrency for 262,144 tokens per request: 48.54x` |
| ~5:00 | `Application startup complete`, then `=== server up ===` and a hello-world reply |
| ~5:00-7:30 | benchmarks at concurrency 1, 16, 64 |

The job script then stops the server. The worker node would wait forever, so cancel with
`scancel <jobid>` if the job is still running (see the note in `node_entry.sh`).

To keep the endpoint up instead, remove the `pkill`/`scancel` lines at the end of
`scripts/node_entry.sh` and `wait` on the server. The API is OpenAI-compatible on port 8000
of the first node (`$HEAD_NODE`); tunnel to it with `ssh -L`.

## Why this configuration

- **2 nodes are the minimum.** Weights are 510 GB; one node has ~384 GB of HBM.
- **DP 2 x TP 4 + expert parallel**, not pipeline parallel. The model's own code guards for
  exactly this multi-node layout (pipeline size 1, expert parallel on, TP > 1, DP > 1).
  MoE experts are sharded across all 8 GPUs, attention weights are replicated per node,
  and the per-layer tensor-parallel all-reduce stays inside one node rather than crossing the network.
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

Tip: when something fails, the useful error is rarely at the bottom of the log. Search
`vllm_serve_<jobid>.log` and `vllm_worker_<jobid>.log` for the first `ERROR`, not the last.

## Known issues and caveats

- **Stray `</think>`** at the start of replies in the smoke test. Likely a reasoning-parser /
  chat-template setting; we did not investigate. A vLLM `--reasoning-parser` flag is the first thing to try.
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
