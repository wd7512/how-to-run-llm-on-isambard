# Running a large LLM on Isambard-AI with vLLM

One command per model; this page explains what it does, how to add a model, and what breaks. Measured results are on the
[stats site](https://wd7512.github.io/how-to-run-llm-on-isambard/). Model names (`MODEL=`) are listed below.

## Prerequisites

- An Isambard-AI Phase 2 account and project, and an SSH certificate from `clifton auth`.
- Free space in `$SCRATCHDIR` for the weights (the default quota is 5 TiB). Weights range from 52 GB to 510 GB per model.
- GPU allocation: see "GPU-hours to reproduce" per model below (0.2 to 2.3 GPU-hours, plus ~0.3 once for the container).

## The short path

```bash
ssh <project>.aip2.isambard
git clone https://github.com/wd7512/how-to-run-llm-on-isambard.git && cd how-to-run-llm-on-isambard
MODEL=qwen38 bash scripts/run.sh                  # everything, chained with Slurm dependencies
MODEL=qwen38 python3 scripts/collect_results.py   # after the serve job finishes
python3 scripts/build_site.py                     # (optional) regenerate docs/data.js and the tables from results/
```

Keep submitting from the repo root: `logs/` is relative to it. Watch with `squeue -u $USER` and
`tail -f logs/serve-<model>_<jobid>.out`.

## What `run.sh` does

| Step | Script | Where | Cost |
|---|---|---|---|
| 0 | `00_setup_login_node.sh` | login node | a Python venv with the Hugging Face CLI. **Never download weights on the login node** (see troubleshooting 1). |
| 1 | `01_download.sbatch` | 1 GPU | downloads the pinned revision at ~1 GB/s, then writes `.revision` so it is skipped next time. |
| 2 | `02_build_container.sbatch` | 1 GPU | pulls the pinned arm64 vLLM image (~11 GB) into a 9.4 GB `.sif`; skipped if it exists. Checks vLLM knows the model's architecture. |
| 3 | `03_serve.sbatch` via `03_serve.sh` | model's nodes x GPUs | starts vLLM, runs a smoke test, then the benchmark at 1, 16 and 64 concurrent users, and exits. |

`03_serve.sh` reads the model's file (`scripts/models/<name>.env`): `NODES`, `GPUS` per node, `TIME_LIMIT`, layout
(`TP`, `DP_PER_NODE`, `EP`) and vLLM flags. The Slurm time limit is a hard cap, so a hung job cannot burn more than the
cap. When the head node finishes, it writes `logs/run_<jobid>/DONE` and the other nodes exit too.

## Models

<!-- BEGIN:models-table -->
| `MODEL=` | Model | Weights | GPUs | Output tok/s, 1 / 64 users | GPU-hours per 1M output tokens | GPU-hours to reproduce | Status |
|---|---|---:|---:|---:|---:|---:|---|
| `dsv41` | [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | 510 GB | 8 | 47 / 1,422 | 1.56 | ~1.53 | worked |
| `glm53` | [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | 328 GB | 8 | 6.8 / 365 | 6.10 | ~2.33 | eager only |
| `qwen38` | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 55.6 GB | 1 | 57 / 1,413 | 0.20 | ~0.40 | worked |
| `flint` | [Flint 40B-A2.5B](https://huggingface.co/RecursiveSI/flint) | 79.6 GB | 2 | 196 / 3,986 | 0.14 | ~0.40 | worked |
| `mimo26` | [MiMo-V2.6-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) | 178 GB | 4 | 230 / 2,380 | 0.47 | ~0.91 | worked |
| `qwen38flash` | [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | 360 GB | 4 | 156 / 2,660 | 0.42 | ~0.73 | worked |
<!-- END:models-table -->

## Per-model notes and exact commands

<!-- BEGIN:model-sections -->
### dsv41

**DeepSeek-V4.1-Flash** ([deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), revision `08c8657f0229`). `DeepSeek-V4.1-Flash-1017`: 552B-parameter mixture-of-experts, 1M-token context, FP8 with FP4 experts, 510 GB. Needs 2 nodes (8 GPUs): the weights alone exceed one node's memory.

```bash
MODEL=dsv41 bash scripts/run.sh               # 2 node(s) x 4 GPU(s), time cap 00:40:00
MODEL=dsv41 python3 scripts/collect_results.py   # when the serve job has finished
```

Measured: 47.0 output tok/s for one user, 1,422 at 64 users; **1.56 GPU-hours per 1M output tokens** (1,024 input + 256 output tokens per request, 64-user run = 262,144 input / 65,536 output tokens); ~1.53 GPU-hours to reproduce. Layout: tensor-parallel 4 x data-parallel 2 + expert-parallel.

- Layout is tensor-parallel 4 per node x data-parallel 2 + expert parallel. Pipeline parallel across nodes fails in vLLM 0.33 for this model (`AssertionError` in `_run_layers`).
- vLLM puts the ~196B-parameter Engram embedding tables in pinned host RAM by itself (about 24 GiB per rank).
- Ray is not in the vLLM image, so the run uses vLLM's built-in multi-node data-parallel mode.

### glm53

**GLM-5.3-Flash** ([zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), revision `46a050bec130`). `zai-org/GLM-5.3-Flash`: MoE with 288 routed experts, 1M-token context, FP8, 328 GB. Needs 2 nodes. Runs with `--enforce-eager` because CUDA graphs hang on this layout, so these numbers are a floor, not the best this hardware can do.

```bash
MODEL=glm53 bash scripts/run.sh               # 2 node(s) x 4 GPU(s), time cap 00:40:00
MODEL=glm53 python3 scripts/collect_results.py   # when the serve job has finished
```

Measured: 6.8 output tok/s for one user, 365 at 64 users; **6.10 GPU-hours per 1M output tokens** (1,024 input + 256 output tokens per request, 64-user run = 262,144 input / 65,536 output tokens); ~2.33 GPU-hours to reproduce. Layout: tensor-parallel 4 x data-parallel 2 + expert-parallel, --enforce-eager (no CUDA graphs).

- CUDA-graph capture hung at 12/19 for 22 minutes with no error (DP 2 x TP 4 + EP), so the preset uses `--enforce-eager`. Open item: find a graph mode that works, or try TP 8 across both nodes as the vLLM recipe does on one 8-GPU node.
- First start JIT-compiles FlashInfer kernels and vLLM gives up after 600 s by default: `VLLM_ENGINE_READY_TIMEOUT_S=3600` is set in `node_entry.sh`.
- The recipe's `glm5_next` tokenizer mode needs the Rust frontend, so the tokenizer is left on auto.

### qwen38

**Qwen3.8-27B** ([Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), revision `37561816e66b`). `Qwen/Qwen3.8-27B`: dense 27B, bf16, 55.6 GB, 262k context. Fits on a single GPU (51 GiB of weights).

```bash
MODEL=qwen38 bash scripts/run.sh               # 1 node(s) x 1 GPU(s), time cap 00:30:00
MODEL=qwen38 python3 scripts/collect_results.py   # when the serve job has finished
```

Measured: 57.3 output tok/s for one user, 1,413 at 64 users; **0.20 GPU-hours per 1M output tokens** (1,024 input + 256 output tokens per request, 64-user run = 262,144 input / 65,536 output tokens); ~0.40 GPU-hours to reproduce. Layout: TP1 x DP1, 1 node(s).

- One GPU is enough; for more throughput raise `GPUS` and `DP_PER_NODE` together in `scripts/models/qwen38.env`.
- With one GPU the KV cache holds only about 1.9 full-length (262k) contexts.

### flint

**Flint 40B-A2.5B** ([RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint), revision `2a247be0c0ae`). `RecursiveSI/flint`: 39.8B MoE with ~2.5B active parameters, 256 experts, bf16, 79.6 GB, 262k context. This is the unquantised base of `OsaurusAI/Flint-40B-A2.5B-JANGH2`; that checkpoint is an Apple-Silicon (MLX) format that vLLM cannot load.

```bash
MODEL=flint bash scripts/run.sh               # 1 node(s) x 2 GPU(s), time cap 00:30:00
MODEL=flint python3 scripts/collect_results.py   # when the serve job has finished
```

Measured: 196.3 output tok/s for one user, 3,986 at 64 users; **0.14 GPU-hours per 1M output tokens** (1,024 input + 256 output tokens per request, 64-user run = 262,144 input / 65,536 output tokens); ~0.40 GPU-hours to reproduce. Layout: 2 GPUs, TP2 + expert-parallel, --attention-backend FLASHINFER.

- `OsaurusAI/Flint-40B-A2.5B-JANGH2` will not load: it is an MLX checkpoint with a custom JANGH 2-bit quantisation. The preset serves the BF16 base instead.
- vLLM picks the FLASH_ATTN backend by default but the Flint model code asserts FlashInfer: the preset sets `--attention-backend FLASHINFER`.
- The dense-MLP kernel needs (6144 / TP) to be a multiple of 1024, so tensor-parallel must be 1, 2 or 8. TP 4 fails with `assert d % bd == 0`. The preset uses TP 2 on 2 GPUs.

### mimo26

**MiMo-V2.6-Flash** ([XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL), revision `f657ba2f7d24`). `XiaomiMiMo/MiMo-V2.6-Flash-RL`: MoE with 256 experts, 1M-token context, FP8 compute from mxfp4-stored weights, 178 GB.

```bash
MODEL=mimo26 bash scripts/run.sh               # 1 node(s) x 4 GPU(s), time cap 00:30:00
MODEL=mimo26 python3 scripts/collect_results.py   # when the serve job has finished
```

Measured: 230.2 output tok/s for one user, 2,380 at 64 users; **0.47 GPU-hours per 1M output tokens** (1,024 input + 256 output tokens per request, 64-user run = 262,144 input / 65,536 output tokens); ~0.91 GPU-hours to reproduce. Layout: tensor-parallel 4, 1 node.

- Needs vLLM 0.31 or newer for the mxfp4 weights (the image has 0.33).
- The vLLM recipe recommends the `-MOPD` checkpoint over this `-RL` one, because `-RL` can get stuck repeating tool calls. The preset runs the `-RL` checkpoint; that behaviour was not tested.
- 4 GPUs is the recipe's minimum; 2 GPUs would not hold the 178 GB of weights plus KV cache.

### qwen38flash

**Qwen3.8-Flash-Next** ([Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), revision `812d96e10a0d`). `Qwen/Qwen3.8-Flash-Next`: MoE, 125B total and 6B active parameters plus a 51B n-gram embedding table, 512 experts, bf16, 360 GB. The embedding table is offloaded to host RAM, leaving 61 GiB of weights per GPU.

```bash
MODEL=qwen38flash bash scripts/run.sh               # 1 node(s) x 4 GPU(s), time cap 00:30:00
MODEL=qwen38flash python3 scripts/collect_results.py   # when the serve job has finished
```

Measured: 156.0 output tok/s for one user, 2,660 at 64 users; **0.42 GPU-hours per 1M output tokens** (1,024 input + 256 output tokens per request, 64-user run = 262,144 input / 65,536 output tokens); ~0.73 GPU-hours to reproduce. Layout: tensor-parallel 4 + expert-parallel, 1 node, n-gram embedding offloaded to host RAM.

- The n-gram embedding table is offloaded to host RAM by default, so the job needs the host memory of a full 4-GPU allocation.
- The recipe says to keep `--max-num-seqs 256` (smaller values hit a Mamba-cache capacity error); it is in the preset's extra args and overrides the default of 64.
- Plain bf16: no speculative decoding (MTP) and no FP8 KV cache. The vendor recipe's faster settings target B200 GPUs and were not tried on GH200.

<!-- END:model-sections -->

## Why this configuration

- **Big models need 2 nodes.** A node has ~384 GiB of HBM. Layout: tensor-parallel 4 per node x data-parallel across nodes +
  expert parallel. MoE experts are sharded across all GPUs, attention weights are replicated per node, and the per-layer
  all-reduce stays inside one node rather than crossing the network. Pipeline parallel across nodes fails in vLLM 0.33.
- **Small models request only what they need.** Isambard charges per GPU, so a 1-GPU model costs a quarter of a node.
- **`--max-model-len 262144`** because 250k context was the target. The KV cache for most models holds several such
  sequences at once; the per-model figure is on the stats site.
- **Isambard-AI is aarch64**, so only `linux/arm64` images work. The cluster driver (580.178.04) is new enough for CUDA 13
  images; on an older driver use a CUDA 12 image tag.

## Troubleshooting

Symptom -> cause -> fix. The scripts already include these fixes; if you change the setup, these are the errors to expect.
Model-specific problems are in the per-model notes above.

1. **Download dies with exit 137 about 20 seconds in.** The login node allows about 1 core and 4 GiB; the OOM killer ends big
   transfers. Background processes there are also reaped when your SSH session ends. -> Download in a Slurm job.
2. **`ray: command not found` / "ray GPUs: 0".** The vLLM image does not contain Ray. -> Use vLLM's built-in multi-node mode
   (`--data-parallel-*`, `--headless`).
3. **`/usr/bin/ld: cannot find -l:libcuda.so.1`** during the first forward pass, from Triton compiling `cuda_utils`. Triton
   finds libcuda via `ldconfig`, which inside the container resolves to a host directory holding a glibc linker script.
   -> `export TRITON_LIBCUDA_PATH=/.singularity.d/libs`.
4. **`AssertionError` at `assert residual is not None and post_mix is not None`** with `--pipeline-parallel-size 2` (DeepSeek).
   -> Use data-parallel + expert-parallel instead.
5. **`RuntimeError: ... missing CUDA CUBIN: ~/.cache/vllm/deep_gemm/...`.** Eight ranks compile and read the same kernel cache
   on shared Lustre. -> `DG_JIT_CACHE_DIR` on node-local `/local`.
6. **`OSError: [Errno 116] Stale file handle`** from Triton's cache, same class of race. -> `TRITON_CACHE_DIR`,
   `VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`, `XDG_CACHE_HOME` all on `/local`. `node_entry.sh` sets all five.
7. **A crashed server keeps burning allocation.** A script that waits for `/health` forever holds the nodes after the server
   died. -> the wait loop checks `kill -0`, `srun --kill-on-bad-exit=1` tears the job down, and `TIME_LIMIT` caps it.
8. **`TimeoutError: Timed out waiting for engine core processes to start ... 600s`** after the weights loaded. Some models
   JIT-compile kernels on first use. -> `VLLM_ENGINE_READY_TIMEOUT_S=3600` (set in `node_entry.sh`).

Tip: the useful error is rarely at the bottom of the log. Search `logs/vllm_serve_<jobid>.log` and
`logs/vllm_worker_<jobid>.log` for the first `ERROR`, not the last.

## Reproducing the benchmark and the cost metric

`node_entry.sh` runs this after the smoke test, for C = 1, 16 and 64:

```bash
vllm bench serve --model <served-name> --tokenizer <model dir> --port 8000 \
  --dataset-name random --random-input-len 1024 --random-output-len 256 --seed 0 \
  --num-prompts $((4*C)) --max-concurrency $C --save-result --result-dir logs/run_<jobid> --result-filename c$C.json
```

- **Input tokens:** 1,024 per request, random tokens (so no prompt-cache hits). **Output tokens:** 256 per request. That is
  4 input tokens for every output token, so each output token carries 4 tokens of prefill.
- **The 64-user run** is 256 requests: 262,144 input and 65,536 output tokens (the saved JSON records the measured counts).
- **GPU-hours per 1M output tokens** = `GPUs / output_tok_s / 3600 * 1e6`, using that run. Change the input or output length
  and the metric changes with it, so quote the token counts whenever you quote the number.
- **GPU-hours** come from Slurm: `sacct -X -P -j <ids> -o JobID,ElapsedRaw,AllocTRES` (GPUs x elapsed seconds / 3600).

## Adding a model

1. Copy a file in `scripts/models/` and set the repo, `MODEL_REVISION` (the commit to pin), served name, `ARCH` (the
   architecture class from the model's `config.json`; step 2 checks vLLM supports it), `NODES`, `GPUS`, `TP`,
   `DP_PER_NODE`, `EP`, `TIME_LIMIT` and `VLLM_EXTRA_ARGS` (take the flags from the model's vLLM recipe).
2. Add its description and notes to `site_src/models.json`.
3. `MODEL=<name> bash scripts/run.sh`, then `collect_results.py` and `build_site.py`.

Sizing: weights (GiB) / GPUs per replica must leave room for KV cache in 96 GiB; check `TP` divides the model's head and
intermediate sizes.

## Known limitations

- **Short benchmarks only.** Nothing here measures a 250k-token prompt (prefill time, long-context decode).
- **Single run per model**, no repeats, so no error bars.
- **Node-local caches mean kernels recompile every job** (~1 minute of start-up).
- **Provenance.** Runs made before `run.sh` existed were copied by hand from job logs; their results files say so under
  `provenance`.
