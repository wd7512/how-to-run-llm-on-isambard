# A guide to running large LLMs on Isambard-AI

How to serve a large open-weight LLM on Isambard-AI (NVIDIA GH200) with vLLM, what throughput to expect, and what node
setup each model needs. Everything here was done and measured on the machine; the scripts in this repo are a **worked
example** of the steps, not a tool you have to use. Results for each model are on the
[stats site](https://wd7512.github.io/how-to-run-llm-on-isambard/).

## 1. What Isambard-AI looks like

These are the facts that shape everything below.

- **Hardware.** Phase 2 nodes have 4 NVIDIA GH200 superchips. Each has a 96 GiB GPU, 72 Arm cores and 115 GiB of RAM. The
  CPUs are **aarch64**, so only `linux/arm64` container images work. Driver 580 runs CUDA 13 images.
- **Slurm, one GPU partition.** There is no CPU-only partition, so even a download job asks for a GPU. **Billing is per GPU**
  (1 GPU = 0.25 node-hours per hour), so request only the GPUs you need.
- **Login nodes are tiny** (about 1 core and 4 GiB). Heavy processes are killed (exit 137) and background processes are
  reaped when your SSH session ends. Do all real work inside Slurm jobs.
- **Storage.** `$SCRATCHDIR` (Lustre, 5 TiB by default, not backed up) holds weights and containers. Node-local `/local` is
  RAM-backed and gone when the job ends; use it for caches that many processes write at once.
- **Containers** use Apptainer: `apptainer build image.sif docker://...`. For multi-node jobs load
  `brics/nccl` and `brics/apptainer-multi-node` and source `/host/adapt.sh` inside the container for the Slingshot network.

## 2. The recipe, by hand

Four steps. The commands are shortened; the scripts have the full versions.

**1. Download the weights, in a job.** One GPU, because there is no CPU partition. About 1 GB/s.

```bash
#SBATCH --gpus=1 --time=01:00:00
module load cray-python/3.11.7        # the login-node Python is 3.6
python3 -m venv $SCRATCHDIR/envs/hf && $SCRATCHDIR/envs/hf/bin/pip install huggingface_hub
$SCRATCHDIR/envs/hf/bin/hf download <repo> --revision <commit-sha> --local-dir $SCRATCHDIR/models/<name>
```

**2. Build the container, in a job.** The image needs more than the login node's 4 GiB. Pin it by digest, not by tag; tags move.

```bash
#SBATCH --gpus=1 --time=01:00:00
export APPTAINER_CACHEDIR=$SCRATCHDIR/tmp/cache APPTAINER_TMPDIR=$SCRATCHDIR/tmp
apptainer build $SCRATCHDIR/containers/vllm.sif docker://vllm/vllm-openai@sha256:<arm64 digest>
```

**3. Serve.** Request the nodes and GPUs the model needs, then run vLLM inside the container once per node.

```bash
#SBATCH --nodes=<N> --gpus-per-node=<G> --ntasks-per-node=1 --time=<cap>
module load brics/nccl brics/apptainer-multi-node
srun --kill-on-bad-exit=1 --export=ALL \
  apptainer exec --nv --bind $SCRATCHDIR vllm.sif  node_entry.sh     # runs `vllm serve ...` (flags in section 4)
```

On one node, `vllm serve <model> --tensor-parallel-size <G>` is enough. Across nodes, vLLM's built-in multi-node
data-parallel mode: node 0 runs `vllm serve ... --data-parallel-size <N> --data-parallel-size-local 1
--data-parallel-address <node 0> --data-parallel-rpc-port 13345`, and every other node runs the same command with
`--headless --data-parallel-start-rank <rank>`. (The image has no Ray.) Set the environment variables in section 4 first.

**4. Benchmark.** `vllm bench serve` against `localhost:8000`; the exact command is in section 8.

## 3. Choosing a node setup

- **Fit the weights.** A GPU has 96 GiB and vLLM uses 90% of it by default. Weights per GPU = weights / the number of GPUs
  they are split across. Leave room for the KV cache: vLLM prints `GPU KV cache size: N tokens` and
  `Maximum concurrency for 262,144 tokens per request: Nx` at start-up, and refuses to start if a single full-length
  context does not fit (Gemma 4 31B on one GPU: it needs 33.3 GiB of KV and only 24.5 GiB is free).
- **Tensor-parallel (`TP`) inside a node** splits each layer across GPUs; the size must divide the model's head and
  intermediate dimensions (Flint's dense MLP needs TP 1, 2 or 8). **Expert-parallel (`EP`)** spreads a mixture-of-experts
  model's experts across GPUs and is what makes the big MoE models fit.
- **Across nodes use data-parallel plus expert-parallel**, with tensor-parallel kept inside each node so the per-layer
  all-reduce never crosses the network. Pipeline-parallel across nodes failed in vLLM 0.33 for DeepSeek.
- **Use the fewest GPUs that fit.** Billing is per GPU, so a model that fits on 1 GPU costs a quarter of a node. More GPUs
  raise throughput but not in proportion (measured: Qwen3.8-27B on 4 GPUs gave about 1.7x the throughput of 1 GPU at 64 users;
  Flint on 4 GPUs gave 1.16x that of 2 GPUs).

What fit where (measured):

<!-- BEGIN:fit-table -->
| GPUs | Nodes | Model | Weights per GPU | Concurrent 262k contexts held | Layout |
|---:|---:|---|---:|---:|---|
| 1 | 1 | [Ornith-1.0-35B](#ornith35b) | 65 GiB | 3.4 | TP1 x DP1, 1 node(s) |
| 1 | 1 | [Gemma 4 26B-A4B](#gemma26b) | 49 GiB | 4.2 | TP1 x DP1, 1 node(s) |
| 1 | 1 | [Qwen3.8-27B](#qwen38) | 51 GiB | 1.9 | TP1 x DP1, 1 node(s) |
| 2 | 1 | [Flint 40B-A2.5B](#flint) | 37 GiB | 11.2 | TP2 x DP1 + expert-parallel, 1 node(s) |
| 2 | 1 | [Gemma 4 31B](#gemma31b) | 30 GiB | 3.2 | TP2 x DP1, 1 node(s) |
| 4 | 1 | [Qwen3.8-Flash-Next](#qwen38flash) | 61 GiB | 6.4 | TP4 x DP1 + expert-parallel, 1 node(s) |
| 4 | 1 | [MiMo-V2.6-Flash](#mimo26) | 42 GiB | 16.5 | TP4 x DP1, 1 node(s) |
| 8 | 2 | [DeepSeek-V4.1-Flash](#dsv41) | 39 GiB | 48.5 | TP4 x DP2 + expert-parallel, 2 node(s) |
| 8 | 2 | [GLM-5.3-Flash](#glm53) | 42 GiB | 13.2 | TP4 x DP2 + expert-parallel, 2 node(s) |
<!-- END:fit-table -->

## 4. What the vLLM flags mean

| Flag | What it does and why |
|---|---|
| `--tensor-parallel-size N` | Split every layer across N GPUs in a node. |
| `--data-parallel-size N`, `--data-parallel-size-local` | N full copies of the attention part across nodes; `-local` is how many run on this node. |
| `--data-parallel-address`, `--data-parallel-rpc-port` | Where the data-parallel replicas find node 0. |
| `--headless`, `--data-parallel-start-rank` | A worker node: no API server, and which data-parallel rank(s) it hosts. |
| `--enable-expert-parallel` | Shard the MoE experts across all GPUs instead of replicating them. |
| `--max-model-len` | Longest context to support; we use 262,144 (250k was the target). Sets the KV cache a single request needs. |
| `--max-num-seqs` | Most requests batched at once (default here 64). |
| `--gpu-memory-utilization 0.90` | Fraction of each GPU vLLM may use. |
| `--enforce-eager` | No CUDA graphs. Slower decode; used for GLM-5.3 where graph capture hung. |
| `--attention-backend` | Force an attention kernel (Flint requires FlashInfer). |
| `--reasoning-parser`, `--tool-call-parser`, `--enable-auto-tool-choice` | Split thinking from the answer, and parse tool calls, in the model's own format. |
| `--enable-prefix-caching` | Reuse KV cache across requests that share a prefix. |
| `--trust-remote-code`, `--served-model-name` | Allow the model's custom code; the name clients use. |

Environment variables set inside the container (all in `node_entry.sh`):

| Variable | Why |
|---|---|
| `TRITON_LIBCUDA_PATH=/.singularity.d/libs` | Triton's JIT compiler otherwise finds a host linker script instead of libcuda and fails to link. |
| `DG_JIT_CACHE_DIR`, `TRITON_CACHE_DIR`, `VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`, `XDG_CACHE_HOME` -> node-local `/local` | Several ranks write the same kernel cache; on shared Lustre they race (`missing CUDA CUBIN`, `Stale file handle`). |
| `VLLM_ENGINE_READY_TIMEOUT_S=3600` | Some models JIT-compile kernels on first start; the default 600 s gives up too soon. |
| `HF_HUB_OFFLINE=1` | Use the weights already on scratch. |

## 5. Results

<!-- BEGIN:models-table -->
| `MODEL=` | Model | Weights | GPUs | Output tok/s, 1 / 64 users | GPU-hours to run the benchmark | GPU-hours to reproduce | Status |
|---|---|---:|---:|---:|---:|---:|---|
| `dsv41` | [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | 510 GB | 8 | 47 / 1,446 | 0.101 | ~1.37 | worked |
| `glm53` | [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | 328 GB | 8 | 7.3 / 449 | 0.324 | ~2.14 | eager only |
| `qwen38` | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 55.6 GB | 1 | 57 / 1,421 | 0.013 | ~0.39 | worked |
| `flint` | [Flint 40B-A2.5B](https://huggingface.co/RecursiveSI/flint) | 79.6 GB | 2 | 194 / 3,983 | 0.009 | ~0.40 | worked |
| `mimo26` | [MiMo-V2.6-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) | 178 GB | 4 | 231 / 2,370 | 0.031 | ~0.90 | worked |
| `qwen38flash` | [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | 360 GB | 4 | 154 / 2,647 | 0.028 | ~0.78 | worked |
| `gemma31b` | [Gemma 4 31B](https://huggingface.co/google/gemma-4-31B) | 62.6 GB | 2 | 80 / 1,656 | 0.022 | ~0.51 | worked |
| `gemma26b` | [Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B) | 51.6 GB | 1 | 207 / 2,600 | 0.007 | ~0.41 | worked |
| `ornith35b` | [Ornith-1.0-35B](https://huggingface.co/ornith-ai/Ornith-1.0-35B) | 70.3 GB | 1 | 220 / 2,714 | 0.007 | ~0.37 | worked |
<!-- END:models-table -->

**GPU-hours to run the benchmark** is the comparable cost (section 8): the GPUs used x the time the 64-user run took, measured.
Every model below has its own notes and the command that was run.

## 6. Per-model notes and commands

<!-- BEGIN:model-sections -->
### dsv41

**DeepSeek-V4.1-Flash** ([deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), revision `08c8657f0229`). `DeepSeek-V4.1-Flash-1017`: 552B-parameter mixture-of-experts, 1M-token context, FP8 with FP4 experts, 510 GB. Needs 2 nodes (8 GPUs): the weights alone exceed one node's memory.

```bash
MODEL=dsv41 bash scripts/run.sh               # 2 node(s) x 4 GPU(s), time cap 00:18:00
MODEL=dsv41 python3 scripts/collect_results.py   # when the serve job has finished
```

What the head node ran (recorded by the run):

```bash
vllm serve <model-dir> \
  --served-model-name dsv41 \
  --tensor-parallel-size 4 \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-address <head-node> \
  --data-parallel-rpc-port 13345 \
  --enable-expert-parallel \
  --max-model-len 262144 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v41 \
  --port 8000
```

Measured: 47.1 output tok/s for one user, 1,446 at 64 users; **0.101 GPU-hours to run the benchmark** (the 64-user run: 256 requests of 1,024 input + 256 output tokens = 262,144 input / 65,536 output tokens, 45 s on 8 GPU(s)); ~1.37 GPU-hours to reproduce. Layout: TP4 x DP2 + expert-parallel, 2 node(s).

- Layout is tensor-parallel 4 per node x data-parallel 2 + expert parallel. Pipeline parallel across nodes fails in vLLM 0.33 for this model (`AssertionError` in `_run_layers`).
- vLLM puts the ~196B-parameter Engram embedding tables in pinned host RAM by itself (about 24 GiB per rank).
- Ray is not in the vLLM image, so the run uses vLLM's built-in multi-node data-parallel mode.
- The preset's `deepseek_v41` reasoning and tool-call parsers fix the stray `&lt;/think&gt;` that the first, parser-less run showed at the start of replies.

### glm53

**GLM-5.3-Flash** ([zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), revision `46a050bec130`). `zai-org/GLM-5.3-Flash`: MoE with 288 routed experts, 1M-token context, FP8, 328 GB. Needs 2 nodes. Runs with `--enforce-eager` because CUDA graphs hang on this layout, so these numbers are a floor, not the best this hardware can do.

```bash
MODEL=glm53 bash scripts/run.sh               # 2 node(s) x 4 GPU(s), time cap 00:33:00
MODEL=glm53 python3 scripts/collect_results.py   # when the serve job has finished
```

What the head node ran (recorded by the run):

```bash
vllm serve <model-dir> \
  --served-model-name glm53 \
  --tensor-parallel-size 4 \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-address <head-node> \
  --data-parallel-rpc-port 13345 \
  --enable-expert-parallel \
  --max-model-len 262144 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --reasoning-parser glm47 \
  --disable-flashinfer-autotune \
  --enforce-eager \
  --port 8000
```

Measured: 7.3 output tok/s for one user, 449 at 64 users; **0.324 GPU-hours to run the benchmark** (the 64-user run: 256 requests of 1,024 input + 256 output tokens = 262,144 input / 65,536 output tokens, 146 s on 8 GPU(s)); ~2.14 GPU-hours to reproduce. Layout: TP4 x DP2 + expert-parallel, 2 node(s).

- CUDA-graph capture hung at 12/19 for 22 minutes with no error (DP 2 x TP 4 + EP), so the preset uses `--enforce-eager`. Open item: find a graph mode that works, or try TP 8 across both nodes as the vLLM recipe does on one 8-GPU node.
- First start JIT-compiles FlashInfer kernels and vLLM gives up after 600 s by default: `VLLM_ENGINE_READY_TIMEOUT_S=3600` is set in `node_entry.sh`.
- The recipe's `glm5_next` tokenizer mode needs the Rust frontend, so the tokenizer is left on auto.

### qwen38

**Qwen3.8-27B** ([Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), revision `37561816e66b`). `Qwen/Qwen3.8-27B`: dense 27B, bf16, 55.6 GB, 262k context. Fits on a single GPU (51 GiB of weights).

```bash
MODEL=qwen38 bash scripts/run.sh               # 1 node(s) x 1 GPU(s), time cap 00:11:00
MODEL=qwen38 python3 scripts/collect_results.py   # when the serve job has finished
```

What the head node ran (recorded by the run):

```bash
vllm serve <model-dir> \
  --served-model-name qwen38 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data \
  --port 8000
```

Measured: 57.2 output tok/s for one user, 1,421 at 64 users; **0.013 GPU-hours to run the benchmark** (the 64-user run: 256 requests of 1,024 input + 256 output tokens = 262,144 input / 65,536 output tokens, 46 s on 1 GPU(s)); ~0.39 GPU-hours to reproduce. Layout: TP1 x DP1, 1 node(s).

- One GPU is enough; for more throughput raise `GPUS` and `DP_PER_NODE` together in `scripts/models/qwen38.env`.
- With one GPU the KV cache holds only about 1.9 full-length (262k) contexts.

### flint

**Flint 40B-A2.5B** ([RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint), revision `2a247be0c0ae`). `RecursiveSI/flint`: 39.8B MoE with ~2.5B active parameters, 256 experts, bf16, 79.6 GB, 262k context. This is the unquantised base of `OsaurusAI/Flint-40B-A2.5B-JANGH2`; that checkpoint is an Apple-Silicon (MLX) format that vLLM cannot load.

```bash
MODEL=flint bash scripts/run.sh               # 1 node(s) x 2 GPU(s), time cap 00:10:00
MODEL=flint python3 scripts/collect_results.py   # when the serve job has finished
```

What the head node ran (recorded by the run):

```bash
vllm serve <model-dir> \
  --served-model-name flint \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --max-model-len 262144 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90 \
  --attention-backend FLASHINFER \
  --port 8000
```

Measured: 193.8 output tok/s for one user, 3,983 at 64 users; **0.009 GPU-hours to run the benchmark** (the 64-user run: 256 requests of 1,024 input + 256 output tokens = 262,144 input / 65,536 output tokens, 16 s on 2 GPU(s)); ~0.40 GPU-hours to reproduce. Layout: TP2 x DP1 + expert-parallel, 1 node(s).

- `OsaurusAI/Flint-40B-A2.5B-JANGH2` will not load: it is an MLX checkpoint with a custom JANGH 2-bit quantisation. The preset serves the BF16 base instead.
- vLLM picks the FLASH_ATTN backend by default but the Flint model code asserts FlashInfer: the preset sets `--attention-backend FLASHINFER`.
- The dense-MLP kernel needs (6144 / TP) to be a multiple of 1024, so tensor-parallel must be 1, 2 or 8. TP 4 fails with `assert d % bd == 0`. The preset uses TP 2 on 2 GPUs.

### mimo26

**MiMo-V2.6-Flash** ([XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL), revision `f657ba2f7d24`). `XiaomiMiMo/MiMo-V2.6-Flash-RL`: MoE with 256 experts, 1M-token context, FP8 compute from mxfp4-stored weights, 178 GB.

```bash
MODEL=mimo26 bash scripts/run.sh               # 1 node(s) x 4 GPU(s), time cap 00:21:00
MODEL=mimo26 python3 scripts/collect_results.py   # when the serve job has finished
```

What the head node ran (recorded by the run):

```bash
vllm serve <model-dir> \
  --served-model-name mimo26 \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --generation-config vllm \
  --tool-call-parser mimo \
  --enable-auto-tool-choice \
  --reasoning-parser mimo \
  --port 8000
```

Measured: 230.6 output tok/s for one user, 2,370 at 64 users; **0.031 GPU-hours to run the benchmark** (the 64-user run: 256 requests of 1,024 input + 256 output tokens = 262,144 input / 65,536 output tokens, 28 s on 4 GPU(s)); ~0.90 GPU-hours to reproduce. Layout: TP4 x DP1, 1 node(s).

- Needs vLLM 0.31 or newer for the mxfp4 weights (the image has 0.33).
- The vLLM recipe recommends the `-MOPD` checkpoint over this `-RL` one, because `-RL` can get stuck repeating tool calls. The preset runs the `-RL` checkpoint; that behaviour was not tested.
- 4 GPUs is the recipe's minimum; 2 GPUs would not hold the 178 GB of weights plus KV cache.

### qwen38flash

**Qwen3.8-Flash-Next** ([Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), revision `812d96e10a0d`). `Qwen/Qwen3.8-Flash-Next`: MoE, 125B total and 6B active parameters plus a 51B n-gram embedding table, 512 experts, bf16, 360 GB. The embedding table is offloaded to host RAM, leaving 61 GiB of weights per GPU.

```bash
MODEL=qwen38flash bash scripts/run.sh               # 1 node(s) x 4 GPU(s), time cap 00:16:00
MODEL=qwen38flash python3 scripts/collect_results.py   # when the serve job has finished
```

What the head node ran (recorded by the run):

```bash
vllm serve <model-dir> \
  --served-model-name qwen38flash \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --max-model-len 262144 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 256 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --port 8000
```

Measured: 154.1 output tok/s for one user, 2,647 at 64 users; **0.028 GPU-hours to run the benchmark** (the 64-user run: 256 requests of 1,024 input + 256 output tokens = 262,144 input / 65,536 output tokens, 25 s on 4 GPU(s)); ~0.78 GPU-hours to reproduce. Layout: TP4 x DP1 + expert-parallel, 1 node(s).

- The n-gram embedding table is offloaded to host RAM by default, so the job needs the host memory of a full 4-GPU allocation.
- The recipe says to keep `--max-num-seqs 256` (smaller values hit a Mamba-cache capacity error); it is in the preset's extra args and overrides the default of 64.
- Plain bf16: no speculative decoding (MTP) and no FP8 KV cache. The vendor recipe's faster settings target B200 GPUs and were not tried on GH200.

### gemma31b

**Gemma 4 31B** ([google/gemma-4-31B](https://huggingface.co/google/gemma-4-31B), revision `5bbc2fb1c1b2`). `google/gemma-4-31B`: dense 31B, bf16, 62.6 GB, 262k context, Apache-2.0. The weights fit on one GPU, but a full 262k context needs 2.

```bash
MODEL=gemma31b bash scripts/run.sh               # 1 node(s) x 2 GPU(s), time cap 00:12:00
MODEL=gemma31b python3 scripts/collect_results.py   # when the serve job has finished
```

What the head node ran (recorded by the run):

```bash
vllm serve <model-dir> \
  --served-model-name gemma31b \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4 \
  --enable-auto-tool-choice \
  --port 8000
```

Measured: 80.1 output tok/s for one user, 1,656 at 64 users; **0.022 GPU-hours to run the benchmark** (the 64-user run: 256 requests of 1,024 input + 256 output tokens = 262,144 input / 65,536 output tokens, 40 s on 2 GPU(s)); ~0.51 GPU-hours to reproduce. Layout: TP2 x DP1, 1 node(s).

- Ungated, so no Hugging Face token is needed.
- `google/gemma-4-31B` is the pre-trained base checkpoint: it has no chat template, so the chat smoke test returns an error and the benchmark (which does not use the chat API) is what is measured. The instruction-tuned variant is `google/gemma-4-31B-it`; the architecture and size are the same, so speed should match, but it was not run.
- On one GPU vLLM refuses to start at 262,144 context: the weights take 58 GiB, leaving 24.5 GiB, and the KV cache for one 262k sequence needs 33.3 GiB (max context on 1 GPU is about 147k). The preset uses 2 GPUs (tensor-parallel 2).
- Uses vLLM's `gemma4` reasoning and tool-call parsers.

### gemma26b

**Gemma 4 26B-A4B** ([google/gemma-4-26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B), revision `24548b62aa02`). `google/gemma-4-26B-A4B`: MoE with 128 experts (8 active, about 4B active parameters), bf16, 51.6 GB, 262k context, Apache-2.0. Fits on a single GPU.

```bash
MODEL=gemma26b bash scripts/run.sh               # 1 node(s) x 1 GPU(s), time cap 00:10:00
MODEL=gemma26b python3 scripts/collect_results.py   # when the serve job has finished
```

What the head node ran (recorded by the run):

```bash
vllm serve <model-dir> \
  --served-model-name gemma26b \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4 \
  --enable-auto-tool-choice \
  --port 8000
```

Measured: 206.7 output tok/s for one user, 2,600 at 64 users; **0.007 GPU-hours to run the benchmark** (the 64-user run: 256 requests of 1,024 input + 256 output tokens = 262,144 input / 65,536 output tokens, 25 s on 1 GPU(s)); ~0.41 GPU-hours to reproduce. Layout: TP1 x DP1, 1 node(s).

- Ungated, so no Hugging Face token is needed.
- `google/gemma-4-26B-A4B` is the pre-trained base checkpoint: it has no chat template, so the chat smoke test returns an error and the benchmark (which does not use the chat API) is what is measured. The instruction-tuned variant is `google/gemma-4-26B-A4B-it`; the architecture and size are the same, so speed should match, but it was not run.
- Uses vLLM's `gemma4` reasoning and tool-call parsers.

### ornith35b

**Ornith-1.0-35B** ([ornith-ai/Ornith-1.0-35B](https://huggingface.co/ornith-ai/Ornith-1.0-35B), revision `5df2ed3f675c`). `ornith-ai/Ornith-1.0-35B`: MoE with 256 experts (8 active), about 35B parameters, bf16, 70.3 GB, 262k context, MIT. A reasoning model on the Qwen3.5-MoE architecture.

```bash
MODEL=ornith35b bash scripts/run.sh               # 1 node(s) x 1 GPU(s), time cap 00:10:00
MODEL=ornith35b python3 scripts/collect_results.py   # when the serve job has finished
```

What the head node ran (recorded by the run):

```bash
vllm serve <model-dir> \
  --served-model-name ornith35b \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --trust-remote-code \
  --port 8000
```

Measured: 220.1 output tok/s for one user, 2,714 at 64 users; **0.007 GPU-hours to run the benchmark** (the 64-user run: 256 requests of 1,024 input + 256 output tokens = 262,144 input / 65,536 output tokens, 24 s on 1 GPU(s)); ~0.37 GPU-hours to reproduce. Layout: TP1 x DP1, 1 node(s).

- Ungated, so no Hugging Face token is needed. Needs vLLM 0.19.1 or newer (the image has 0.33).
- Uses the model card's flags: `qwen3` reasoning parser, `qwen3_xml` tool-call parser, prefix caching.
- It is a reasoning model: the 64-token smoke-test reply is all thinking (returned in the `reasoning` field, empty `content`), which is expected and does not affect the benchmark.
- One GPU is enough: 65.5 GiB of weights leave a KV cache that holds about 3.4 full 262k-token contexts.

<!-- END:model-sections -->

## 7. The worked example: this repo's scripts

You can follow section 2 by hand. The repo wraps it so a run is repeatable and its results are recorded:

```bash
git clone https://github.com/wd7512/how-to-run-llm-on-isambard.git && cd how-to-run-llm-on-isambard
MODEL=qwen38 bash scripts/run.sh                  # download -> build container -> serve + benchmark, chained
MODEL=qwen38 python3 scripts/collect_results.py   # writes results/qwen38_benchmarks.json from the saved run
python3 scripts/build_site.py                     # regenerates docs/data.js and the tables from results/
```

| Script | Does |
|---|---|
| `00_setup_login_node.sh`, `01_download.sbatch` | step 1: venv, then the pinned-revision download job (skipped if already done) |
| `02_build_container.sbatch` | step 2: the pinned image, plus a check that vLLM knows the model's architecture |
| `03_serve.sh`, `03_serve.sbatch`, `node_entry.sh` | step 3 and 4: nodes, GPUs and time cap from the model's file; sets the environment; starts vLLM; smoke test; benchmark; signals the other nodes to stop |
| `scripts/models/<name>.env` | one small file per model: repo, pinned commit, nodes, GPUs, layout, vLLM flags |

Everything that affects a result is pinned or recorded: the image by digest (vLLM 0.33.0, torch 2.14, CUDA 13.0), each model by
Hugging Face commit, the benchmark by seed and token lengths, and each run saves its own metadata, effective command and measured
token counts under `logs/run_<jobid>/`. The serve job has a hard time cap so a hang cannot burn more than the cap.

**Adding a model:** copy a file in `scripts/models/`, set the repo, `MODEL_REVISION`, `ARCH` (from the model's `config.json`),
`NODES`, `GPUS`, `TP`, `DP_PER_NODE`, `EP`, `TIME_LIMIT` and the flags from the model's vLLM recipe; add its description and
notes to `site_src/models.json`; then run the three commands above.

## 8. Reproducing the benchmark and the cost metric

```bash
vllm bench serve --model <served-name> --tokenizer <model dir> --port 8000 \
  --dataset-name random --random-input-len 1024 --random-output-len 256 --seed 0 \
  --num-prompts $((4*C)) --max-concurrency $C --save-result --result-dir <dir> --result-filename c$C.json     # C = 1, 16, 64
```

- **Input tokens:** 1,024 per request, random tokens (so no prompt-cache hits). **Output tokens:** 256 per request. That is
  4 input tokens for every output token, so each output token carries 4 tokens of prefill.
- **The 64-user run** is 256 requests: 262,144 input and 65,536 output tokens (the saved JSON records the measured counts).
- **GPU-hours to run the benchmark** = GPUs x the duration of the 64-user run / 3600. It is measured, not extrapolated, and
  every model runs the same 256 requests (262,144 input, 65,536 output tokens), so it compares models. Change the input or
  output length and it changes with it, so quote the token counts whenever you quote the number.
- **GPU-hours** come from Slurm: `sacct -X -P -j <ids> -o JobID,ElapsedRaw,AllocTRES` (GPUs x elapsed seconds / 3600).

## 9. Troubleshooting

Symptom -> cause -> fix. The scripts already include these fixes. Model-specific problems are in section 6.

1. **Download dies with exit 137 about 20 seconds in.** The login node's 4 GiB; the OOM killer ends big transfers. -> Download in a Slurm job.
2. **`ray: command not found` / "ray GPUs: 0".** The vLLM image has no Ray. -> vLLM's built-in multi-node mode (section 2).
3. **`/usr/bin/ld: cannot find -l:libcuda.so.1`** at the first forward pass (Triton compiling `cuda_utils`). -> `TRITON_LIBCUDA_PATH` (section 4).
4. **`AssertionError ... residual is not None and post_mix is not None`** with `--pipeline-parallel-size 2` (DeepSeek). -> data-parallel + expert-parallel instead.
5. **`missing CUDA CUBIN: ~/.cache/vllm/deep_gemm/...`, `OSError: [Errno 116] Stale file handle`.** Ranks race on kernel caches on Lustre. -> node-local cache directories (section 4).
6. **A crashed or hung server keeps burning allocation.** -> the wait loop checks the server is alive, `srun --kill-on-bad-exit=1`, and a hard `--time` cap.
7. **`TimeoutError: Timed out waiting for engine core processes to start ... 600s`.** Kernels are JIT-compiling on first start. -> `VLLM_ENGINE_READY_TIMEOUT_S=3600`.

Tip: the useful error is rarely at the bottom of the log. Search `logs/vllm_serve_<jobid>.log` and `logs/vllm_worker_<jobid>.log` for the first `ERROR`, not the last.

## 10. Known limitations

- **Short benchmarks only.** Nothing here measures a 250k-token prompt (prefill time, long-context decode speed).
- **Two runs per model.** Seven of nine models agreed within about 4% between runs (most within 1%); DeepSeek (+14% at 16 users) and GLM (up to +23% at 64 users) varied more, so treat the two-node numbers as roughly +/-20%. Each results file keeps the earlier run under `previous_run`. No repeats beyond that, so no proper error bars.
- **Node-local caches mean kernels recompile every job** (about 1 minute of start-up).
