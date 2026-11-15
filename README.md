# how-to-run-llm-on-isambard

Running a 552B-parameter LLM (**DeepSeek-V4.1-Flash-1017**) on **2 Isambard-AI nodes (8 x GH200)** with vLLM:
what worked, what broke, what it cost, and the scripts to repeat it.

- **[Stats page](https://wd7512.github.io/how-to-run-llm-on-isambard/)** - measured throughput, latency and GPU-hour cost
- **[HOWTO.md](HOWTO.md)** - exact reproducible steps and a troubleshooting log of every failure we hit
- **[`scripts/`](scripts/)** - the Slurm and container scripts
- **[`results/`](results/)** - raw benchmark and GPU-hour data (JSON)

## Results (15 November 2026)

| Concurrent users | Aggregate output tok/s | Per-user tok/s | Time to first token |
|---:|---:|---:|---:|
| 1 | 47 | 49 | 231 ms |
| 16 | 491 | 33 | 691 ms |
| 64 | 1,422 | 24 | 1,037 ms |

- Weights are 39 GiB per GPU; the KV cache holds 12.7M tokens, enough for **48 concurrent 262k-token contexts**.
- Cost: **7.3 GPU-hours (1.8 node-hours)** in total, of which only 1.0 was the working run. The rest was
  download, container build and six failed attempts, all documented in [HOWTO.md](HOWTO.md#troubleshooting).
- Benchmarks are short (1,024 input / 256 output tokens) and from a single run. They do not cover
  250k-token prompts.

## Quick start

On an Isambard-AI Phase 2 login node:

```bash
git clone https://github.com/wd7512/how-to-run-llm-on-isambard.git && cd how-to-run-llm-on-isambard
bash scripts/00_setup_login_node.sh          # venv with the Hugging Face CLI
sbatch scripts/01_download.sbatch            # ~10 min, 510 GB
sbatch scripts/02_build_container.sbatch     # ~20 min, pinned arm64 vLLM image
sbatch scripts/03_serve_2node.sbatch         # ~8 min on 2 nodes; smoke test + benchmarks
```

## Status

The scripts are the ones that produced these results, with paths parameterised and the container pinned
by digest. That parameterised version has **not been re-run end-to-end**. See the note at the top of
[HOWTO.md](HOWTO.md).

## What was the hard part

Not the model: vLLM 0.33 already supports it. Getting it to start on Isambard needed four fixes, all in
[`scripts/node_entry.sh`](scripts/node_entry.sh): data-parallel + expert-parallel instead of
pipeline-parallel, a `TRITON_LIBCUDA_PATH` override for the container's JIT compiler, node-local kernel
caches instead of shared Lustre, and no Ray.

Unofficial and unaffiliated with Isambard/BriCS or DeepSeek. Isambard-specific facts come from the
[Isambard documentation](https://docs.isambard.ac.uk/).
