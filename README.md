# how-to-run-llm-on-isambard

Run very large open-weight LLMs on **2 Isambard-AI nodes (8 x GH200)** with vLLM: working scripts,
measured speed, and what it costs you in GPU-hours. One switch picks the model.

- **[Stats page](https://wd7512.github.io/how-to-run-llm-on-isambard/)**: measured throughput, latency and cost per model
- **[HOWTO.md](HOWTO.md)**: exact steps, and the errors you may hit
- **[`scripts/`](scripts/)**: Slurm and container scripts; one small file per model in [`scripts/models/`](scripts/models/)
- **[`results/`](results/)**: raw benchmark and GPU-hour data (JSON)

## Models

| `MODEL=` | Model | Weights | GPUs | Output tok/s, 1 / 64 users | GPU-hours to reproduce | Status |
|---|---|---:|---:|---:|---:|---|
| `dsv41` (default) | DeepSeek-V4.1-Flash-1017 | 510 GB | 8 (2 nodes) | 47 / 1,422 | ~1.5 | worked, CUDA graphs on |
| `glm53` | GLM-5.3-Flash | 328 GB | 8 (2 nodes) | 6.9 / 365 | ~2.3 | works in `--enforce-eager` only; CUDA graphs hang |
| `qwen38` | Qwen3.8-27B | 55.6 GB | **1** | 57 / 1,399 | ~0.4 | worked |
| `flint` | Flint 40B-A2.5B (BF16 base of the JANGH2 quant) | 79.6 GB | **2** | 196 / 3,986 | ~0.4 | worked |
| `mimo26` | MiMo-V2.6-Flash-RL | 178 GB | 4 (1 node) | 230 / 2,380 | ~0.9 | worked |
| `qwen38flash` | Qwen3.8-Flash-Next | 360 GB | 4 (1 node) | 156 / 2,660 | ~0.7 | worked |

Isambard-AI charges per GPU (1 GPU = 0.25 node-hours per hour), so the small models are requested with only
the GPUs they need: set per model with `GPUS` in `scripts/models/<name>.env`. GPU-hours = download +
container build (shared, ~0.3) + one serve-and-benchmark run. Benchmarks use short prompts (1,024 in /
256 out) and one run each.

`OsaurusAI/Flint-40B-A2.5B-JANGH2` itself is an Apple-Silicon (MLX) checkpoint that vLLM cannot load, so the
`flint` preset serves the original BF16 model it was made from.

## Quick start

On an Isambard-AI Phase 2 login node:

```bash
git clone https://github.com/wd7512/how-to-run-llm-on-isambard.git && cd how-to-run-llm-on-isambard
export MODEL=qwen38                          # dsv41 (default) | glm53 | qwen38 | flint | mimo26 | qwen38flash
bash scripts/00_setup_login_node.sh          # venv with the Hugging Face CLI
sbatch scripts/01_download.sbatch            # 2-10 min
sbatch scripts/02_build_container.sbatch     # ~20 min, once; shared by all models
bash scripts/03_serve.sh                     # requests the nodes/GPUs the model needs; smoke test + benchmarks
```

## Add another model

Copy a file in `scripts/models/`, change the repo, served name, architecture class, layout
(`NODES`, `GPUS`, `TP`, `DP_PER_NODE`, `EP`) and vLLM flags,
then run the same four commands with `MODEL=<yourfile>`. Check the vLLM recipe for the model first.

## Status

`glm53`, `qwen38`, `flint`, `mimo26` and `qwen38flash` were run from a fresh clone of this repo. `dsv41` numbers come from the original
absolute-path scripts; the current model-switch version has not been re-run for it.

Unofficial and unaffiliated with Isambard/BriCS, DeepSeek or Z.ai, Alibaba or Recursive. Isambard facts come from the
[Isambard documentation](https://docs.isambard.ac.uk/).
