# how-to-run-llm-on-isambard

Run very large open-weight LLMs on **2 Isambard-AI nodes (8 x GH200)** with vLLM: working scripts,
measured speed, and what it costs you in GPU-hours. One switch picks the model.

- **[Stats page](https://wd7512.github.io/how-to-run-llm-on-isambard/)**: measured throughput, latency and cost per model
- **[HOWTO.md](HOWTO.md)**: exact steps, and the errors you may hit
- **[`scripts/`](scripts/)**: Slurm and container scripts; one small file per model in [`scripts/models/`](scripts/models/)
- **[`results/`](results/)**: raw benchmark and GPU-hour data (JSON)

## Models

| `MODEL=` | Model | Weights | Output tok/s, 1 user / 64 users | GPU-hours to reproduce | Status |
|---|---|---:|---:|---:|---|
| `dsv41` (default) | DeepSeek-V4.1-Flash-1017 | 510 GB | 47 / 1,422 | ~1.5 | worked, CUDA graphs on |
| `glm53` | GLM-5.3-Flash | 328 GB | 6.9 / 365 | ~2.3 | works in `--enforce-eager` only; CUDA graphs hang |

GPU-hours = download + container build (shared, ~0.3) + one serve-and-benchmark run. Serving on 2 nodes
afterwards costs 8 GPU-hours per hour. Benchmarks use short prompts (1,024 in / 256 out) and one run each.

## Quick start

On an Isambard-AI Phase 2 login node:

```bash
git clone https://github.com/wd7512/how-to-run-llm-on-isambard.git && cd how-to-run-llm-on-isambard
export MODEL=glm53                           # or dsv41 (default)
bash scripts/00_setup_login_node.sh          # venv with the Hugging Face CLI
sbatch scripts/01_download.sbatch            # 6-10 min
sbatch scripts/02_build_container.sbatch     # ~20 min, once; shared by all models
sbatch scripts/03_serve_2node.sbatch         # 2 nodes; smoke test + benchmarks
```

## Add another model

Copy `scripts/models/glm53.env`, change the repo, served name, architecture class and vLLM flags,
then run the same four commands with `MODEL=<yourfile>`. Check the vLLM recipe for the model first.

## Status

`glm53` was run end to end from a fresh clone of this repo. `dsv41` numbers come from the original
absolute-path scripts; the current model-switch version has not been re-run for it.

Unofficial and unaffiliated with Isambard/BriCS, DeepSeek or Z.ai. Isambard facts come from the
[Isambard documentation](https://docs.isambard.ac.uk/).
