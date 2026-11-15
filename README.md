# how-to-run-llm-on-isambard

Run large open-weight LLMs on Isambard-AI (GH200) with vLLM: one command per model, measured speed, and
what it costs in GPU-hours.

- **[Stats site](https://wd7512.github.io/how-to-run-llm-on-isambard/)**: overview of all models, then a page per model with its own steps
- **[HOWTO.md](HOWTO.md)**: the full guide, per-model notes and troubleshooting
- **`scripts/`**: `run.sh` (one command), `collect_results.py`, `build_site.py`, and one small file per model in `scripts/models/`
- **`results/`**: one JSON per model (benchmarks, token counts, GPU-hours, provenance)

## Run a model

On an Isambard-AI Phase 2 login node:

```bash
git clone https://github.com/wd7512/how-to-run-llm-on-isambard.git && cd how-to-run-llm-on-isambard
MODEL=qwen38 bash scripts/run.sh                  # setup + download + container build + serve + benchmark, chained
MODEL=qwen38 python3 scripts/collect_results.py   # when the serve job has finished: writes results/qwen38_benchmarks.json
```

`MODEL` is any name in the table below. `run.sh` skips steps already done, requests only the GPUs the model
needs (billing is per GPU), and caps the serve job's time so a hang cannot burn allocation.

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

**GPU-hours per 1M output tokens** is the comparable cost: GPUs ÷ aggregate output tok/s at 64 users ÷ 3,600 × 1,000,000
(lower is cheaper). It is only meaningful under the fixed workload: **every request is 1,024 input + 256 output
tokens** (4 input tokens per output token), 256 requests per run, random-token prompts, seed 0, 64 at a time.
Isambard-AI charges per GPU (1 GPU = 0.25 node-hours per hour). "GPU-hours to reproduce" is download + container
build (shared) + one serve-and-benchmark run.

## Determinism

Everything that affects a result is pinned or recorded: the container by digest (vLLM 0.33.0, torch 2.14, CUDA 13.0),
each model by Hugging Face commit (`MODEL_REVISION` in its `.env`), the benchmark by seed and token lengths, and every
run saves its own metadata and measured token counts (`logs/run_<jobid>/`). `collect_results.py` writes the results
file from those, and `build_site.py` generates the site and tables from them, so no number is typed in by hand.
Runs made before this pipeline existed are marked in their results file under `provenance`.

Unofficial and unaffiliated with Isambard/BriCS or any model vendor. Isambard facts come from the
[Isambard documentation](https://docs.isambard.ac.uk/).
