# how-to-run-llm-on-isambard

A practical guide to running large open-weight LLMs on Isambard-AI (NVIDIA GH200) with vLLM: **how to do it, what
throughput to expect, and what node setup each model needs**, with nine models measured as worked examples.

- **[Stats site](https://wd7512.github.io/how-to-run-llm-on-isambard/)**: an overview of all models, then a page per model with
  exactly how it was served (the Slurm request, container command, environment and `vllm serve` flags)
- **[HOWTO.md](HOWTO.md)**: the guide: what Isambard-AI is like, the recipe step by step, how to choose a node setup, what each
  flag means, troubleshooting
- **`scripts/` and `results/`**: the scripts that produced the numbers (a worked example, not a tool you have to use) and the
  raw results, one JSON per model

## Models

<!-- BEGIN:models-table -->
| `MODEL=` | Model | Weights | GPUs | Output tok/s, 1 / 64 users | GPU-hours per 1M output tokens | GPU-hours to reproduce | Status |
|---|---|---:|---:|---:|---:|---:|---|
| `dsv41` | [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | 510 GB | 8 | 47 / 1,446 | 1.54 | ~1.37 | worked |
| `glm53` | [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | 328 GB | 8 | 7.3 / 449 | 4.95 | ~2.14 | eager only |
| `qwen38` | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 55.6 GB | 1 | 57 / 1,421 | 0.20 | ~0.39 | worked |
| `flint` | [Flint 40B-A2.5B](https://huggingface.co/RecursiveSI/flint) | 79.6 GB | 2 | 194 / 3,983 | 0.14 | ~0.40 | worked |
| `mimo26` | [MiMo-V2.6-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) | 178 GB | 4 | 231 / 2,370 | 0.47 | ~0.90 | worked |
| `qwen38flash` | [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | 360 GB | 4 | 154 / 2,647 | 0.42 | ~0.78 | worked |
| `gemma31b` | [Gemma 4 31B](https://huggingface.co/google/gemma-4-31B) | 62.6 GB | 2 | 80 / 1,656 | 0.34 | ~0.51 | worked |
| `gemma26b` | [Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B) | 51.6 GB | 1 | 207 / 2,600 | 0.11 | ~0.41 | worked |
| `ornith35b` | [Ornith-1.0-35B](https://huggingface.co/ornith-ai/Ornith-1.0-35B) | 70.3 GB | 1 | 220 / 2,714 | 0.10 | ~0.37 | worked |
<!-- END:models-table -->

**GPU-hours per 1M output tokens** is the comparable cost: GPUs ÷ aggregate output tok/s at 64 users ÷ 3,600 × 1,000,000
(lower is cheaper). It is only meaningful under the fixed workload: **every request is 1,024 input + 256 output
tokens** (4 input tokens per output token), 256 requests per run, random-token prompts, seed 0, 64 at a time.
Isambard-AI charges per GPU (1 GPU = 0.25 node-hours per hour). "GPU-hours to reproduce" is download + container
build (shared) + one serve-and-benchmark run.

Everything that affects a result is pinned or recorded (image digest, model commit, benchmark seed and token lengths, the
effective command of each run); see section 7 of the [guide](HOWTO.md#7-the-worked-example-this-repos-scripts).

Unofficial and unaffiliated with Isambard/BriCS or any model vendor. Isambard facts come from the
[Isambard documentation](https://docs.isambard.ac.uk/).
