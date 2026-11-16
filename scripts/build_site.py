#!/usr/bin/env python3
"""Generate everything that shows results, from one source of truth.

    python3 scripts/build_site.py

Inputs : results/<model>_benchmarks.json (written by collect_results.py), site_src/models.json (descriptions,
         notes), scripts/models/<model>.env (layout, GPUs, time cap).
Outputs: docs/data.js (the stats site), the model table in README.md and HOWTO.md, and the per-model sections
         in HOWTO.md. Nothing is typed in by hand twice.
"""
import json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda *p: os.path.join(ROOT, *p)
REPO_URL = "https://github.com/wd7512/how-to-run-llm-on-isambard"


def env_of(key):
    out = {}
    for line in open(J("scripts", "models", key + ".env")):
        line = line.split("#")[0] if not line.startswith("export VLLM_EXTRA_ARGS") else line
        for m in re.finditer(r'(\w+)=("[^"]*"|\S+)', line):
            out[m.group(1)] = m.group(2).strip('"')
    return out


def load():
    meta = json.load(open(J("site_src", "models.json")))
    models = []
    for key, m in sorted(meta.items(), key=lambda kv: kv[1]["order"]):
        path = J("results", key + "_benchmarks.json")
        if not os.path.exists(path):
            continue
        r = json.load(open(path))
        e = env_of(key)
        top = [x for x in r["results"] if x["concurrency"] == 64][0]
        cm = r["cost_metric"]
        models.append(dict(m, key=key, nodes=int(e["NODES"]), gpus_per_node=int(e["GPUS"]), gpus=int(e["NODES"]) * int(e["GPUS"]),
                           time_cap=e["TIME_LIMIT"], revision=r.get("revision", e.get("MODEL_REVISION")), date=r["date"],
                           layout=r["layout"], software=r["software"], workload=r["workload"], load=r["load"], results=r["results"],
                           cost_metric=cm, gpu_hours=r["gpu_hours"], provenance=r.get("provenance", {}),
                           test_gpuh=cm["run_gpu_hours"], test_s=cm["run_duration_s"], vllm_args=e.get("VLLM_EXTRA_ARGS", ""),
                           repeat_pct=[round((n["output_tok_s"] / o - 1) * 100, 1) for n, o in zip(r["results"], r["previous_run"]["output_tok_s"])] if r.get("previous_run") else None,
                           serve_command=r.get("serve_command"), worker_command=r.get("worker_command"), env=r.get("env", {}),
                           tok64=top["output_tok_s"], tok1=r["results"][0]["output_tok_s"]))
    return models


def fmt(n, d=0):
    return f"{n:,.{d}f}"


def table_md(models):
    rows = ["| `MODEL=` | Model | Weights | GPUs | Output tok/s, 1 / 64 users | GPU-hours to run the benchmark | GPU-hours to reproduce | Status |",
            "|---|---|---:|---:|---:|---:|---:|---|"]
    for m in models:
        rows.append(f"| `{m['key']}` | [{m['label']}](https://huggingface.co/{m['hf']}) | {m['weights_gb']:g} GB | {m['gpus']} | "
                    f"{fmt(m['tok1'], 0 if m['tok1'] >= 10 else 1)} / {fmt(m['tok64'])} | {m['test_gpuh']:.3f} | ~{m['gpu_hours']['total']:.2f} | {m['status']} |")
    return "\n".join(rows)


def sections_md(models):
    out = []
    for m in models:
        cm = m["cost_metric"]
        out.append(f"### {m['key']}\n")
        out.append(f"**{m['label']}** ([{m['hf']}](https://huggingface.co/{m['hf']}), revision `{m['revision'][:12]}`). "
                   f"{re.sub(r'<[^>]+>', '`', m['description']).replace('``', '')}\n")
        out.append("```bash\n"
                   f"MODEL={m['key']} bash scripts/run.sh               # {m['nodes']} node(s) x {m['gpus_per_node']} GPU(s), time cap {m['time_cap']}\n"
                   f"MODEL={m['key']} python3 scripts/collect_results.py   # when the serve job has finished\n```\n")
        if m.get("serve_command"):
            cmd = re.sub(r" (--[a-z0-9.-]+)", r" \\\n  \1", m["serve_command"])
            out.append("What the head node ran (recorded by the run):\n\n```bash\n" + cmd + "\n```\n")
        out.append(f"Measured: {fmt(m['tok1'], 1)} output tok/s for one user, {fmt(m['tok64'])} at 64 users; "
                   f"**{m['test_gpuh']:.3f} GPU-hours to run the benchmark** (the 64-user run: {cm['run_requests']} requests of 1,024 input + 256 output tokens "
                   f"= {cm['run_input_tokens']:,} input / {cm['run_output_tokens']:,} output tokens, {m['test_s']:.0f} s on {m['gpus']} GPU(s)); "
                   f"~{m['gpu_hours']['total']:.2f} GPU-hours to reproduce. Layout: {m['layout']}.\n")
        if m["notes"]:
            out.append("\n".join("- " + re.sub(r"<code>(.*?)</code>", r"`\1`", n) for n in m["notes"]) + "\n")
    return "\n".join(out)


def fit_md(models):
    rows = ["| GPUs | Nodes | Model | Weights per GPU | Concurrent 262k contexts held | Layout |", "|---:|---:|---|---:|---:|---|"]
    for m in sorted(models, key=lambda m: (m["gpus"], m["test_gpuh"])):
        l = m["load"]
        rows.append(f"| {m['gpus']} | {m['nodes']} | [{m['label']}](#{m['key']}) | {l.get('weights_gib_per_gpu', 0):.0f} GiB | "
                    f"{l.get('max_concurrency_at_262144_ctx', 0):.1f} | {m['layout']} |")
    return "\n".join(rows)


def replace_block(path, name, content):
    s = open(path).read()
    a, b = f"<!-- BEGIN:{name} -->", f"<!-- END:{name} -->"
    i, j = s.index(a) + len(a), s.index(b)
    open(path, "w").write(s[:i] + "\n" + content + "\n" + s[j:])


def main():
    models = load()
    data = {"repo": REPO_URL, "models": models}
    open(J("docs", "data.js"), "w").write("// Generated by scripts/build_site.py from results/ and site_src/models.json. Do not edit.\n"
                                          "window.SITE = " + json.dumps(data, indent=1) + ";\n")
    replace_block(J("README.md"), "models-table", table_md(models))
    replace_block(J("HOWTO.md"), "models-table", table_md(models))
    replace_block(J("HOWTO.md"), "fit-table", fit_md(models))
    replace_block(J("HOWTO.md"), "model-sections", sections_md(models))
    print("built", len(models), "models:", ", ".join(m["key"] for m in models))


if __name__ == "__main__":
    main()
