#!/usr/bin/env python3
"""Turn a finished serve job into results/<model>_benchmarks.json (no hand-typed numbers).

Run on the login node after the serve job has finished:
    module load cray-python/3.11.7
    MODEL=qwen38 python3 scripts/collect_results.py            # latest run in logs/jobs_<model>.txt
    MODEL=qwen38 python3 scripts/collect_results.py --serve 1234 --download 1200 --build 1201

Reads: logs/run_<serve>/{meta,smoke,c1,c16,c64}.json (written by node_entry.sh), logs/vllm_serve_<serve>.log,
and Slurm accounting (sacct) for GPU-hours. Python 3.6 compatible.
"""
import argparse, json, os, re, subprocess, sys, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")


def gpu_hours(job_ids):
    """GPU-hours per job from sacct: allocated GPUs x elapsed seconds / 3600."""
    out = {}
    ids = [j for j in job_ids if j and j != "none"]
    if not ids:
        return out
    res = subprocess.run(["sacct", "-X", "-P", "-n", "-j", ",".join(ids), "-o", "JobID,ElapsedRaw,AllocTRES,State"],
                         stdout=subprocess.PIPE, universal_newlines=True).stdout
    for line in res.strip().splitlines():
        jid, elapsed, tres, state = line.split("|")
        m = re.search(r"gres/gpu=(\d+)", tres)
        gpus = int(m.group(1)) if m else 0
        out[jid] = {"gpus": gpus, "elapsed_s": int(elapsed), "state": state, "gpu_hours": round(gpus * int(elapsed) / 3600, 3)}
    return out


def grep_log(path, pattern, flags=0):
    rx = re.compile(pattern, flags)
    with open(path, errors="replace") as f:
        for line in f:
            m = rx.search(line)
            if m:
                return m
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL", "dsv41"))
    ap.add_argument("--serve"); ap.add_argument("--download"); ap.add_argument("--build")
    a = ap.parse_args()

    jobs = {}
    jf = os.path.join(LOGS, "jobs_%s.txt" % a.model)
    if os.path.exists(jf):
        last = open(jf).read().strip().splitlines()[-1]
        jobs = dict(kv.split("=") for kv in last.split())
    serve = a.serve or jobs.get("SERVE")
    if not serve:
        sys.exit("no serve job id: pass --serve or run scripts/run.sh first")
    download = a.download or jobs.get("DOWNLOAD"); build = a.build or jobs.get("BUILD")
    run = os.path.join(LOGS, "run_%s" % serve)
    meta = json.load(open(os.path.join(run, "meta.json")))
    results = []
    for c in (1, 16, 64):
        b = json.load(open(os.path.join(run, "c%d.json" % c)))
        results.append({
            "concurrency": c, "requests": b.get("completed", 0) + b.get("failed", 0), "failed": b.get("failed", 0),
            "output_tok_s": round(b["output_throughput"], 2), "mean_ttft_ms": round(b["mean_ttft_ms"], 2),
            "mean_tpot_ms": round(b["mean_tpot_ms"], 2), "total_input_tokens": b.get("total_input_tokens"),
            "total_output_tokens": b.get("total_output_tokens"), "duration_s": round(b.get("duration", 0), 1)})
    slog = os.path.join(LOGS, "vllm_serve_%s.log" % serve)
    m1 = grep_log(slog, r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds")
    m2 = grep_log(slog, r"GPU KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")
    load = {}
    if m1: load.update({"weights_gib_per_gpu": float(m1.group(1)), "load_seconds": round(float(m1.group(2)))})
    if m2: load.update({"kv_cache_tokens": int(m2.group(1).replace(",", "")), "max_concurrency_at_262144_ctx": float(m2.group(3))})
    gpus = meta["nodes"] * meta["gpus_per_node"]
    top = results[-1]
    dur = top["duration_s"]
    smoke = json.load(open(os.path.join(run, "smoke.json")))
    msg = smoke["choices"][0]["message"]
    hours = gpu_hours([download, build, serve])
    steps = {"download": hours.get(download, {}).get("gpu_hours"), "build": hours.get(build, {}).get("gpu_hours"),
             "serve": hours.get(serve, {}).get("gpu_hours")}
    notes = []
    # "GPU-hours to reproduce" must include the real download and the container build, even when this run skipped them.
    prev_path = os.path.join(ROOT, "results", "%s_benchmarks.json" % a.model)
    prev = json.load(open(prev_path)).get("gpu_hours", {}) if os.path.exists(prev_path) else {}
    dl_elapsed = hours.get(download, {}).get("elapsed_s", 0)
    if dl_elapsed < 120 and prev.get("download"):
        steps["download"] = prev["download"]; notes.append("download cost kept from the earlier real download (this run only verified the files)")
    if steps["build"] is None:
        steps["build"] = json.load(open(os.path.join(ROOT, "site_src", "shared_costs.json")))["container_build_gpu_hours"]
        notes.append("container build cost is the measured shared build (this run reused the image)")
    out = {
        "model": meta["repo"], "preset": meta["model"], "revision": meta["revision"], "date": datetime.date.today().isoformat(),
        "system": "Isambard-AI Phase 2, %d x GH200 (96 GiB HBM each), %d node%s" % (gpus, meta["nodes"], "" if meta["nodes"] == 1 else "s"),
        "software": {"vllm": meta["vllm"], "torch": meta["torch"], "cuda": meta["cuda"], "image": meta["image"]},
        "layout": "TP%d x DP%d%s, %d node(s)" % (meta["tp"], meta["dp_per_node"] * meta["nodes"], " + expert-parallel" if meta["ep"] else "", meta["nodes"]),
        "vllm_args": meta["extra_args"], "gpus": gpus, "job_ids": {"download": download, "build": build, "serve": serve},
        "workload": {"tool": "vllm bench serve", "dataset": "random", "input_tokens_per_request": 1024,
                     "output_tokens_per_request": 256, "seed": 0, "requests_per_level": "4 x concurrency"},
        "load": load, "results": results,
        "smoke_reply": {"content": (msg.get("content") or "")[:200], "has_reasoning_field": bool(msg.get("reasoning") or msg.get("reasoning_content"))},
        "cost_metric": {
            "definition": "GPUs / output_tok_s / 3600 * 1e6, from the 64-concurrent-user run",
            "run_requests": top["requests"], "run_input_tokens": top["total_input_tokens"], "run_output_tokens": top["total_output_tokens"],
            "run_duration_s": dur, "gpus": gpus, "run_gpu_hours": round(gpus * dur / 3600, 4),
            "gpu_hours_per_1m_output_tokens": round(gpus / (top["output_tok_s"] * 3600) * 1e6, 3)},
        "gpu_hours": dict(steps, total=round(sum(v for v in steps.values() if v), 2), notes=notes),
    }
    path = os.path.join(ROOT, "results", "%s_benchmarks.json" % a.model)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)
    print("wrote", path)
    print(json.dumps({"results": [(r["concurrency"], r["output_tok_s"]) for r in results], "gpu_hours": out["gpu_hours"],
                      "per_1m": out["cost_metric"]["gpu_hours_per_1m_output_tokens"], "input_tokens_64": top["total_input_tokens"],
                      "output_tokens_64": top["total_output_tokens"]}))


if __name__ == "__main__":
    main()
