#!/usr/bin/env python3
"""CPU-only checks, run before any GPU job is submitted (GPU-hours are not free).

    MODEL=<name> python3 scripts/preflight.py        # reads the settings that scripts/config.sh exports

Checks: the pinned revision exists; weights fit in scratch; weights fit per GPU (with room for KV cache); the GPU layout
is self-consistent; the model's attention heads divide by TP; the time caps make sense; and, if the container image exists,
that vLLM knows the architecture, the parser and attention-backend names, and accepts every flag. Exits 1 on FAIL.
Python 3.6 compatible; the vLLM checks run inside the container (CPU only, no --nv).
"""
import json, os, re, shutil, subprocess, sys, urllib.request

E = os.environ
HBM_GIB, USABLE = 96.0, 0.90
fails, warns = [], []


def ok(msg): print("  ok    " + msg)
def warn(msg): warns.append(msg); print("  WARN  " + msg)
def fail(msg): fails.append(msg); print("  FAIL  " + msg)


def get(url):
    return urllib.request.urlopen(url, timeout=30).read().decode()


def find_all(obj, key, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key and isinstance(v, int): out.append(v)
            find_all(v, key, out)
    elif isinstance(obj, list):
        for v in obj: find_all(v, key, out)


def main():
    repo, rev, model = E["MODEL_REPO"], E["MODEL_REVISION"], E["MODEL"]
    nodes, gpus, tp, dp, ep = int(E["NODES"]), int(E["GPUS"]), int(E["TP"]), int(E["DP_PER_NODE"]), E["EP"] == "1"
    print("preflight %s (%s @ %s)" % (model, repo, rev[:12]))
    if E.get("KNOWN_BROKEN") and E.get("FORCE") != "1":
        fail("known not to work: " + E["KNOWN_BROKEN"] + " (FORCE=1 to try anyway)")

    # 1. revision + size
    size = 0
    try:
        tree = json.loads(get("https://huggingface.co/api/models/%s/tree/%s?recursive=1" % (repo, rev)))
        size = sum(f.get("size", 0) for f in tree if f["type"] == "file")
        ok("revision exists: %.1f GB in %d files" % (size / 1e9, len(tree)))
    except Exception as e:
        fail("cannot read revision %s of %s from Hugging Face (%s)" % (rev[:12], repo, e))
    gib = size / 2 ** 30

    # 2. scratch space
    free = shutil.disk_usage(E.get("SCRATCHDIR", "/")).free / 2 ** 30
    if size and free < gib * 1.1: fail("scratch has %.0f GiB free, the weights need %.0f GiB" % (free, gib))
    elif size: ok("scratch has %.0f GiB free for %.0f GiB of weights" % (free, gib))

    # 3. layout and memory
    if gpus != tp * dp: fail("GPUS=%d but TP x DP_PER_NODE = %d" % (gpus, tp * dp))
    else: ok("layout: %d node(s) x %d GPU(s) = TP %d x DP %d per node" % (nodes, gpus, tp, dp))
    if size:
        total = nodes * gpus
        per_gpu = gib / total if ep else gib / tp          # expert-parallel spreads experts over every GPU
        what = "weights / all %d GPUs (expert-parallel)" % total if ep else "weights / TP %d" % tp
        if per_gpu > HBM_GIB * USABLE: fail("%.0f GiB per GPU (%s) is more than the %.0f GiB vLLM may use" % (per_gpu, what, HBM_GIB * USABLE))
        elif per_gpu > 55: warn("%.0f GiB of weights per GPU (%s) leaves little KV cache; check 'Maximum concurrency' for MAX_LEN %s" % (per_gpu, what, E["MAX_LEN"]))
        else: ok("%.0f GiB of weights per GPU (%s)" % (per_gpu, what))

    # 4. heads / dims divide by TP (warnings: some models handle uneven splits)
    try:
        cfg = json.loads(get("https://huggingface.co/%s/raw/%s/config.json" % (repo, rev)))
        heads, kv, mpos = [], [], []
        find_all(cfg, "num_attention_heads", heads); find_all(cfg, "num_key_value_heads", kv); find_all(cfg, "max_position_embeddings", mpos)
        if heads and any(h % tp for h in heads): warn("attention heads %s are not all divisible by TP %d" % (sorted(set(heads)), tp))
        if kv and any(k % tp and tp % k for k in kv): warn("KV heads %s neither divide nor are divided by TP %d" % (sorted(set(kv)), tp))
        if heads and not any(h % tp for h in heads): ok("attention heads %s divisible by TP %d" % (sorted(set(heads)), tp))
        if mpos and int(E["MAX_LEN"]) > max(mpos): fail("MAX_LEN %s exceeds the model's max_position_embeddings %d" % (E["MAX_LEN"], max(mpos)))
    except Exception as e:
        warn("could not read config.json (%s)" % e)

    # 5. caps
    t = [int(x) for x in E.get("TIME_LIMIT", "00:30:00").split(":")]
    cap, dl = t[0] * 3600 + t[1] * 60 + t[2], int(E.get("STARTUP_DEADLINE_S", "1200"))
    if dl >= cap: fail("STARTUP_DEADLINE_S %d is not shorter than TIME_LIMIT %d s" % (dl, cap))
    else: ok("startup deadline %d s, job time cap %d s (%.2f GPU-hours worst case)" % (dl, cap, cap * nodes * gpus / 3600))

    # 6. vLLM knows the model, the parsers and the flags (inside the container, CPU only)
    sif = E.get("SIF", "")
    if not os.path.exists(sif):
        warn("container image not built yet; skipping the vLLM checks (the build job checks the architecture)")
    else:
        code = r'''
import json, os, shlex, sys
res = {}
from vllm.model_executor.models.registry import ModelRegistry
res["arch"] = os.environ["ARCH"] in ModelRegistry.get_supported_archs()
args = shlex.split(os.environ.get("VLLM_EXTRA_ARGS", ""))
try:
    from vllm.reasoning import ReasoningParserManager as R
    from vllm.tool_parsers import ToolParserManager as T
    from vllm.v1.attention.backends.registry import AttentionBackendEnum as B
    bad = []
    for flag, names in (("--reasoning-parser", set(R.list_registered())), ("--tool-call-parser", set(T.list_registered())), ("--attention-backend", {e.name for e in B})):
        if flag in args and args[args.index(flag) + 1] not in names: bad.append("%s %s" % (flag, args[args.index(flag) + 1]))
    res["bad_names"] = bad
except Exception as e:
    res["names_error"] = str(e)[:150]
try:
    import dataclasses
    from vllm.engine.arg_utils import EngineArgs
    names = {"--" + f.name.replace("_", "-") for f in dataclasses.fields(EngineArgs)}
    complete = True
    try:
        from vllm.entrypoints.common.cli_args import FrontendArgs
        names |= {"--" + f.name.replace("_", "-") for f in dataclasses.fields(FrontendArgs)}
    except Exception:
        complete = False        # frontend flags (parsers etc.) not enumerable: only report flags that are clearly wrong
    unknown = []
    for a in args:
        if a.startswith("--"):
            base = a.split(".")[0]
            base = "--" + base[5:] if base.startswith("--no-") and ("--" + base[5:]) in names else base
            if base not in names: unknown.append(a)
    # A flag that is not an option may still be a legacy alias handled elsewhere: if its text appears anywhere in the
    # installed vLLM it is only a warning; if it appears nowhere it is certainly wrong.
    import vllm
    root = os.path.dirname(vllm.__file__)
    blob = ""
    for d, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".py"):
                try: blob += open(os.path.join(d, f), errors="ignore").read()
                except Exception: pass
    res["unknown_flags"] = [a for a in unknown if a not in blob]
    res["alias_flags"] = [a for a in unknown if a in blob]
    res["flags_complete"] = complete
except Exception as e:
    res["flags_error"] = str(e)[:150]
print("PREFLIGHT_JSON " + json.dumps(res))
'''
        r = subprocess.run(["apptainer", "exec", sif, "python3", "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                           env=dict(E, VLLM_TARGET_DEVICE="cuda"))   # no GPU on the login node: skip device probing
        line = [l for l in r.stdout.splitlines() if l.startswith("PREFLIGHT_JSON ")]
        if not line: warn("could not run the vLLM checks in the container: " + (r.stderr.strip().splitlines() or ["?"])[-1][:200])
        else:
            res = json.loads(line[0][len("PREFLIGHT_JSON "):])
            ok("vLLM supports architecture %s" % E["ARCH"]) if res.get("arch") else fail("vLLM does not list architecture %s" % E["ARCH"])
            if res.get("bad_names"): fail("unknown parser/backend name: " + ", ".join(res["bad_names"]))
            elif "names_error" in res: warn("could not check parser names (%s)" % res["names_error"])
            else: ok("parser and attention-backend names are valid")
            if "flags_error" in res: warn("could not check the flags (%s)" % res["flags_error"])
            elif res.get("unknown_flags"):
                (fail if res.get("flags_complete") else warn)("flag(s) vLLM does not recognise anywhere: %s" % " ".join(res["unknown_flags"]))
            elif res.get("alias_flags"):
                warn("flag(s) %s are not engine options but appear in vLLM (legacy alias?); the model that uses them has run" % " ".join(res["alias_flags"]))
            else: ok("every flag in VLLM_EXTRA_ARGS is a vLLM option")

    print("preflight: %d FAIL, %d WARN" % (len(fails), len(warns)))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
