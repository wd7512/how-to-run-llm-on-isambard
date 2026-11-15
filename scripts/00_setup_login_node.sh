#!/bin/bash
# One-off, light-weight setup on the LOGIN node (a pip install is small enough).
# Run:  bash scripts/00_setup_login_node.sh
set -eu
source "$(dirname "$0")/config.sh"
module load cray-python/3.11.7

mkdir -p "$SCRATCHDIR"/{models,containers,envs,tmp} logs
python3 -m venv "$HF_VENV"
"$HF_VENV/bin/pip" install -q huggingface_hub
"$HF_VENV/bin/hf" --help >/dev/null && echo "hf CLI ready in $HF_VENV"

# Sanity check: confirm the repo exists and is the size we expect for the chosen MODEL.
curl -s "https://huggingface.co/api/models/$MODEL_REPO?blobs=true" | python3 -c '
import sys, json
d = json.load(sys.stdin)
s = d["siblings"]
print(d["id"], len(s), "files", round(sum(x.get("size") or 0 for x in s) / 1e9, 1), "GB")'
