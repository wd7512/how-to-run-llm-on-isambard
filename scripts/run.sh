#!/bin/bash
# ONE command from a login node: setup -> download -> build container -> serve + benchmark.
# Steps are chained with Slurm dependencies, so nothing needs babysitting. Already-done steps are skipped.
#   MODEL=qwen38 bash scripts/run.sh          # then: MODEL=qwen38 python3 scripts/collect_results.py
set -eu
cd "$(dirname "$0")/.."
source scripts/config.sh
mkdir -p logs

bash scripts/00_setup_login_node.sh >/dev/null

DEPS=""; DL=none; BD=none
if [ "$(cat "$MODEL_DIR/.revision" 2>/dev/null || true)" != "$MODEL_REVISION" ]; then
  DL=$(sbatch --parsable --job-name="dl-$MODEL" scripts/01_download.sbatch)
  DEPS="afterok:$DL"
fi
if [ ! -f "$SIF" ]; then
  BD=$(sbatch --parsable scripts/02_build_container.sbatch)
  DEPS="${DEPS:+$DEPS,}afterok:$BD"
fi
SV=$(DEPENDENCY="$DEPS" bash scripts/03_serve.sh)

echo "MODEL=$MODEL DOWNLOAD=$DL BUILD=$BD SERVE=$SV" | tee -a "logs/jobs_$MODEL.txt"
echo "Watch:  squeue -u \$USER ; tail -f logs/serve-${MODEL}_${SV}.out"
echo "Then:   MODEL=$MODEL python3 scripts/collect_results.py"
