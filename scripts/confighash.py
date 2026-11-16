#!/usr/bin/env python3
"""Hash of everything that determines a model's benchmark result. Used to skip re-running unchanged models.

    python3 scripts/confighash.py <model>        # prints a short hash

Covers the model's settings file (repo, pinned commit, layout, vLLM flags), the container image, and RUN_VERSION in
config.sh (bump it when node_entry.sh or the benchmark changes in a way that affects results). Ignores comments and
the safety caps (TIME_LIMIT, STARTUP_DEADLINE_S), which do not change a result. Python 3.6 compatible.
"""
import hashlib, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IGNORED = ("export TIME_LIMIT", "export STARTUP_DEADLINE_S")


def canon(path, keep=None):
    out = []
    for line in open(path):
        s = line.rstrip("\n")
        if not s.strip() or s.lstrip().startswith("#"):
            continue
        if s.startswith(IGNORED):
            continue
        if keep and not s.startswith(keep):
            continue
        out.append(re.sub(r"\s+#.*$", "", s).strip())
    return "\n".join(out)


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else os.environ["MODEL"]
    parts = [canon(os.path.join(ROOT, "scripts", "models", model + ".env")),
             canon(os.path.join(ROOT, "scripts", "config.sh"), keep=("export IMAGE_REF=", "export RUN_VERSION="))]
    print(hashlib.sha256("\n--\n".join(parts).encode()).hexdigest()[:16])


if __name__ == "__main__":
    main()
