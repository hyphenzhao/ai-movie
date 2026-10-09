#!/bin/bash
# Durable driver: short candidates first, then the full film and acceptance.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p "$ROOT/workspace/_cache"
exec 9>"$ROOT/workspace/_cache/v34-regression.lock"
flock -n 9 || { echo "A v3.4 regression is already running"; exit 1; }
exec > >(tee -a "$ROOT/workspace/_cache/v34_regression.log") 2>&1
exec "$ROOT/.venv/bin/python" -u scripts/run_v34_regression.py
