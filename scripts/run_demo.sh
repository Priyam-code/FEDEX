#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m fedcontext demo --out runs/demo --config configs/demo.yaml --seeds 7 19 43
