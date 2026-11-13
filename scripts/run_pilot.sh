#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
pilot_python="${PYTHON:-python}"
if [[ -x .venv/bin/python ]]; then pilot_python=.venv/bin/python; fi
"$pilot_python" -m mrrate_vlm --config config/batch26.yaml run \
  --batches "${1:-26}" --source-batches "${1:-26}" --max-studies "${2:-30}" \
  --max-total-gb "${3:-10}" \
  --split "${4:-validation}" "${@:5}"
