#!/usr/bin/env bash
set -euo pipefail

# Run everything from this folder so the local .env and config.yaml are used.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# .env is only for secrets such as OPENROUTER_API_KEY.
if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi

read_config_flag() {
  # Ask Python to read config.yaml so bash does not need YAML parsing logic.
  python - "$1" <<'PY'
import sys
from utils import load_project_config

config = load_project_config()
experiment = config.get("experiment", {})
value = experiment.get(sys.argv[1], config.get(sys.argv[1], 0))
print(str(value))
PY
}

RUN_INFERENCE="$(read_config_flag run_inference)"
RUN_EVALUATION="$(read_config_flag run_evaluation)"

# Keep shell logic tiny: config decides which Python entrypoints to run.
if [[ "$RUN_INFERENCE" == "1" ]]; then
  python main.py
fi

if [[ "$RUN_EVALUATION" == "1" ]]; then
  python evaluation.py
fi
