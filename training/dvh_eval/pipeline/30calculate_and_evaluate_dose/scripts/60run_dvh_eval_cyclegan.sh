#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_ROOT="$PROJECT_ROOT/outputs/dvh_results"
MODEL_REGEX='^2_experiment_cyclegan_abdomen_'

MODELS=()
while IFS= read -r m; do
  MODELS+=("$m")
done < <(
  find "$RESULTS_ROOT" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; \
    | grep -E "$MODEL_REGEX" \
    | sort
)

if [ "${#MODELS[@]}" -eq 0 ]; then
  echo "[ERROR] No matching models found in $RESULTS_ROOT for regex: $MODEL_REGEX" >&2
  exit 1
fi

ARGS=()
for m in "${MODELS[@]}"; do
  ARGS+=(--model "$m")
done

echo "[INFO] Running DVH evaluation for CycleGAN models: ${MODELS[*]}"
python3 "$PROJECT_ROOT/scripts/60run_dvh_evaluation.py" "${ARGS[@]}"
