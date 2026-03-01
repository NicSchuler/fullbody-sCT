#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/run_pipeline.py" \
  --synthrad-data-root /local/scratch/datasets/FullbodySCT \
  --subfolder-name testPipeline \
  --body-part-filter AB \
  --patient-filter 1ABA005 1ABA030 1ABA047 \
  --preprocessing-method 34normalized_n4_centerspecific_03LIC \
  --method CUT \
  --epochs 50 \
  --batch-size 1 \
  --start 30
