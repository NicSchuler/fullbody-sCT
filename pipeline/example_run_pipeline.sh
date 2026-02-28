#!/usr/bin/env bash
set -euo pipefail

python run_pipeline.py \
  --synthrad-data-root /local/scratch/datasets/FullbodySCT \
  --subfolder-name testPipeline \
  --body-part-filter AB \
  --patient-filter 1ABA005 1ABA030 1ABA047 \
  --preprocessing-method 32p99 \
  --method CUT \
  --epochs 50 \
  --batch-size 1 \
  --start 30
