#!/usr/bin/env bash
set -euo pipefail

# Run from repo root: /Users/flavianthur/Documents/dvh
# Requires MATLAB available as `matlab` in PATH.

# 1) Run DVH pipeline for one model
#matlab -batch "run('matlab/configs/run_model_2_experiment_cut_synthrad_abdomen_32p99.m')"

# 2) Run DVH pipeline for all generated model configs
matlab -batch "run('matlab/configs/run_all_models.m')"

# 3) Optional: export RTDOSE DICOM from matRad workspaces
python3 scripts/export_rtdose_from_matrad.py \
  --results-root outputs/dvh_results \
  --dicom-root outputs/dicom
