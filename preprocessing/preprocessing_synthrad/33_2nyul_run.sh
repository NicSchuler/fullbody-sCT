#!/usr/bin/env bash
set -euo pipefail

BASE_ROOT=/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/33nyul
NYUL_READY=${BASE_ROOT}/25NiftiNyulReady
TRAIN_ROOT=${NYUL_READY}/trainingforcalc
VALTEST_ROOT=${NYUL_READY}/valtest
OUT_ROOT=${BASE_ROOT}/3normalized
MODEL_PATH=${BASE_ROOT}/nyul_model_params.npy

mkdir -p "${OUT_ROOT}"

echo "Fitting Nyul mapping on training set..."
nyul-normalize \
  "${TRAIN_ROOT}/MR" \
  -m "${TRAIN_ROOT}/masks" \
  -o "${OUT_ROOT}" \
  -v \
  --output-max-value 1 \
  --output-min-value 0 \
  --min-percentile 2 \
  --max-percentile 98 \
  -ssh "${MODEL_PATH}"

echo "Applying Nyul mapping to val+test..."
nyul-normalize \
  "${VALTEST_ROOT}/MR" \
  -m "${VALTEST_ROOT}/masks" \
  -o "${OUT_ROOT}" \
  -v \
  --output-max-value 1 \
  --output-min-value 0 \
  --min-percentile 2 \
  --max-percentile 98 \
  -lsh "${MODEL_PATH}"

echo "Finished at:     $(date)"
