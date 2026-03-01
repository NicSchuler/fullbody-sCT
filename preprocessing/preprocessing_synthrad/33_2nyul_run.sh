#!/usr/bin/env bash
set -euo pipefail

BASE_ROOT="${BASE_ROOT:-/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/33nyul}"
NYUL_READY="${NYUL_READY:-${BASE_ROOT}/3_1NiftiNyulReady}"
TRAIN_ROOT="${TRAIN_ROOT:-${NYUL_READY}/trainingforcalc}"
VALTEST_ROOT="${VALTEST_ROOT:-${NYUL_READY}/valtest}"
OUT_ROOT="${OUT_ROOT:-${BASE_ROOT}/3_2normalized}"
MODEL_PATH="${MODEL_PATH:-${BASE_ROOT}/nyul_model_params.npy}"
NYUL_NORMALIZE_BIN="${NYUL_NORMALIZE_BIN:-nyul-normalize}"

mkdir -p "${OUT_ROOT}"


echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "PATH=$PATH"
which python || true
which nyul-normalize || true

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
