#!/usr/bin/env bash
set -euo pipefail

# Minimal one-shot pipeline runner.
# Edit the variables below as needed and run:
#   bash run_pipeline.sh

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Paths for the split stage (uses outputs from 40slice_creator)
SLICES_ROOT="/local/scratch/datasets/FullbodySCT/SynthRAD2025/task1_backup/5slicesOutputForModels"
MANIFEST="/local/scratch/datasets/FullbodySCT/SynthRAD2025/task1_backup/splits_manifest.csv"
LIST_DIR="/local/scratch/datasets/FullbodySCT/SynthRAD2025/task1_backup/split_lists"
MATERIALIZE_DIR="/local/scratch/datasets/FullbodySCT/SynthRAD2025/task1_backup/materialized_splits"

echo "[1/7] Convert to NIfTI"
python "$BASE_DIR/10convert_mha_to_nifti.py"

echo "[2/7] Sanity check"
python "$BASE_DIR/11sanity_check.py"

echo "[3/7] Resampling"
python "$BASE_DIR/20resampling.py"

echo "[4/7] Prepare for Nyul"
python "$BASE_DIR/25prepare_for_nyul.py"

echo "[5/7] Nyul normalization"
bash "$BASE_DIR/30nyul_run.sh"

echo "[6/7] Create slices (unified A/B)"
python "$BASE_DIR/40slice_creator.py"

echo "[7/7] Stratified split + materialize for Pix2Pix and CycleGAN"
python "$BASE_DIR/50_dataset_split.py" \
  --input-root "$SLICES_ROOT" \
  --ratios 0.7 0.15 0.15 \
  --seed 42 \
  --out-manifest "$MANIFEST" \
  --out-list-dir "$LIST_DIR" \
  --materialize-dir "$MATERIALIZE_DIR" \
  --mode both

echo "[done] Pipeline finished."
