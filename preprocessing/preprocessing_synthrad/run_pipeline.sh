#!/usr/bin/env bash
set -euo pipefail
echo "This runner is deprecated. Please execute the preprocessing step-by-step:"
echo "  1) 10convert_mha_to_nifti.py"
echo "  2) 11sanity_check.py"
echo "  3) 20resampling.py"
echo "  4) 21datasplit.py (build manifest)"
echo "  5) 25prepare_for_nyul.py"
echo "  6) 30nyul_train_apply.py"
echo "  7) 40slice_creator.py"
echo "  8) 50_dataset_split.py --use-manifest <manifest.csv> --materialize-dir <out>"
exit 1
