#!/usr/bin/env bash
set -euo pipefail
echo "This runner is deprecated. Please execute the preprocessing step-by-step:"
echo "  1) 10convert_mha_to_nifti.py"
echo "  2) 11sanity_check.py"
echo "  3) 12bodymasks_from_CT.py"
echo "  4) 13run_totalsegmentator.py"
echo "  5) 20resampling.py"
echo "  6) 21datasplit.py (build manifest)"
echo "  7) 22resample_totalsegmentator_masks.py"
echo "  8) 25prepare_for_nyul.py"
echo "  9) 30nyul_train_apply.py"
echo " 10) 40slice_creator.py"
echo " 11) 50_dataset_split.py --use-manifest <manifest.csv> --materialize-dir <out>"
exit 1
