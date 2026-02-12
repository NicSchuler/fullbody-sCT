# DVH Evaluation Workflow (Test Images)

This folder provides a reproducible DVH dosimetry workflow for test reconstructions using matRad.

## Download-focused setup

This flow prepares download-ready NIfTI data under:

`/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/11dvhEvalCases`

Optional helper to pick patients:

```bash
python training/dvh_evl/pipeline/05collect_abdomen_test_middle_slices.py --clean
```

Run the 3 steps:

```bash
# Step 10: create test_patients_shared/<patient> with CT + reference_grid.json
python training/dvh_evl/pipeline/10choose_reference_grid.py \
  --ct-root /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/2resampledNifti_reconstructed_dims \
  --export-root /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/11dvhEvalCases \
  --patients PATIENT_A PATIENT_B PATIENT_C

# Step 20: create TS_CT masks per shared patient
python training/dvh_evl/pipeline/20run_totalsegmentator_on_ct.py \
  --export-root /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/11dvhEvalCases \
  --device gpu

# Step 25: create per-model sCT folders for the same patients
python training/dvh_evl/pipeline/25prepare_nifti_export_for_slicer.py \
  --sct-base /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/9latestTestImages \
  --export-root /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/11dvhEvalCases \
  --model-name 2_experiment_cut_synthrad_abdomen_32p99 \
  --epoch 50
```

Expected folder layout:

```text
11dvhEvalCases/
  test_patients_shared/
    <patient>/
      CT/
        <patient>_CT.nii.gz
      TS_CT/
        liver.nii.gz
        kidney_left.nii.gz
        kidney_right.nii.gz
        spinal_cord.nii.gz
        skin.nii.gz
      reference_grid.json

  <model_name>/
    <patient>/
      <patient>_sCT.nii.gz
```

Notes:
- Step 25 supports `--model-name` and `--epoch` to target `9latestTestImages/<model_name>/test_<epoch>/reconstruction`.
- If no model is provided, step 25 auto-discovers models that contain `test_<epoch>/reconstruction`.
- Use `--model-names` in step 25 for multiple explicit models.
- Use `--patients` or `--patients-file` on steps 10/20/25 to control the exact patient subset.