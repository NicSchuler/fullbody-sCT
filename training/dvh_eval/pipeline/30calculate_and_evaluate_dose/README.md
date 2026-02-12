# DVH Evaluation Pipeline (NIfTI -> DICOM -> MATLAB)

This repository contains a practical pipeline for:
1. Converting NIfTI files from `11dvhEvalCases/` into DICOM.
2. Running dosimetric / DVH evaluation in MATLAB using a local self-contained evaluator based on matRad.

## Folder assumptions

Input layout (already present in this repo):
- `11dvhEvalCases/test_patients_shared/<CASE>/CT/<CASE>_CT.nii.gz`
- `11dvhEvalCases/test_patients_shared/<CASE>/TS_CT/*.nii.gz` (structure masks)
- `11dvhEvalCases/<MODEL>/<CASE>/<CASE>_sCT.nii.gz` (and additional model folders over time)

Generated output layout:
- `outputs/dicom/<MODEL>/<CASE>/CT/*.dcm`
- `outputs/dicom/<MODEL>/<CASE>/sCT/*.dcm`
- `outputs/dicom/<MODEL>/<CASE>/RTSTRUCT/rtstruct.dcm` (if enabled)
- `outputs/dvh_results/<MODEL>/<CASE>/...` (recommended MATLAB stage layout)

## 1) Automated NIfTI -> DICOM conversion

## Install dependencies

```bash
pip install SimpleITK
# Optional (needed only for RTSTRUCT generation from TS_CT masks):
pip install nibabel rt-utils
```

## Run conversion

```bash
python3 scripts/nifti_to_dicom.py \
  --data-root 11dvhEvalCases \
  --out-root outputs/dicom \
  --build-rtstruct
```

Useful options:
- Convert only one case:
  ```bash
  python3 scripts/nifti_to_dicom.py --case AB_1ABA068
  ```
- Convert only selected model folders (repeatable):
  ```bash
  python3 scripts/nifti_to_dicom.py --model 2_experiment_cut_synthrad_abdomen_32p99
  ```

Notes:
- CT and sCT are exported as DICOM CT image series.
- The script automatically processes every subfolder in `11dvhEvalCases/` except `test_patients_shared`.
- `--build-rtstruct` creates one RTSTRUCT from `TS_CT/*.nii.gz` masks.
- If RTSTRUCT creation fails due missing packages, install `nibabel` and `rt-utils`.
- `reference_grid.json` is used to validate CT/sCT geometry (shape, spacing, origin, direction) before writing DICOM.

## 2) MATLAB DVH analysis stage

## Clone required repo (local)

```bash
git clone https://github.com/e0404/matRad.git
```

Use the `dev` branch for matRad if needed by your setup:
```bash
cd matRad && git checkout dev
```

MATLAB requirement:
- `matRad` DICOM import requires the MATLAB **Image Processing Toolbox**.

## Run MATLAB wrapper

Files in this repo:
- `matlab/run_dvh_pipeline.m`
- `matlab/example_config.m`
- `matlab/run_dvh_eval_local_photons.m`
- `matlab/matRad_importDicom_local.m`
- `matlab/matRad_createCst_local.m`

Workflow:
1. Edit `matlab/example_config.m` and set:
   - `cfg.matRadRoot`
   - `cfg.dicomRoot` to one model folder, e.g. `outputs/dicom/2_experiment_cut_synthrad_abdomen_32p99`
   - `cfg.targetRoiName` (for your use case: `kidney_left` or `kidney_right`)
2. Start MATLAB in this repository root.
3. Run:

```matlab
run('matlab/example_config.m')
```

What the wrapper currently does:
- Adds matRad to MATLAB path.
- Iterates all case IDs.
- Checks expected DICOM inputs.
- Imports CT + RTSTRUCT for each case (from your generated DICOM output).
- Uses `11dvhEvalCases/test_patients_shared/<CASE>/reference_grid.json` for input voxel spacing (no MATLAB Image Processing Toolbox required for this step).
- Optimizes photon plan on CT (target ROI = `cfg.targetRoiName`).
- Recalculates dose on sCT with the same optimized beam weights.
- Computes DVH/QI metrics and exports CT, sCT, and delta tables.
- Stores per-case:
  - `dvh_eval.mat`
  - `pipeline_params.mat`
  - `matrad_workspace.mat`
  - `dvh_metrics_ct.csv`, `dvh_metrics_sct.csv`, `dvh_metrics_delta.csv`

## Quick end-to-end checklist

1. Convert NIfTI to DICOM (automated script or Slicer manual).
2. Verify `outputs/dicom/<MODEL>/<CASE>/CT`, `sCT`, and `RTSTRUCT` exist.
3. Configure external repo paths in `matlab/example_config.m` and set `cfg.dicomRoot` to the model you want to evaluate.
4. Run MATLAB pipeline wrapper.
5. Check exported metrics in `outputs/dvh_results/<MODEL>/<CASE>/`.

## 3) Export RTDOSE DICOM (for Slicer)

You can export dose cubes from `matrad_workspace.mat` into DICOM RTDOSE files:

Requirements:
```bash
pip install pydicom h5py numpy
```

```bash
python3 scripts/export_rtdose_from_matrad.py \
  --results-root outputs/dvh_results \
  --dicom-root outputs/dicom
```

Per case, this writes:
- `outputs/dvh_results/<MODEL>/<CASE>/dose_ct.dcm`
- `outputs/dvh_results/<MODEL>/<CASE>/dose_sct.dcm`

Optional filters:
- one model:
  ```bash
  python3 scripts/export_rtdose_from_matrad.py --model 2_experiment_cut_synthrad_abdomen_32p99
  ```
- one case:
  ```bash
  python3 scripts/export_rtdose_from_matrad.py --case AB_1ABA068
  ```

## 4) Notebook DVH Comparison (per model)

Use notebook:
- `notebooks/dvh_model_comparison.ipynb`

Reusable Python entrypoint:
- `dvh_analysis.py` -> `run_model_analysis(model_name, ...)`

Minimal call (1 line after import):
```python
from dvh_analysis import run_model_analysis
result = run_model_analysis('2_experiment_cut_synthrad_abdomen_32p99')
```

Outputs are written to:
- `outputs/dvh_analysis/<MODEL>/dvh_combined_metrics.csv`
- `outputs/dvh_analysis/<MODEL>/dvh_summary_by_roi.csv`
- `outputs/dvh_analysis/<MODEL>/dvh_summary_by_case.csv`
- plot PNGs for each metric (delta boxplot, CT-vs-sCT scatter, case-level PTV/OAR mean deltas)
