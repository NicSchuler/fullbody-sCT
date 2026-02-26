# TotalSegmentator in Preprocessing

Use step `13run_totalsegmentator.py` to create masks needed later by
`22resample_totalsegmentator_masks.py`.

## Requirements

- `totalsegmentator` python package installed in the active environment
- Input cases in:
  `/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/1initNifti`

## Usage

```bash
python 13run_totalsegmentator.py
```

Optional arguments:

```bash
python 13run_totalsegmentator.py \
  --input-root /path/to/1initNifti \
  --prefix AB_ TH_ HN_ \
  --device gpu
```

## Output

For each processed case and modality (`CT_reg`, `MR`):

`<CASE>/<MODALITY>/totalsegmentator_output/`

Main masks used downstream:
- `liver.nii.gz`
- `torso_fat.nii.gz`
- `subcutaneous_fat.nii.gz`
- `skeletal_muscle.nii.gz`
