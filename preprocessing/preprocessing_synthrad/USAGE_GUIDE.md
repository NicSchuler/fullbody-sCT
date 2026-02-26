# Quick Reference: Running the Preprocessing Pipeline

## Complete Pipeline for One Normalization Method

### Example: Using Per-File P99 Normalization (32p99)

```bash
# Step 1: Resampling (run once, creates non-normalized data)
python 20resampling.py
# Output: 2resampledNifti/

# Step 1b: TotalSegmentator masks on init NIfTI (required by step 22 / npeaks)
python 13run_totalsegmentator.py
# Output: 1initNifti/<CASE>/<MODALITY>/totalsegmentator_output/

# Step 2: Resample TotalSegmentator masks to 2resampledNifti
python 22resample_totalsegmentator_masks.py
# Output: 2resampledNifti/<CASE>/totalsegmentator_masks/

# Step 3: Apply normalization (Abdomen only)
python 32perfile_p99_standardization.py
# Output: 32p99/3normalized/ (Abdomen cases only)

# Step 4: Create 2D slices
python 40slice_creator.py 32p99
# Output: 32p99/5slices/

# Step 5: Split into train/val/test
python 50_split_folderstructure.py 32p99
# Output: 32p99/6materialized_splits/

# Step 6: Create A+B combined images for pix2pix
python 60combine_A_B_for_pix2pix.py 32p99
# Output: 32p99/6materialized_splits/pix2pix/AB/

# Step 7: Create body-region subsets
python 70create_subsets_per_body_region.py 32p99
# Output: 32p99/7materialized_splits_BodyRegion/
```

## Running for All Normalization Methods

```bash
# Step 1: Resampling (run once)
python 20resampling.py

# Step 1b: TotalSegmentator masks on init NIfTI (run once)
python 13run_totalsegmentator.py

# Step 2: Resample TotalSegmentator masks (run once)
python 22resample_totalsegmentator_masks.py

# Step 3-7: For each normalization method
for method in 31baseline 32p99 33nyul 34npeaks; do
    echo "Processing normalization method: $method"
    
    # Standardization
    python ${method}_standardization.py
    
    # Create slices
    python 40slice_creator.py $method
    
    # Split dataset
    python 50_split_folderstructure.py $method
    
    # Combine A+B for pix2pix
    python 60combine_A_B_for_pix2pix.py $method
    
    # Create body region subsets
    python 70create_subsets_per_body_region.py $method
    
    echo "Completed $method"
    echo "---"
done
```

## Individual Script Usage

### 31-34: Standardization Scripts
```bash
# 31baseline: supports --all-data flag for all body regions
python 31baseline_standardization.py              # Abdomen only (default)
python 31baseline_standardization.py --all-data   # All body regions

# 32p99, 33nyul, 34npeaks: process ABDOMEN only (AB_* prefix)
python 32perfile_p99_standardization.py  # Abdomen only
bash 33nyul_run.sh  # Note: shell script - Abdomen only
python 34npeaks_standardization.py  # Abdomen only
```

### 40: Slice Creator
```bash
# Syntax: python 40slice_creator.py [method]
python 40slice_creator.py 31baseline
python 40slice_creator.py 32p99
python 40slice_creator.py 33nyul
python 40slice_creator.py 34npeaks
```

### 50: Split Folder Structure
```bash
# Syntax: python 50_split_folderstructure.py [method]
python 50_split_folderstructure.py 31baseline
python 50_split_folderstructure.py 32p99
python 50_split_folderstructure.py 33nyul
python 50_split_folderstructure.py 34npeaks

# Or with manual path override:
python 50_split_folderstructure.py \
    --slices-root /path/to/32p99/5slices \
    --out-dir /path/to/32p99/6materialized_splits
```

### 60: Combine A+B for Pix2Pix
```bash
# Syntax: python 60combine_A_B_for_pix2pix.py [method]
python 60combine_A_B_for_pix2pix.py 31baseline
python 60combine_A_B_for_pix2pix.py 32p99
python 60combine_A_B_for_pix2pix.py 33nyul
python 60combine_A_B_for_pix2pix.py 34npeaks
```

### 70: Create Body Region Subsets
```bash
# Syntax: python 70create_subsets_per_body_region.py [method]
python 70create_subsets_per_body_region.py 31baseline
python 70create_subsets_per_body_region.py 32p99
python 70create_subsets_per_body_region.py 33nyul
python 70create_subsets_per_body_region.py 34npeaks
```

## Output Directory Structure

After running the complete pipeline for all methods:

```
Synthrad_combined_preprocessed/
├── 1initNifti/                              # Raw data
├── 2resampledNifti/                         # Resampled (not normalized)
│
├── 31baseline/                              # Baseline normalization outputs
│   ├── 3normalized/                         # Normalized data
│   ├── 5slices/                             # 2D slices
│   ├── 6materialized_splits/                # Train/val/test splits
│   │   ├── pix2pix/
│   │   │   ├── train/{A,B}
│   │   │   ├── val/{A,B}
│   │   │   ├── test/{A,B}
│   │   │   └── AB/{train,val,test}          # Combined A+B images
│   │   └── cyclegan/
│   │       ├── train/{trainA,trainB}
│   │       ├── val/{valA,valB}
│   │       └── test/{testA,testB}
│   └── 7materialized_splits_BodyRegion/     # Body region subsets
│       ├── AB/  (abdomen)
│       ├── HN/  (head & neck)
│       ├── TH/  (thorax)
│       └── ...
│
├── 32p99/                                   # P99 normalization outputs
│   ├── 3normalized/
│   ├── 5slices/
│   ├── 6materialized_splits/
│   └── 7materialized_splits_BodyRegion/
│
├── 33nyul/                                  # Nyul normalization outputs
│   ├── 3normalized/
│   ├── 5slices/
│   ├── 6materialized_splits/
│   └── 7materialized_splits_BodyRegion/
│
└── 34npeaks/                                # N-peaks normalization outputs
    ├── 3normalized/
    ├── 5slices/
    ├── 6materialized_splits/
    └── 7materialized_splits_BodyRegion/
```

## Notes

- **Default method**: If you don't specify a method, scripts default to `32p99`
- **TotalSegmentator placement**: Run `13run_totalsegmentator.py` after init NIfTI creation and before `22resample_totalsegmentator_masks.py`
- **Abdomen filtering**: All normalization methods (32p99, 33nyul, 34npeaks) process **abdomen data only** (AB_* prefix)
  - Exception: `31baseline` has `--all-data` flag to process all body regions
- **Run once**: `20resampling.py` only needs to be run once
- **Parallel processing**: Different normalization methods can be run in parallel
- **Method validation**: Scripts will validate that the normalization method is valid
- **Auto-configuration**: Scripts automatically configure input/output paths based on the normalization method
- **Error checking**: Scripts check if required input directories exist before running


# Postprocessing

## 80: Volume Creator
Reconstructs 3D NIfTI volumes from 2D NIfTI slices generated during testing. Stacks slices at 256x256 resolution and optionally reconstructs volumes at original patient-specific dimensions through reverse resampling.

```bash
# Basic usage: reconstruct sCT volumes at both 256x256 and original dimensions
python ./80volume_creator.py pix2pix_synthrad_allregions

# With all options: copy original CT/MR files and create validation volumes
python ./80volume_creator.py pix2pix_synthrad_allregions --copy_originals --reconstruct_original
```

**Output files** (per patient):
- `sCT_256_dim.nii.gz` - Reconstructed sCT at 256x256 resolution
- `sCT_original_dim_mask_not_aligned.nii.gz` - sCT at original dimensions
- With `--copy_originals`: Original CT/MR files from both resampled and initial data
- With `--reconstruct_original`: Validation volume showing CT after preprocessing round-trip

**Important**: Volumes at original dimensions are NOT aligned with original masks. Safe for dosimetric analysis, but not for image-level comparison with masked regions.
