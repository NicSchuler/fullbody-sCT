# Quick Reference: Running the Preprocessing Pipeline

## Complete Pipeline for One Normalization Method

### Example: Using Per-File P99 Normalization (32p99)

```bash
# Step 1: Resampling (run once, creates non-normalized data)
python 20resampling.py
# Output: 2resampledNifti/

# Step 2: Apply normalization
python 32perfile_p99_standardization.py
# Output: 3normalized_32p99/

# Step 3: Create 2D slices
python 40slice_creator.py 32p99
# Output: 5slices_32p99/

# Step 4: Split into train/val/test
python 50_split_folderstructure.py 32p99
# Output: 6materialized_splits_32p99/

# Step 5: Create A+B combined images for pix2pix
python 60combine_A_B_for_pix2pix.py 32p99
# Output: 6materialized_splits_32p99/pix2pix/AB/

# Step 6: Create body-region subsets
python 70create_subsets_per_body_region.py 32p99
# Output: 7materialized_splits_32p99BodyRegion/
```

## Running for All Normalization Methods

```bash
# Step 1: Resampling (run once)
python 20resampling.py

# Step 2-6: For each normalization method
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
python 31baseline_standardization.py
python 32perfile_p99_standardization.py
bash 33nyul_run.sh  # Note: shell script
python 34npeaks_standardization.py
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
    --slices-root /path/to/5slices_32p99 \
    --out-dir /path/to/6materialized_splits_32p99
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
├── 3normalized_31baseline/                  # Normalized data
├── 3normalized_32p99/
├── 3normalized_33nyul/
├── 3normalized_34npeaks/
│
├── 5slices_31baseline/                      # 2D slices
├── 5slices_32p99/
├── 5slices_33nyul/
├── 5slices_34npeaks/
│
├── 6materialized_splits_31baseline/         # Train/val/test splits
├── 6materialized_splits_32p99/
│   ├── pix2pix/
│   │   ├── train/{A,B}
│   │   ├── val/{A,B}
│   │   ├── test/{A,B}
│   │   └── AB/{train,val,test}              # Combined A+B images
│   └── cyclegan/
│       ├── train/{trainA,trainB}
│       ├── val/{valA,valB}
│       └── test/{testA,testB}
├── 6materialized_splits_33nyul/
├── 6materialized_splits_34npeaks/
│
├── 7materialized_splits_31baselineBodyRegion/  # Body region subsets
├── 7materialized_splits_32p99BodyRegion/
│   ├── AB/  (abdomen)
│   ├── HN/  (head & neck)
│   ├── TH/  (thorax)
│   └── ...
├── 7materialized_splits_33nyulBodyRegion/
└── 7materialized_splits_34npeaksBodyRegion/
```

## Notes

- **Default method**: If you don't specify a method, scripts default to `32p99`
- **Run once**: `20resampling.py` only needs to be run once
- **Parallel processing**: Different normalization methods can be run in parallel
- **Method validation**: Scripts will validate that the normalization method is valid
- **Auto-configuration**: Scripts automatically configure input/output paths based on the normalization method
- **Error checking**: Scripts check if required input directories exist before running
