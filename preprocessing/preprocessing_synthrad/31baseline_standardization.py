"""
31: Baseline Standardization
==============================
CT: Min-max normalization using fixed HU window [-1024, 1200] → [0, 1]
MRI: Min-max normalization clipping at 2000, then scale to [0, 1]

This is a simple baseline approach using fixed thresholds.
"""

import numpy as np
import SimpleITK as sitk
from pathlib import Path


# ==========================
# Configuration
# ==========================
BASE_ROOT = Path("/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed")

src_root = BASE_ROOT / "2resampledNifti"  # Input: resampled but not normalized
out_root = BASE_ROOT / "3normalized_31baseline"  # Output: normalized

save_zipped = True
# ==========================


def normalize_ct_baseline(arr):
    """
    CT baseline normalization: Fixed HU window [-1024, 1200] → [0, 1]
    
    Mapping:
        -1024 HU (air/background) -> 0
        +1200 HU (dense bone) -> 1
    
    This ensures consistent scaling across all CT images.
    """
    arr = np.clip(arr, -1024, 1200)
    arr = (arr + 1024) / 2224.0  # Maps -1024→0, 1200→1
    return arr.astype(np.float32)


def normalize_mr_baseline(arr):
    """
    MRI baseline normalization: Clip at 2000, then scale to [0, 1]
    
    Mapping:
        0 (background) -> 0
        2000 (clipped maximum) -> 1
    
    Simple global thresholding approach.
    """
    arr = np.clip(arr, 0, 2000)
    arr = arr / 2000.0  # Maps 0→0, 2000→1
    return arr.astype(np.float32)


def process_image(in_path: Path, out_path: Path):
    """
    Load image, apply baseline normalization, and save.
    """
    img = sitk.ReadImage(str(in_path))
    arr = sitk.GetArrayFromImage(img)
    
    # Determine modality from filename
    is_ct = "CT" in in_path.name
    is_mr = "MR" in in_path.name
    is_mask = "mask" in in_path.name.lower()
    
    # Apply normalization based on modality
    if is_mask:
        # Masks stay as-is (binary 0/1)
        arr_norm = arr
    elif is_ct:
        arr_norm = normalize_ct_baseline(arr)
    elif is_mr:
        arr_norm = normalize_mr_baseline(arr)
    else:
        # Unknown modality - skip normalization
        arr_norm = arr
    
    # Create output image
    out_img = sitk.GetImageFromArray(arr_norm)
    out_img.SetSpacing(img.GetSpacing())
    out_img.SetDirection(img.GetDirection())
    out_img.SetOrigin(img.GetOrigin())
    
    if is_mask:
        out_img = sitk.Cast(out_img, sitk.sitkUInt8)
    
    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out_img, str(out_path), useCompression=save_zipped)


def process_case(case_dir: Path, out_root: Path):
    """
    Process all images in a case directory.
    """
    case_id = case_dir.name
    
    # Process all subdirectories (CT_reg, MR, masks)
    for subdir in case_dir.iterdir():
        if subdir.is_dir():
            for nii_file in subdir.glob("*.nii*"):
                # Create corresponding output path
                out_file = out_root / case_id / subdir.name / nii_file.name
                process_image(nii_file, out_file)
    
    print(f"[OK] {case_id}")


def main():
    count = 0
    total = sum(1 for d in src_root.iterdir() if d.is_dir())
    
    print(f"Starting 31 Baseline Standardization")
    print(f"Input:  {src_root}")
    print(f"Output: {out_root}")
    print(f"CT:  [-1024, 1200] HU → [0, 1]")
    print(f"MRI: [0, 2000] → [0, 1]")
    print("-" * 60)
    
    for case_dir in sorted(src_root.iterdir()):
        if case_dir.is_dir():
            process_case(case_dir, out_root)
            count += 1
            
            if count % 25 == 0:
                print(f"Processed {count}/{total} cases...")
    
    print(f"\nCompleted: {count}/{total} cases processed")


if __name__ == "__main__":
    main()
