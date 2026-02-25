"""
32: Per-File P99 Standardization
==================================
CT: Min-max normalization using fixed HU window [-1024, 1200] → [0, 1]
MRI: Per-file p99 normalization (current technique used in training)

This is the technique currently being used - per-image adaptive normalization
for MRI using 99th percentile of foreground voxels.

Note: This script processes ABDOMEN data only (AB_* prefix).
"""

import numpy as np
import SimpleITK as sitk
from pathlib import Path


# ==========================
# Configuration
# ==========================
BASE_ROOT = Path("/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed")

src_root = BASE_ROOT / "2resampledNifti"  # Input: resampled but not normalized
out_root = BASE_ROOT / "experiment2" / "32p99" / "3normalized"  # Output: normalized

save_zipped = True
# ==========================


def normalize_ct(arr):
    """
    CT normalization: Fixed HU window [-1024, 1200] → [0, 1]
    
    Mapping:
        -1024 HU (air/background) -> 0
        +1200 HU (dense bone) -> 1
    
    This ensures consistent scaling across all CT images.
    """
    arr = np.clip(arr, -1024, 1200)
    arr = (arr + 1024) / 2224.0  # Maps -1024→0, 1200→1
    return arr.astype(np.float32)


def normalize_mr_p99(arr):
    """
    MRI p99 normalization: Per-file adaptive normalization
    
    Mapping:
        0 (background) -> 0
        p99 (99th percentile of non-zero values) -> 1
        Values above p99 are clipped to 1
    
    This is per-image min-max rescaling with outlier removal,
    accounting for variable MRI intensity ranges across scans.
    """
    # Calculate p99 only on non-zero (foreground) pixels
    foreground = arr[arr > 0]
    if foreground.size > 0:
        p99 = np.percentile(foreground, 99)
        if p99 > 0:
            arr = arr / p99  # Scale by p99
            arr = np.clip(arr, 0, 1)  # Clip to [0, 1] range
        else:
            arr = np.zeros_like(arr)
    else:
        arr = np.zeros_like(arr)
    return arr.astype(np.float32)


def process_image(in_path: Path, out_path: Path):
    """
    Load image, apply p99 normalization, and save.
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
        arr_norm = normalize_ct(arr)
    elif is_mr:
        arr_norm = normalize_mr_p99(arr)
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
    
    # Filter for abdomen only (AB_* prefix)
    all_dirs = [d for d in src_root.iterdir() if d.is_dir()]
    case_dirs = [d for d in all_dirs if d.name.startswith("AB_")]
    total = len(case_dirs)
    
    print(f"Starting 32 Per-File P99 Standardization")
    print(f"Input:  {src_root}")
    print(f"Output: {out_root}")
    print(f"Filter: Abdomen only (AB_*)")
    print(f"Cases:  {total} abdomen cases")
    print(f"CT:  [-1024, 1200] HU → [0, 1]")
    print(f"MRI: Per-file p99 normalization → [0, 1]")
    print("-" * 60)
    
    for case_dir in sorted(case_dirs):
        process_case(case_dir, out_root)
        count += 1
        
        if count % 25 == 0:
            print(f"Processed {count}/{total} cases...")
    
    print(f"\nCompleted: {count}/{total} cases processed")


if __name__ == "__main__":
    main()
