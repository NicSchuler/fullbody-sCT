"""
31: Baseline Standardization
==============================
CT: Min-max normalization using fixed HU window [-1024, 1200] → [0, 1]
MRI: Min-max normalization clipping at 2000, then scale to [0, 1]

This is a simple baseline approach using fixed thresholds.

Usage:
    python 31baseline_standardization.py              # Abdomen only (default)
    python 31baseline_standardization.py --all-data   # All body regions
    python 31baseline_standardization.py --mr-only    # MR only (for inference)
    python 31baseline_standardization.py --mr-only --all-data --src-root /path/to/input --out-root /path/to/output
"""

import argparse
import sys
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm


# ==========================
# Configuration
# ==========================
BASE_ROOT = Path("/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed")

src_root = BASE_ROOT / "2resampledNifti"  # Input: resampled but not normalized
out_root = BASE_ROOT / "31baseline" / "3normalized"  # Output: normalized

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


def process_case(case_dir: Path, out_root: Path, mr_only: bool = False):
    """
    Process all images in a case directory.

    Args:
        case_dir: Input case directory
        out_root: Output root directory
        mr_only: If True, skip CT_reg subdirectory processing (for inference)
    """
    case_id = case_dir.name

    # Process all subdirectories (CT_reg, MR, masks)
    for subdir in case_dir.iterdir():
        if subdir.is_dir():
            # Skip CT_reg if in MR-only mode
            if mr_only and subdir.name == "CT_reg":
                continue

            for nii_file in subdir.glob("*.nii*"):
                # Create corresponding output path
                out_file = out_root / case_id / subdir.name / nii_file.name
                process_image(nii_file, out_file)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply baseline intensity normalization to NIfTI volumes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Original behavior (abdomen only):
    python 31baseline_standardization.py

    # All body regions:
    python 31baseline_standardization.py --all-data

    # Inference mode (MR only, all regions, custom paths):
    python 31baseline_standardization.py --mr-only --all-data \\
        --src-root /path/to/2resampledNifti \\
        --out-root /path/to/31baseline/3normalized
        """
    )

    parser.add_argument(
        "--src-root", type=str, default=None,
        help=f"Source directory (default: {src_root})"
    )
    parser.add_argument(
        "--out-root", type=str, default=None,
        help=f"Output directory (default: {out_root})"
    )
    parser.add_argument(
        "--all-data", action="store_true",
        help="Process all body regions, not just abdomen"
    )
    parser.add_argument(
        "--mr-only", action="store_true",
        help="Process MR only, skip CT normalization (for inference)"
    )
    parser.add_argument(
        "--patient-ids", nargs="+", default=None,
        help="Specific patient IDs to process (default: all)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Use command line args or defaults
    input_root = Path(args.src_root) if args.src_root else src_root
    output_root = Path(args.out_root) if args.out_root else out_root
    abdomen_only = not args.all_data
    mr_only = args.mr_only

    # Get case directories
    if args.patient_ids:
        case_dirs = [input_root / pid for pid in args.patient_ids if (input_root / pid).is_dir()]
    else:
        all_dirs = [d for d in input_root.iterdir() if d.is_dir()]
        # Filter for abdomen only if requested
        if abdomen_only:
            case_dirs = [d for d in all_dirs if d.name.startswith("AB_")]
        else:
            case_dirs = all_dirs

    filter_msg = "Abdomen only (AB_*)" if abdomen_only else "All body regions"
    total = len(case_dirs)

    print("=" * 60)
    print("31 Baseline Standardization")
    print("=" * 60)
    print(f"Input:    {input_root}")
    print(f"Output:   {output_root}")
    print(f"Filter:   {filter_msg}")
    print(f"MR only:  {mr_only}")
    print(f"CT:       [-1024, 1200] HU → [0, 1]")
    print(f"MRI:      [0, 2000] → [0, 1]")
    print("=" * 60)

    count = 0
    for case_dir in tqdm(sorted(case_dirs)):
        process_case(case_dir, output_root, mr_only=mr_only)
        count += 1

    print(f"\nCompleted: {count}/{total} cases processed")


if __name__ == "__main__":
    main()
