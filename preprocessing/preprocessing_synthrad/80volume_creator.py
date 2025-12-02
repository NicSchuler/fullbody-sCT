#!/usr/bin/env python

"""
80volume_creator.py

Reconstructs 3D NIfTI volumes from 2D prediction slices.

Usage:
    python 80volume_creator.py <result_dir> [normalization_method]

Examples:
    python 80volume_creator.py /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results/pix2pix_synthrad_THUF1 32p99
    python 80volume_creator.py /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results/cyclegan_THUF

This script:
    1. Reads fake_B prediction slices (PNG format) from <result_dir>/test_latest/images/
    2. Groups slices by patient ID
    3. Looks up original metadata (affine, spacing) from 5slices_{method} directory
    4. Reconstructs 3D volumes by stacking slices in order
    5. Saves as NIfTI to <result_dir>/test_latest/volumes/
"""

import os
import sys
import re
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import nibabel as nib
from PIL import Image

# ===================== CONFIG =====================

BASE_ROOT = "/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed"
DEFAULT_NORMALIZATION_METHOD = "32p99"

# ==================================================


def parse_slice_filename(filename: str):
    """
    Parse slice filename to extract patient_id, slice_num, and type.
    
    Example: "AB_1ABA009-38_fake_B.png"
    Returns: ("AB_1ABA009", 38, "fake_B")
    
    Returns None if parsing fails.
    """
    # Pattern: {PATIENT_ID}-{SLICE_NUM}_{TYPE}.png
    pattern = r'^(.+?)-(\d+)_(.+)\.(png|nii|nii\.gz)$'
    match = re.match(pattern, filename)
    
    if not match:
        return None
    
    patient_id = match.group(1)
    slice_num = int(match.group(2))
    slice_type = match.group(3)
    
    return patient_id, slice_num, slice_type


def find_reference_slice(patient_id: str, slice_num: int, slice_dir: Path):
    """
    Find the original reference slice to extract metadata.
    
    Searches for: {patient_id}-{slice_num}.nii[.gz] in slice_dir
    """
    # Try both .nii and .nii.gz
    for ext in ['.nii.gz', '.nii']:
        ref_path = slice_dir / f"{patient_id}-{slice_num}{ext}"
        if ref_path.exists():
            return ref_path
    
    return None


def load_png_slice(png_path: Path) -> np.ndarray:
    """
    Load PNG slice and convert to float32 [0, 1] range.
    """
    img = Image.open(png_path)
    arr = np.array(img, dtype=np.float32)
    
    # Normalize to [0, 1] if needed
    if arr.max() > 1.0:
        arr = arr / 255.0
    
    # Ensure 2D (grayscale)
    if arr.ndim == 3:
        if arr.shape[2] == 3:  # RGB
            arr = arr.mean(axis=2)
        elif arr.shape[2] == 1:
            arr = arr[:, :, 0]
    
    return arr


def reconstruct_volume(
    patient_id: str,
    slices_dict: dict,
    images_dir: Path,
    reference_dir: Path,
    output_dir: Path
):
    """
    Reconstruct 3D volume for one patient from fake_B slices.
    
    Args:
        patient_id: Patient identifier (e.g., "AB_1ABA009")
        slices_dict: Dict mapping slice_num -> filename
        images_dir: Directory containing prediction PNG slices
        reference_dir: Directory containing original slices for metadata
        output_dir: Directory to save reconstructed volume
    """
    # Sort slices by slice number
    sorted_slices = sorted(slices_dict.items())
    slice_nums = [s[0] for s in sorted_slices]
    slice_files = [s[1] for s in sorted_slices]
    
    if not slice_nums:
        print(f"!WARNING! No slices found for {patient_id}")
        return
    
    # Load all slices
    slice_arrays = []
    reference_img = None
    
    for slice_num, slice_file in zip(slice_nums, slice_files):
        # Load PNG
        png_path = images_dir / slice_file
        slice_arr = load_png_slice(png_path)
        slice_arrays.append(slice_arr)
        
        # Get reference metadata from first slice if not yet loaded
        if reference_img is None:
            ref_path = find_reference_slice(patient_id, slice_num, reference_dir)
            if ref_path:
                reference_img = nib.load(ref_path)
    
    # Stack slices along z-axis
    volume = np.stack(slice_arrays, axis=-1)  # (x, y, z)
    
    # Get affine transform from reference or use identity
    if reference_img is not None:
        affine = reference_img.affine.copy()
        
        # Adjust z-dimension spacing if needed (slice count might differ)
        # Keep x,y spacing, adjust z origin based on first slice number
        z_offset = slice_nums[0]
        if z_offset > 0:
            # Adjust origin for first slice
            z_spacing = affine[2, 2]
            affine[2, 3] += z_offset * z_spacing
    else:
        print(f"!WARNING! No reference metadata found for {patient_id}, using identity affine")
        # Default: 1mm isotropic spacing
        affine = np.eye(4)
    
    # Create NIfTI image
    nifti_img = nib.Nifti1Image(volume, affine)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{patient_id}_fake_B.nii.gz"
    nib.save(nifti_img, output_path)
    
    print(f"  ✓ {patient_id}: {len(slice_nums)} slices → {output_path.name}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python 80volume_creator.py <result_dir> [normalization_method]")
        print("\nExample:")
        print("  python 80volume_creator.py /path/to/100results/pix2pix_synthrad_THUF1 32p99")
        sys.exit(1)
    
    result_dir = Path(sys.argv[1])
    norm_method = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_NORMALIZATION_METHOD
    
    # Paths
    images_dir = result_dir / "test_latest" / "images"
    output_dir = result_dir / "test_latest" / "volumes"
    reference_dir = Path(BASE_ROOT) / f"5slices_{norm_method}" / "pix2pix_2d" / "full" / "B"
    
    # Validate input
    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        sys.exit(1)
    
    if not reference_dir.exists():
        print(f"WARNING: Reference directory not found: {reference_dir}")
        print("         Will use default affine (identity matrix)")
        reference_dir = None
    
    print(f"=" * 60)
    print(f"Volume Reconstruction")
    print(f"Images:    {images_dir}")
    print(f"Reference: {reference_dir}")
    print(f"Output:    {output_dir}")
    print(f"=" * 60)
    print()
    
    # Scan for fake_B slices
    fake_b_slices = defaultdict(dict)  # patient_id -> {slice_num: filename}
    
    for filepath in sorted(images_dir.glob("*.png")):
        parsed = parse_slice_filename(filepath.name)
        if parsed is None:
            continue
        
        patient_id, slice_num, slice_type = parsed
        
        # Only process fake_B
        if slice_type != "fake_B":
            continue
        
        fake_b_slices[patient_id][slice_num] = filepath.name
    
    if not fake_b_slices:
        print("ERROR: No fake_B slices found!")
        sys.exit(1)
    
    print(f"Found {len(fake_b_slices)} patients with fake_B predictions")
    print()
    
    # Reconstruct volumes
    for patient_id in tqdm(sorted(fake_b_slices.keys())):
        reconstruct_volume(
            patient_id,
            fake_b_slices[patient_id],
            images_dir,
            reference_dir,
            output_dir
        )
    
    print()
    print(f"✓ Reconstruction complete. Volumes saved to: {output_dir}")


if __name__ == "__main__":
    main()
