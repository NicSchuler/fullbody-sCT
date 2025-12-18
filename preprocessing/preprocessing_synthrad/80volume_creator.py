#!/usr/bin/env python

"""
80volume_creator.py

Reconstructs 3D NIfTI volumes from 2D NIfTI slices.

Usage:
    python 80volume_creator.py <model_region_folder>

Examples:
    python 80volume_creator.py pix2pix_synthrad_abdomen
    python 80volume_creator.py cyclegan_brain

This script:
    1. Reads NIfTI slices from 9latestTestImages/<model_region>/fake_nifti/ and real_nifti/
    2. Groups slices by patient ID
    3. Gets original metadata (affine, spacing) from 2resampledNifti/<patient_id>/CT_reg/
    4. Reconstructs 3D volumes by stacking slices in order
    5. Saves reconstructed volumes to 9latestTestImages/<model_region>/reconstruction/<patient_id>/
    6. Copies original CT volume to the patient reconstruction folder for comparison
"""

import os
import sys
import re
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import nibabel as nib

# ===================== CONFIG =====================

BASE_ROOT = Path("/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed")
TEST_IMAGES_DIR = BASE_ROOT / "9latestTestImages"
RESAMPLED_DIR = BASE_ROOT / "2resampledNifti"

# ==================================================


def parse_nifti_slice_filename(filename: str):
    """
    Parse NIfTI slice filename to extract patient_id and slice_num.
    
    Example: "AB_1ABA009_38.nii" or "AB_1ABA009_38.nii.gz"
    Returns: ("AB_1ABA009", 38)
    
    Returns None if parsing fails.
    """
    # Pattern: {PATIENT_ID}_{SLICE_NUM}.nii[.gz]
    # Patient ID format: AB_1ABC123 (letters, underscore, digit, letters, digits)
    pattern = r'^(AB_\d[A-Z]{2,3}\d{3})_(\d+)\.nii(\.gz)?$'
    match = re.match(pattern, filename)
    
    if not match:
        return None
    
    patient_id = match.group(1)
    slice_num = int(match.group(2))
    
    return patient_id, slice_num


def find_original_ct(patient_id: str) -> Path:
    """
    Find the original CT volume in 2resampledNifti.
    
    Searches for: 2resampledNifti/{patient_id}/CT_reg/*.nii.gz
    """
    ct_dir = RESAMPLED_DIR / patient_id / "CT_reg"
    if ct_dir.exists():
        for f in ct_dir.iterdir():
            if f.suffix == '.gz' or f.suffix == '.nii':
                return f
    return None


def load_nifti_slice(nifti_path: Path) -> tuple:
    """
    Load a 2D NIfTI slice and return the data array and header info.
    
    Returns: (array, affine, header)
    """
    img = nib.load(nifti_path)
    data = img.get_fdata().astype(np.float32)
    
    # Handle 3D images with single slice dimension
    if data.ndim == 3:
        # Squeeze out singleton dimensions
        if data.shape[2] == 1:
            data = data[:, :, 0]
        elif data.shape[0] == 1:
            data = data[0, :, :]
        elif data.shape[1] == 1:
            data = data[:, 0, :]
    
    return data, img.affine, img.header


def reconstruct_volume_from_nifti(
    patient_id: str,
    slices_dict: dict,
    slices_dir: Path,
    output_path: Path,
    reference_affine: np.ndarray = None
):
    """
    Reconstruct 3D volume for one patient from NIfTI slices.
    
    Args:
        patient_id: Patient identifier (e.g., "AB_1ABA009")
        slices_dict: Dict mapping slice_num -> filename
        slices_dir: Directory containing NIfTI slices
        output_path: Full path for the output NIfTI file
        reference_affine: Affine matrix from reference volume (optional)
    
    Returns:
        The reconstructed nibabel image or None if failed
    """
    # Sort slices by slice number
    sorted_slices = sorted(slices_dict.items())
    slice_nums = [s[0] for s in sorted_slices]
    slice_files = [s[1] for s in sorted_slices]
    
    if not slice_nums:
        print(f"!WARNING! No slices found for {patient_id}")
        return None
    
    # Load all slices
    slice_arrays = []
    first_affine = None
    
    for slice_num, slice_file in zip(slice_nums, slice_files):
        nifti_path = slices_dir / slice_file
        if not nifti_path.exists():
            print(f"  !WARNING! Slice not found: {nifti_path}")
            continue
            
        data, affine, header = load_nifti_slice(nifti_path)
        slice_arrays.append(data)
        
        if first_affine is None:
            first_affine = affine
    
    if not slice_arrays:
        print(f"!WARNING! No valid slices loaded for {patient_id}")
        return None
    
    # Stack slices along z-axis
    volume = np.stack(slice_arrays, axis=-1)  # (x, y, z)
    
    # Use reference affine if provided, otherwise use first slice's affine
    if reference_affine is not None:
        affine = reference_affine.copy()
    elif first_affine is not None:
        affine = first_affine.copy()
    else:
        print(f"!WARNING! No affine found for {patient_id}, using identity")
        affine = np.eye(4)
    
    # Create NIfTI image
    nifti_img = nib.Nifti1Image(volume, affine)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nifti_img, output_path)
    
    return nifti_img


def process_patient(
    patient_id: str,
    fake_slices: dict,
    real_slices: dict,
    fake_dir: Path,
    real_dir: Path,
    output_dir: Path
) -> bool:
    """
    Process one patient: reconstruct fake and real volumes, copy original CT.
    
    Args:
        patient_id: Patient identifier
        fake_slices: Dict of slice_num -> filename for fake slices
        real_slices: Dict of slice_num -> filename for real slices
        fake_dir: Directory containing fake NIfTI slices
        real_dir: Directory containing real NIfTI slices
        output_dir: Base output directory for reconstructions
    
    Returns:
        True if successful, False otherwise
    """
    patient_out_dir = output_dir / patient_id
    patient_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find and load original CT for reference affine
    original_ct_path = find_original_ct(patient_id)
    reference_affine = None
    
    if original_ct_path:
        try:
            original_img = nib.load(original_ct_path)
            reference_affine = original_img.affine
            
            # Copy original CT to output folder
            dest_ct_path = patient_out_dir / f"{patient_id}_original_CT.nii.gz"
            if not dest_ct_path.exists():
                shutil.copy2(original_ct_path, dest_ct_path)
        except Exception as e:
            print(f"  !WARNING! Could not load original CT for {patient_id}: {e}")
    
    success = True
    
    # Reconstruct fake (synthetic) volume
    if fake_slices:
        fake_output = patient_out_dir / f"{patient_id}_synthetic_CT.nii.gz"
        result = reconstruct_volume_from_nifti(
            patient_id, fake_slices, fake_dir, fake_output, reference_affine
        )
        if result is None:
            success = False
    
    # Reconstruct real (input) volume
    if real_slices:
        real_output = patient_out_dir / f"{patient_id}_real_input.nii.gz"
        result = reconstruct_volume_from_nifti(
            patient_id, real_slices, real_dir, real_output, reference_affine
        )
        if result is None:
            success = False
    
    return success


def main():
    if len(sys.argv) < 2:
        print("Usage: python 80volume_creator.py <model_region_folder>")
        print("\nExample:")
        print("  python 80volume_creator.py pix2pix_synthrad_abdomen")
        print("  python 80volume_creator.py cyclegan_brain")
        print("\nAvailable folders in 9latestTestImages:")
        if TEST_IMAGES_DIR.exists():
            for d in sorted(TEST_IMAGES_DIR.iterdir()):
                if d.is_dir() and not d.name.startswith('.'):
                    # Check if it has fake_nifti subfolder
                    if (d / "fake_nifti").exists():
                        print(f"  - {d.name}")
        sys.exit(1)
    
    model_region = sys.argv[1]
    
    # Paths
    model_dir = TEST_IMAGES_DIR / model_region
    fake_dir = model_dir / "fake_nifti"
    real_dir = model_dir / "real_nifti"
    output_dir = model_dir / "reconstruction"
    
    # Validate input
    if not model_dir.exists():
        print(f"ERROR: Model/region directory not found: {model_dir}")
        sys.exit(1)
    
    if not fake_dir.exists():
        print(f"ERROR: fake_nifti directory not found: {fake_dir}")
        sys.exit(1)
    
    has_real = real_dir.exists()
    
    print("=" * 70)
    print("Volume Reconstruction from NIfTI Slices")
    print("=" * 70)
    print(f"Model/Region:    {model_region}")
    print(f"Fake slices:     {fake_dir}")
    print(f"Real slices:     {real_dir} {'✓' if has_real else '(not found)'}")
    print(f"Original CTs:    {RESAMPLED_DIR}")
    print(f"Output:          {output_dir}")
    print("=" * 70)
    print()
    
    # Scan for NIfTI slices
    fake_slices = defaultdict(dict)  # patient_id -> {slice_num: filename}
    real_slices = defaultdict(dict)
    
    # Process fake slices
    for filepath in sorted(fake_dir.glob("*.nii*")):
        parsed = parse_nifti_slice_filename(filepath.name)
        if parsed is None:
            continue
        patient_id, slice_num = parsed
        fake_slices[patient_id][slice_num] = filepath.name
    
    # Process real slices if available
    if has_real:
        for filepath in sorted(real_dir.glob("*.nii*")):
            parsed = parse_nifti_slice_filename(filepath.name)
            if parsed is None:
                continue
            patient_id, slice_num = parsed
            real_slices[patient_id][slice_num] = filepath.name
    
    if not fake_slices:
        print("ERROR: No NIfTI slices found in fake_nifti!")
        print("Expected filename format: AB_1ABC123_42.nii")
        sys.exit(1)
    
    # Get all unique patient IDs
    all_patients = set(fake_slices.keys()) | set(real_slices.keys())
    
    print(f"Found {len(all_patients)} patients")
    print(f"  - Fake slices: {sum(len(v) for v in fake_slices.values())} total")
    if has_real:
        print(f"  - Real slices: {sum(len(v) for v in real_slices.values())} total")
    print()
    
    # Process each patient
    success_count = 0
    for patient_id in tqdm(sorted(all_patients), desc="Reconstructing volumes"):
        success = process_patient(
            patient_id,
            fake_slices.get(patient_id, {}),
            real_slices.get(patient_id, {}),
            fake_dir,
            real_dir,
            output_dir
        )
        if success:
            success_count += 1
    
    print()
    print("=" * 70)
    print(f"✓ Reconstruction complete!")
    print(f"  Processed: {success_count}/{len(all_patients)} patients")
    print(f"  Output: {output_dir}")
    print()
    print("Each patient folder contains:")
    print("  - {patient}_synthetic_CT.nii.gz  (reconstructed from fake slices)")
    if has_real:
        print("  - {patient}_real_input.nii.gz    (reconstructed from real slices)")
    print("  - {patient}_original_CT.nii.gz   (copied from 2resampledNifti)")
    print("=" * 70)


if __name__ == "__main__":
    main()
