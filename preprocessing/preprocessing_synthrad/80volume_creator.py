#!/usr/bin/env python

"""
80volume_creator.py

Reconstructs 3D NIfTI volumes from 2D NIfTI slices at both resampled and original dimensions.

Usage:
    python 80volume_creator.py <model_region_folder> [--copy_originals] [--reconstruct_original]

Arguments:
    model_region_folder     Model/region folder name in 9latestTestImages (e.g., pix2pix_synthrad_abdomen)
    --copy_originals        Optional: Copy original CT/MR files to output folder (default: False)
    --reconstruct_original  Optional: Reconstruct sCT at original dimensions (default: False)

Examples:
    python 80volume_creator.py pix2pix_synthrad_abdomen
    python 80volume_creator.py pix2pix_synthrad_abdomen --copy_originals
    python 80volume_creator.py pix2pix_synthrad_abdomen --reconstruct_original
    python 80volume_creator.py pix2pix_synthrad_abdomen --copy_originals --reconstruct_original

This script:
    1. Reads NIfTI slices from 9latestTestImages/<model_region>/fake_nifti/
    2. Groups slices by patient ID
    3. Reconstructs 3D volumes at resampled resolution (256x256) by stacking slices
    4. Optionally applies reverse resampling to create volumes at original 1initNifti dimensions
    5. Optionally copies original CT/MR files for comparison

Output Files:
    Always created:
    - sCT_256_dim.nii.gz (reconstructed sCT at 256x256 resolution)
    - sCT_original_dim_mask_not_aligned.nii.gz (sCT at original dimensions)

    Created when --reconstruct_original is set:
    - CT_pre_and_postprocessed_to_original_dim_mask_not_aligned.nii.gz (validation volume)

    Created when --copy_originals is set:
    - CT_256_dim_copy_2resampledNifti.nii.gz (original CT at 256x256 from 2resampledNifti)
    - MR_256_dim_copy_2resampledNifti.nii.gz (original MR at 256x256 from 2resampledNifti)
    - CT_original_dim_copy_1initNifti.nii.gz (original CT at original dimensions from 1initNifti)
    - MR_original_dim_copy_1initNifti.nii.gz (original MR at original dimensions from 1initNifti)

IMPORTANT ALIGNMENT WARNING:
    The reconstructed volumes at original dimensions are NOT aligned with the original masks!
    - Safe for: Dosimetric analysis
    - NOT safe for: Image-level comparison with masked regions
    This is because the reconstruction process involves resampling transformations that may
    introduce slight spatial misalignments with the original mask coordinates.
"""

import sys
import re
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import nibabel as nib
import SimpleITK as sitk

# ===================== CONFIG =====================

BASE_ROOT = Path("/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed")
TEST_IMAGES_DIR = BASE_ROOT / "9latestTestImages"
RESAMPLED_DIR = BASE_ROOT / "2resampledNifti"
INIT_DIR = BASE_ROOT / "1initNifti"

# ==================================================


def get_body_region(patient_id: str) -> str:
    """
    Extract body region from patient_id.

    Example: "AB_1ABA009" -> "AB"
             "HN_1ABC123" -> "HN"

    Returns:
        Body region prefix (e.g., "AB", "HN", "TH", "BRAIN", "PELVIS")
    """
    return patient_id.split("_")[0]


def find_original_ct_init(patient_id: str) -> Path:
    """
    Find the original CT volume in 1initNifti.

    Searches for: 1initNifti/{patient_id}/CT_reg/*.nii.gz
    """
    ct_dir = INIT_DIR / patient_id / "CT_reg"
    if ct_dir.exists():
        for f in ct_dir.iterdir():
            if f.suffix == '.gz' or f.suffix == '.nii':
                return f
    return None


def find_original_mr_init(patient_id: str) -> Path:
    """
    Find the original MR volume in 1initNifti.

    Searches for: 1initNifti/{patient_id}/MR/*.nii.gz
    """
    mr_dir = INIT_DIR / patient_id / "MR"
    if mr_dir.exists():
        for f in mr_dir.iterdir():
            if f.suffix == '.gz' or f.suffix == '.nii':
                return f
    return None


def reverse_resample_to_original(resampled_img: nib.Nifti1Image, patient_id: str) -> nib.Nifti1Image:
    """
    Reverse resampling transformations to restore original 1initNifti dimensions.

    This reverses the transformations applied by 20resampling.py:
    1. For TH/AB/PELVIS: Upsample 256x256 → 512x512 (halve spacing, double size)
    2. Pad upsampled volume to center of original dimensions

    Args:
        resampled_img: The reconstructed volume at resampled resolution (256x256xZ)
        patient_id: Patient identifier (e.g., "AB_1ABA009")

    Returns:
        Volume with exact same dimensions as 1initNifti, or None if failed
    """
    # Load original CT for metadata
    original_ct_path = find_original_ct_init(patient_id)

    if original_ct_path is None:
        print(f"  !WARNING! Missing original CT for {patient_id}")
        return None

    # Load original CT to get target dimensions
    original_ct_img = nib.load(original_ct_path)
    original_shape = original_ct_img.shape
    original_affine = original_ct_img.affine

    # Get body region to determine if upsampling is needed
    body_region = get_body_region(patient_id)

    # Get resampled data as numpy array
    resampled_data = resampled_img.get_fdata().astype(np.float32)

    # Step 1: Upsample for TH/AB/PELVIS regions (256→512)
    if body_region.upper() in {"TH", "AB", "PELVIS"}:
        # Convert to SimpleITK for resampling
        # Note: nibabel uses (x, y, z) convention, SimpleITK uses (z, y, x)
        # So we need to transpose before and after SimpleITK operations
        resampled_data_sitk = np.transpose(resampled_data, (2, 1, 0))  # (x,y,z) -> (z,y,x)
        sitk_img = sitk.GetImageFromArray(resampled_data_sitk)

        # Get spacing and convert to list for SimpleITK
        spacing = resampled_img.header.get_zooms()
        sitk_img.SetSpacing([float(spacing[0]), float(spacing[1]), float(spacing[2])])

        # Get origin from affine matrix
        origin = resampled_img.affine[:3, 3]
        sitk_img.SetOrigin([float(origin[0]), float(origin[1]), float(origin[2])])

        # Get direction matrix (SimpleITK expects flattened 3x3 in row-major order)
        direction = resampled_img.affine[:3, :3]
        # Normalize direction vectors (SimpleITK requires orthonormal direction matrix)
        direction_flat = []
        for i in range(3):
            vec = direction[:, i]
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            direction_flat.extend([float(vec[0]), float(vec[1]), float(vec[2])])
        sitk_img.SetDirection(direction_flat)

        current_spacing = sitk_img.GetSpacing()
        current_size = sitk_img.GetSize()

        # Halve spacing (double resolution)
        new_spacing = (
            current_spacing[0] / 2.0,
            current_spacing[1] / 2.0,
            current_spacing[2]  # z unchanged
        )

        # Double size in x, y
        new_size = [
            current_size[0] * 2,  # x
            current_size[1] * 2,  # y
            current_size[2]       # z unchanged
        ]

        # Resample
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetOutputDirection(sitk_img.GetDirection())
        resampler.SetOutputOrigin(sitk_img.GetOrigin())
        resampler.SetDefaultPixelValue(-1024)

        upsampled = resampler.Execute(sitk_img)
        volume_data_sitk = sitk.GetArrayFromImage(upsampled)  # Returns (z, y, x)
        # Convert back to nibabel convention: (z, y, x) -> (x, y, z)
        volume_data = np.transpose(volume_data_sitk, (2, 1, 0))  # Now (x, y, z) at 512x512xZ
    else:
        # HN/BRAIN: keep at 256x256xZ
        volume_data = resampled_data

    # Step 2: Pad volume to center of original dimensions
    # volume_data is in nibabel convention (x, y, z)
    # original_shape is also (x, y, z) for nibabel
    full_volume = np.full(original_shape, -1024, dtype=np.float32)

    # Calculate padding to center the volume
    vol_x, vol_y, vol_z = volume_data.shape
    orig_x, orig_y, orig_z = original_shape

    # Handle z-axis: match the number of slices (no centering in z)
    z_slices = min(vol_z, orig_z)

    # Calculate positions for centering
    # If volume is larger than original, we crop from center of volume
    # If volume is smaller than original, we pad to center of original
    if vol_x <= orig_x:
        # Pad case: place volume in center of original
        start_x_orig = (orig_x - vol_x) // 2
        end_x_orig = start_x_orig + vol_x
        start_x_vol = 0
        end_x_vol = vol_x
    else:
        # Crop case: take center of volume
        start_x_orig = 0
        end_x_orig = orig_x
        start_x_vol = (vol_x - orig_x) // 2
        end_x_vol = start_x_vol + orig_x

    if vol_y <= orig_y:
        # Pad case: place volume in center of original
        start_y_orig = (orig_y - vol_y) // 2
        end_y_orig = start_y_orig + vol_y
        start_y_vol = 0
        end_y_vol = vol_y
    else:
        # Crop case: take center of volume
        start_y_orig = 0
        end_y_orig = orig_y
        start_y_vol = (vol_y - orig_y) // 2
        end_y_vol = start_y_vol + orig_y

    # Place the volume at the center (or crop volume to fit)
    full_volume[start_x_orig:end_x_orig, start_y_orig:end_y_orig, :z_slices] = \
        volume_data[start_x_vol:end_x_vol, start_y_vol:end_y_vol, :z_slices]

    # Create NIfTI image with original metadata
    restored_img = nib.Nifti1Image(full_volume, original_affine)

    return restored_img


def parse_nifti_slice_filename(filename: str):
    """
    Parse NIfTI slice filename to extract patient_id and slice_num.
    
    Example: "AB_1ABA009_38.nii" or "AB_1ABA009_38.nii.gz"
    Returns: ("AB_1ABA009", 38)
    
    Returns None if parsing fails.
    """
    # Pattern: {PATIENT_ID}_{SLICE_NUM}.nii[.gz]
    # Patient ID format: PREFIX_1ABC123 where PREFIX can be AB, HN, TH, brain, pelvis, etc.
    pattern = r'^([A-Za-z]+_\d[A-Z]{1,3}\d{2,3})_(\d+)\.nii(\.gz)?$'
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


def find_original_mr(patient_id: str) -> Path:
    """
    Find the original MR volume in 2resampledNifti.

    Searches for: 2resampledNifti/{patient_id}/MR/*.nii.gz
    """
    mr_dir = RESAMPLED_DIR / patient_id / "MR"
    if mr_dir.exists():
        for f in mr_dir.iterdir():
            if f.suffix == '.gz' or f.suffix == '.nii':
                return f
    return None


def load_nifti_slice(nifti_path: Path) -> tuple:
    """
    Load a 2D NIfTI slice and return the data array and header info.

    Applies inverse transformations to undo the rotation and flip from test_synth.py:
    - Slices were saved with: np.rot90(data, -1) followed by np.fliplr(data)
    - We undo these with: np.fliplr followed by np.rot90(data, 1)

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

    # Undo the transformations applied in test_synth.py
    # Original: np.fliplr(np.rot90(data, -1))
    # Inverse: np.rot90(np.fliplr(data), 1)
    data = np.fliplr(data)
    data = np.rot90(data, 1)

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
    fake_dir: Path,
    output_dir: Path,
    copy_originals: bool = False,
    reconstruct_original: bool = False
) -> bool:
    """
    Process one patient: reconstruct sCT volume at resampled dimensions,
    optionally at original dimensions, and optionally copy original CT/MR files.

    Args:
        patient_id: Patient identifier (e.g., "AB_1ABA009")
        fake_slices: Dict of slice_num -> filename for fake slices
        fake_dir: Directory containing fake NIfTI slices
        output_dir: Base output directory for reconstructions
        copy_originals: Whether to copy original CT/MR files to output
        reconstruct_original: Whether to reconstruct sCT at original dimensions

    Returns:
        True if successful, False otherwise
    """
    patient_out_dir = output_dir / patient_id
    patient_out_dir.mkdir(parents=True, exist_ok=True)

    # Find and load resampled CT for reference affine
    resampled_ct_path = find_original_ct(patient_id)  # from 2resampledNifti
    reference_affine = None

    if resampled_ct_path:
        try:
            resampled_img = nib.load(resampled_ct_path)
            reference_affine = resampled_img.affine

            # Optionally copy resampled CT to output folder
            if copy_originals:
                dest_ct_resampled = patient_out_dir / "CT_256_dim_copy_2resampledNifti.nii.gz"
                if not dest_ct_resampled.exists():
                    shutil.copy2(resampled_ct_path, dest_ct_resampled)
        except Exception as e:
            print(f"  !WARNING! Could not load resampled CT for {patient_id}: {e}")

    # Find and optionally copy resampled MR
    if copy_originals:
        resampled_mr_path = find_original_mr(patient_id)  # from 2resampledNifti
        if resampled_mr_path:
            try:
                dest_mr_resampled = patient_out_dir / "MR_256_dim_copy_2resampledNifti.nii.gz"
                if not dest_mr_resampled.exists():
                    shutil.copy2(resampled_mr_path, dest_mr_resampled)
            except Exception as e:
                print(f"  !WARNING! Could not copy resampled MR for {patient_id}: {e}")

    # Find and optionally copy original (1initNifti) CT and MR
    if copy_originals:
        original_ct_init_path = find_original_ct_init(patient_id)
        if original_ct_init_path:
            try:
                dest_ct_original = patient_out_dir / "CT_original_dim_copy_1initNifti.nii.gz"
                if not dest_ct_original.exists():
                    shutil.copy2(original_ct_init_path, dest_ct_original)
            except Exception as e:
                print(f"  !WARNING! Could not copy original CT from 1initNifti for {patient_id}: {e}")

        original_mr_init_path = find_original_mr_init(patient_id)
        if original_mr_init_path:
            try:
                dest_mr_original = patient_out_dir / "MR_original_dim_copy_1initNifti.nii.gz"
                if not dest_mr_original.exists():
                    shutil.copy2(original_mr_init_path, dest_mr_original)
            except Exception as e:
                print(f"  !WARNING! Could not copy original MR from 1initNifti for {patient_id}: {e}")

    # Optionally create reverse-resampled version of original CT for validation
    # This allows validation of the reverse resampling process
    if reconstruct_original and resampled_ct_path:
        try:
            resampled_ct_img = nib.load(resampled_ct_path)
            reversed_ct = reverse_resample_to_original(resampled_ct_img, patient_id)
            if reversed_ct is not None:
                dest_ct_reversed = patient_out_dir / "CT_pre_and_postprocessed_to_original_dim_mask_not_aligned.nii.gz"
                nib.save(reversed_ct, dest_ct_reversed)
        except Exception as e:
            print(f"  !WARNING! Could not reverse resample original CT for {patient_id}: {e}")

    success = True

    # Reconstruct synthetic volume at resampled resolution (256x256)
    if fake_slices:
        # Create resampled version (always created)
        fake_output_resampled = patient_out_dir / "sCT_256_dim.nii.gz"
        resampled_result = reconstruct_volume_from_nifti(
            patient_id, fake_slices, fake_dir, fake_output_resampled, reference_affine
        )

        if resampled_result is None:
            success = False
        else:
            # Create original dimension version by reversing resampling (always created)
            try:
                original_result = reverse_resample_to_original(resampled_result, patient_id)

                if original_result is not None:
                    fake_output_original = patient_out_dir / "sCT_original_dim_mask_not_aligned.nii.gz"
                    nib.save(original_result, fake_output_original)
                else:
                    print(f"  !WARNING! Could not create original dimension sCT for {patient_id}")
                    success = False
            except Exception as e:
                print(f"  !WARNING! Error during reverse resampling for {patient_id}: {e}")
                success = False

    return success


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Reconstruct 3D NIfTI volumes from 2D NIfTI slices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 80volume_creator.py pix2pix_synthrad_abdomen
  python 80volume_creator.py pix2pix_synthrad_abdomen --copy_originals
  python 80volume_creator.py pix2pix_synthrad_abdomen --reconstruct_original
  python 80volume_creator.py pix2pix_synthrad_abdomen --copy_originals --reconstruct_original

Output Files:
  Always created:
    sCT_256_dim.nii.gz - Reconstructed sCT at 256x256 resolution
    sCT_original_dim_mask_not_aligned.nii.gz - sCT at original dimensions

  With --reconstruct_original:
    CT_pre_and_postprocessed_to_original_dim_mask_not_aligned.nii.gz - Validation volume

  With --copy_originals:
    CT_256_dim_copy_2resampledNifti.nii.gz - Original CT at 256x256
    MR_256_dim_copy_2resampledNifti.nii.gz - Original MR at 256x256
    CT_original_dim_copy_1initNifti.nii.gz - Original CT at original dimensions
    MR_original_dim_copy_1initNifti.nii.gz - Original MR at original dimensions

IMPORTANT: Volumes at original dimensions are NOT aligned with original masks!
           Safe for dosimetric analysis, NOT safe for image-level comparison.
        """
    )

    parser.add_argument(
        "model_region_folder",
        type=str,
        help="Model/region folder name in 9latestTestImages (e.g., pix2pix_synthrad_abdomen)"
    )

    parser.add_argument(
        "--copy_originals",
        action="store_true",
        default=False,
        help="Copy original CT/MR files to output folder (default: False)"
    )

    parser.add_argument(
        "--reconstruct_original",
        action="store_true",
        default=False,
        help="Reconstruct sCT at original dimensions via reverse resampling (default: False)"
    )

    # Show available folders if no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        print("\nAvailable folders in 9latestTestImages:")
        if TEST_IMAGES_DIR.exists():
            for d in sorted(TEST_IMAGES_DIR.iterdir()):
                if d.is_dir() and not d.name.startswith('.'):
                    if (d / "fake_nifti").exists():
                        print(f"  - {d.name}")
        sys.exit(1)

    args = parser.parse_args()
    model_region = args.model_region_folder

    # Paths
    model_dir = TEST_IMAGES_DIR / model_region
    fake_dir = model_dir / "fake_nifti"
    output_dir = model_dir / "reconstruction"

    # Validate input
    if not model_dir.exists():
        print(f"ERROR: Model/region directory not found: {model_dir}")
        sys.exit(1)

    if not fake_dir.exists():
        print(f"ERROR: fake_nifti directory not found: {fake_dir}")
        sys.exit(1)

    print("=" * 80)
    print("Volume Reconstruction from NIfTI Slices")
    print("=" * 80)
    print(f"Model/Region:           {model_region}")
    print(f"Fake slices:            {fake_dir}")
    print(f"Resampled data:         {RESAMPLED_DIR}")
    print(f"Original data:          {INIT_DIR}")
    print(f"Output:                 {output_dir}")
    print(f"Copy originals:         {args.copy_originals}")
    print(f"Reconstruct original:   {args.reconstruct_original}")
    print("=" * 80)
    print()

    # Display alignment warning
    print("!" * 80)
    print("IMPORTANT ALIGNMENT WARNING:")
    print("!" * 80)
    print("The reconstructed volumes at original dimensions are NOT aligned with the")
    print("original masks! This is due to resampling transformations that may introduce")
    print("slight spatial misalignments.")
    print()
    print("  ✓ SAFE for:     Dosimetric analysis")
    print("  ✗ NOT SAFE for: Image-level comparison with masked regions")
    print("!" * 80)
    print()

    # Scan for NIfTI slices
    fake_slices = defaultdict(dict)  # patient_id -> {slice_num: filename}

    # Process fake slices
    for filepath in sorted(fake_dir.glob("*.nii*")):
        parsed = parse_nifti_slice_filename(filepath.name)
        if parsed is None:
            continue
        patient_id, slice_num = parsed
        fake_slices[patient_id][slice_num] = filepath.name

    if not fake_slices:
        print("ERROR: No NIfTI slices found in fake_nifti!")
        print("Expected filename format: AB_1ABC123_42.nii")
        sys.exit(1)

    # Get all unique patient IDs
    all_patients = set(fake_slices.keys())

    print(f"Found {len(all_patients)} patients")
    print(f"  - Fake slices: {sum(len(v) for v in fake_slices.values())} total")
    print()

    # Process each patient
    success_count = 0
    for patient_id in tqdm(sorted(all_patients), desc="Reconstructing volumes"):
        success = process_patient(
            patient_id,
            fake_slices.get(patient_id, {}),
            fake_dir,
            output_dir,
            copy_originals=args.copy_originals,
            reconstruct_original=args.reconstruct_original
        )
        if success:
            success_count += 1

    print()
    print("=" * 80)
    print(f"✓ Reconstruction complete!")
    print(f"  Processed: {success_count}/{len(all_patients)} patients")
    print(f"  Output: {output_dir}")
    print()
    print("Each patient folder contains:")
    print()
    print("  Always created:")
    print("    - sCT_256_dim.nii.gz")
    print("      Reconstructed sCT at 256x256 resolution")
    print("    - sCT_original_dim_mask_not_aligned.nii.gz")
    print("      Reconstructed sCT at original patient-specific dimensions")
    print()

    if args.reconstruct_original:
        print("  Validation output (--reconstruct_original enabled):")
        print("    - CT_pre_and_postprocessed_to_original_dim_mask_not_aligned.nii.gz")
        print("      Validation: Original CT resampled to 256x256 then back to original")
        print()

    if args.copy_originals:
        print(f"  Copied original files (--copy_originals enabled):")
        print(f"    - CT_256_dim_copy_2resampledNifti.nii.gz")
        print(f"      Original CT from 2resampledNifti at 256x256 resolution")
        print(f"    - MR_256_dim_copy_2resampledNifti.nii.gz")
        print(f"      Original MR from 2resampledNifti at 256x256 resolution")
        print(f"    - CT_original_dim_copy_1initNifti.nii.gz")
        print(f"      Original CT from 1initNifti at original dimensions")
        print(f"    - MR_original_dim_copy_1initNifti.nii.gz")
        print(f"      Original MR from 1initNifti at original dimensions")
        print()

    if args.reconstruct_original:
        print("!" * 80)
        print("REMINDER: Volumes at original dimensions are NOT aligned with original masks!")
        print("          Safe for dosimetric analysis, NOT safe for image-level comparison.")
        print("!" * 80)

    print("=" * 80)


if __name__ == "__main__":
    main()
