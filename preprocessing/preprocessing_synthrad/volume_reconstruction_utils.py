#!/usr/bin/env python

"""
volume_reconstruction_utils.py

Shared utility functions for volume reconstruction scripts.
Used by 81sct_volume_reconstructor.py and 82resampled_to_original.py.
"""

from pathlib import Path

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
