#!/usr/bin/env python

"""
81sct_volume_reconstructor.py

Reconstructs 3D NIfTI volumes from 2D NIfTI slices at original dimensions.

Usage:
    python 81sct_volume_reconstructor.py <model_region_folder> [--copy_originals]

Arguments:
    model_region_folder     Model/region folder name in 9latestTestImages (e.g., pix2pix_synthrad_allregion_final/test_50)
    --copy_originals        Optional: Copy original CT/MR files to output folder (default: False)

Examples:
    python 81sct_volume_reconstructor.py pix2pix_synthrad_allregion_final/test_50
    python 81sct_volume_reconstructor.py pix2pix_synthrad_allregion_final/test_50 --copy_originals

This script:
    1. Reads NIfTI slices from 9latestTestImages/<model_region>/fake_nifti/
    2. Groups slices by patient ID
    3. Stacks slices into 3D volumes
    4. Applies reverse resampling to create volumes at original 1initNifti dimensions
    5. Optionally copies original CT/MR files for comparison

Output Files:
    Always created:
    - sCT_original_dim_reconstructed_alignment.nii.gz (sCT at original dimensions)

    Created when --copy_originals is set:
    - CT_original_dim_copy_1initNifti.nii.gz (original CT at original dimensions from 1initNifti)
    - MR_original_dim_copy_1initNifti.nii.gz (original MR at original dimensions from 1initNifti)

IMPORTANT ALIGNMENT NOTE:
    The reconstructed volumes use "reconstructed_alignment" naming to indicate that the
    spatial alignment has been reconstructed from the resampled slices. While dimensions
    match the original 1initNifti volumes, slight spatial differences may exist due to
    the resampling transformations.

# Batch processing in shell for a list of models:
BASE=/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/9latestTestImages
folders=(
    pix2pix_synthrad_abdomen_final
    pix2pix_synthrad_brain_final
    pix2pix_synthrad_headneck_final
    pix2pix_synthrad_pelvis_final
    pix2pix_synthrad_thorax_final
    pix2pix_synthrad_allregion_final
    cyclegan_abdomen_final
    cyclegan_brain_final
    cyclegan_head_neck_final
    cyclegan_pelvis_final
    cyclegan_thorax_final
    cyclegan_allregions_final
    cut_synthrad_abdomen_final
    cut_synthrad_brain_final
    cut_synthrad_HN_final
    cut_synthrad_pelvis_final
    cut_synthrad_TH_final
    cut_synthrad_allregions_final
)
for f in "${folders[@]}"; do
    echo "Running: $f"
    python 81sct_volume_reconstructor.py "$f/test_50"
done
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

from volume_reconstruction_utils import (
    BASE_ROOT,
    TEST_IMAGES_DIR,
    RESAMPLED_DIR,
    INIT_DIR,
    find_original_ct_init,
    find_original_mr_init,
    reverse_resample_to_original,
)


def parse_nifti_slice_filename(filename: str):
    """
    Parse NIfTI slice filename to extract patient_id and slice_num.

    Example:
    - AB_1ABA009_38.nii
    - HN_1HNAxxx_5.nii
    - *.nii.gz supported

    Returns: (patient_id, slice_num) or None
    """
    pattern = r'^([A-Za-z]+_\d[A-Z]{1,3}(?:\d{2,3}|xxx))_(\d+)\.nii(\.gz)?$'
    match = re.match(pattern, filename)

    if not match:
        return None

    patient_id = match.group(1)
    slice_num = int(match.group(2))

    return patient_id, slice_num


def find_resampled_ct(patient_id: str, resampled_dir: Path = None) -> Path:
    """
    Find the CT volume in 2resampledNifti for reference affine.

    Searches for: 2resampledNifti/{patient_id}/CT_reg/*.nii.gz

    Args:
        patient_id: Patient identifier
        resampled_dir: Optional custom resampled directory (default: RESAMPLED_DIR)
    """
    _resampled_dir = resampled_dir if resampled_dir is not None else RESAMPLED_DIR
    ct_dir = _resampled_dir / patient_id / "CT_reg"
    if ct_dir.exists():
        for f in ct_dir.iterdir():
            if f.suffix == '.gz' or f.suffix == '.nii':
                return f
    return None


def find_resampled_mr(patient_id: str, resampled_dir: Path = None) -> Path:
    """
    Find the MR volume in 2resampledNifti for reference affine.

    Searches for: 2resampledNifti/{patient_id}/MR/*.nii.gz

    Args:
        patient_id: Patient identifier
        resampled_dir: Optional custom resampled directory (default: RESAMPLED_DIR)
    """
    _resampled_dir = resampled_dir if resampled_dir is not None else RESAMPLED_DIR
    mr_dir = _resampled_dir / patient_id / "MR"
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
    init_dir: Path = None,
    resampled_dir: Path = None,
    use_mr_reference: bool = False
) -> bool:
    """
    Process one patient: reconstruct sCT volume at original dimensions,
    and optionally copy original CT/MR files.

    Args:
        patient_id: Patient identifier (e.g., "AB_1ABA009")
        fake_slices: Dict of slice_num -> filename for fake slices
        fake_dir: Directory containing fake NIfTI slices
        output_dir: Base output directory for reconstructions
        copy_originals: Whether to copy original CT/MR files to output
        init_dir: Optional custom init directory for original volumes
        resampled_dir: Optional custom resampled directory for reference affine
        use_mr_reference: If True, use MR as reference instead of CT (for inference)

    Returns:
        True if successful, False otherwise
    """
    patient_out_dir = output_dir / patient_id
    patient_out_dir.mkdir(parents=True, exist_ok=True)

    # Find and load resampled volume for reference affine
    reference_affine = None
    if use_mr_reference:
        # For inference: prefer MR, fallback to CT
        resampled_path = find_resampled_mr(patient_id, resampled_dir)
        if resampled_path is None:
            resampled_path = find_resampled_ct(patient_id, resampled_dir)
    else:
        # For training/validation: prefer CT, fallback to MR
        resampled_path = find_resampled_ct(patient_id, resampled_dir)
        if resampled_path is None:
            resampled_path = find_resampled_mr(patient_id, resampled_dir)

    if resampled_path:
        try:
            resampled_img = nib.load(resampled_path)
            reference_affine = resampled_img.affine
        except Exception as e:
            print(f"  !WARNING! Could not load resampled reference for {patient_id}: {e}")

    # Find and optionally copy original (1initNifti) CT and MR
    if copy_originals:
        original_ct_init_path = find_original_ct_init(patient_id, init_dir)
        if original_ct_init_path:
            try:
                dest_ct_original = patient_out_dir / "CT_original_dim_copy_1initNifti.nii.gz"
                if not dest_ct_original.exists():
                    shutil.copy2(original_ct_init_path, dest_ct_original)
            except Exception as e:
                print(f"  !WARNING! Could not copy original CT from 1initNifti for {patient_id}: {e}")

        original_mr_init_path = find_original_mr_init(patient_id, init_dir)
        if original_mr_init_path:
            try:
                dest_mr_original = patient_out_dir / "MR_original_dim_copy_1initNifti.nii.gz"
                if not dest_mr_original.exists():
                    shutil.copy2(original_mr_init_path, dest_mr_original)
            except Exception as e:
                print(f"  !WARNING! Could not copy original MR from 1initNifti for {patient_id}: {e}")

    success = True

    # Reconstruct synthetic volume from slices
    if fake_slices:
        # First reconstruct at resampled resolution (temporary, not saved)
        # We need this intermediate step for the reverse_resample_to_original function
        temp_output = patient_out_dir / "_temp_resampled.nii.gz"
        resampled_result = reconstruct_volume_from_nifti(
            patient_id, fake_slices, fake_dir, temp_output, reference_affine
        )

        if resampled_result is None:
            success = False
        else:
            # Create original dimension version by reversing resampling
            try:
                original_result = reverse_resample_to_original(
                    resampled_result, patient_id,
                    init_dir=init_dir, use_mr_reference=use_mr_reference
                )

                if original_result is not None:
                    fake_output_original = patient_out_dir / "sCT_original_dim_reconstructed_alignment.nii.gz"
                    nib.save(original_result, fake_output_original)
                else:
                    print(f"  !WARNING! Could not create original dimension sCT for {patient_id}")
                    success = False
            except Exception as e:
                print(f"  !WARNING! Error during reverse resampling for {patient_id}: {e}")
                success = False
            finally:
                # Clean up temporary file
                if temp_output.exists():
                    temp_output.unlink()

    return success


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Reconstruct 3D NIfTI volumes from 2D NIfTI slices at original dimensions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training/validation mode (uses CT as reference):
  python 81sct_volume_reconstructor.py pix2pix_synthrad_abdomen/test_50
  python 81sct_volume_reconstructor.py pix2pix_synthrad_abdomen/test_50 --copy_originals

  # Inference mode (uses MR as reference, custom directories):
  python 81sct_volume_reconstructor.py model_name/test_50 \\
      --test-images-dir /path/to/9inference \\
      --init-dir /path/to/1initNifti \\
      --resampled-dir /path/to/2resampledNifti \\
      --use-mr-reference

Output Files:
  Always created:
    sCT_original_dim_reconstructed_alignment.nii.gz - Reconstructed sCT at original dimensions

  With --copy_originals:
    CT_original_dim_copy_1initNifti.nii.gz - Original CT from 1initNifti at original dimensions
    MR_original_dim_copy_1initNifti.nii.gz - Original MR from 1initNifti at original dimensions

IMPORTANT: The "reconstructed_alignment" naming indicates that spatial alignment has been
           reconstructed from resampled slices. Dimensions match original volumes.
        """
    )

    parser.add_argument(
        "model_region_folder",
        type=str,
        help="Model/region folder name in test images dir (e.g., pix2pix_synthrad_abdomen/test_50)"
    )

    parser.add_argument(
        "--copy_originals",
        action="store_true",
        default=False,
        help="Copy original CT/MR files to output folder (default: False)"
    )

    parser.add_argument(
        "--test-images-dir",
        type=str,
        default=None,
        help=f"Override test images directory (default: {TEST_IMAGES_DIR})"
    )

    parser.add_argument(
        "--init-dir",
        type=str,
        default=None,
        help=f"Override init directory for original volumes (default: {INIT_DIR})"
    )

    parser.add_argument(
        "--resampled-dir",
        type=str,
        default=None,
        help=f"Override resampled directory for reference affine (default: {RESAMPLED_DIR})"
    )

    parser.add_argument(
        "--use-mr-reference",
        action="store_true",
        default=False,
        help="Use MR as reference instead of CT (for inference mode)"
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

    # Determine directories
    test_images_dir = Path(args.test_images_dir) if args.test_images_dir else TEST_IMAGES_DIR
    init_dir = Path(args.init_dir) if args.init_dir else INIT_DIR
    resampled_dir = Path(args.resampled_dir) if args.resampled_dir else RESAMPLED_DIR

    # Paths
    model_dir = test_images_dir / model_region
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
    print("sCT Volume Reconstruction from NIfTI Slices")
    print("=" * 80)
    print(f"Model/Region:           {model_region}")
    print(f"Test images dir:        {test_images_dir}")
    print(f"Fake slices:            {fake_dir}")
    print(f"Init dir (originals):   {init_dir}")
    print(f"Resampled dir:          {resampled_dir}")
    print(f"Output:                 {output_dir}")
    print(f"Copy originals:         {args.copy_originals}")
    print(f"Use MR reference:       {args.use_mr_reference}")
    print("=" * 80)
    print()

    # Display alignment note
    print("-" * 80)
    print("ALIGNMENT NOTE:")
    print("-" * 80)
    print("Output files use 'reconstructed_alignment' naming to indicate that spatial")
    print("alignment has been reconstructed from resampled slices. Dimensions match")
    print("the original 1initNifti volumes.")
    print("-" * 80)
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
            init_dir=init_dir,
            resampled_dir=resampled_dir,
            use_mr_reference=args.use_mr_reference
        )
        if success:
            success_count += 1

    print()
    print("=" * 80)
    print(f"Reconstruction complete!")
    print(f"  Processed: {success_count}/{len(all_patients)} patients")
    print(f"  Output: {output_dir}")
    print()
    print("Each patient folder contains:")
    print()
    print("  Always created:")
    print("    - sCT_original_dim_reconstructed_alignment.nii.gz")
    print("      Reconstructed sCT at original patient-specific dimensions")
    print()

    if args.copy_originals:
        print(f"  Copied original files (--copy_originals enabled):")
        print(f"    - CT_original_dim_copy_1initNifti.nii.gz")
        print(f"      Original CT from 1initNifti at original dimensions")
        print(f"    - MR_original_dim_copy_1initNifti.nii.gz")
        print(f"      Original MR from 1initNifti at original dimensions")
        print()

    print("=" * 80)


if __name__ == "__main__":
    main()
