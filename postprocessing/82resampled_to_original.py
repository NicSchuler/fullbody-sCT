#!/usr/bin/env python

"""
82resampled_to_original.py

Converts all volumes in 2resampledNifti back to original dimensions.

Usage:
    python 82resampled_to_original.py [--output_dir <path>] [--patients <id1,id2,...>]

Arguments:
    --output_dir    Output directory (default: 2resampledNifti_reconstructed_dims)
    --patients      Optional patient filter. Process only these patient IDs.

Examples:
    python 82resampled_to_original.py
    python 82resampled_to_original.py --output_dir /path/to/output
    python 82resampled_to_original.py --patients AB_1ABA009
    python 82resampled_to_original.py --patients AB_1ABA009 AB_1ABA010
    python 82resampled_to_original.py --patients AB_1ABA009,AB_1ABA010

This script:
    1. Iterates over all patient folders in 2resampledNifti
    2. Processes all .nii.gz files in each subfolder (CT_reg, MR, new_masks, totalsegmentator_masks)
    3. Applies reverse resampling to restore original 1initNifti dimensions
    4. Saves to output directory with _cp256 replaced by _original_dim_reconstructed_alignment

Output Structure:
    <output_dir>/
    ├── {patient_id}/
    │   ├── CT_reg/
    │   │   └── {patient_id}_CT_reg_original_dim_reconstructed_alignment.nii.gz
    │   ├── MR/
    │   │   └── {patient_id}_MR_original_dim_reconstructed_alignment.nii.gz
    │   ├── new_masks/
    │   │   └── {patient_id}_mask_from_CT_treshold_original_dim_reconstructed_alignment.nii.gz
    │   └── totalsegmentator_masks/
    │       ├── liver_original_dim_reconstructed_alignment.nii.gz
    │       ├── skeletal_muscle_original_dim_reconstructed_alignment.nii.gz
    │       ├── subcutaneous_fat_original_dim_reconstructed_alignment.nii.gz
    │       └── torso_fat_original_dim_reconstructed_alignment.nii.gz

IMPORTANT ALIGNMENT NOTE:
    The output volumes use "reconstructed_alignment" naming to indicate that the
    spatial alignment has been reconstructed from the resampled volumes. While dimensions
    match the original 1initNifti volumes, slight spatial differences may exist due to
    the resampling transformations.
"""

import sys
import argparse
from pathlib import Path
from tqdm import tqdm

import nibabel as nib

from volume_reconstruction_utils import (
    BASE_ROOT,
    RESAMPLED_DIR,
    INIT_DIR,
    reverse_resample_to_original,
)


def rename_file_for_output(filename: str) -> str:
    """
    Rename file by replacing _cp256 with _original_dim_reconstructed_alignment.

    Examples:
        AB_1ABA005_CT_reg_cp256.nii.gz -> AB_1ABA005_CT_reg_original_dim_reconstructed_alignment.nii.gz
        liver_cp256.nii.gz -> liver_original_dim_reconstructed_alignment.nii.gz
    """
    return filename.replace("_cp256", "_original_dim_reconstructed_alignment")


def process_patient_folder(
    patient_id: str,
    input_patient_dir: Path,
    output_patient_dir: Path
):
    """
    Process all NIfTI files in a patient folder.

    Args:
        patient_id: Patient identifier (e.g., "AB_1ABA009")
        input_patient_dir: Path to patient folder in 2resampledNifti
        output_patient_dir: Path to patient folder in output directory

    Returns:
        Tuple of (successful_count, total_count)
    """
    successful = 0
    total = 0

    # Process all subfolders
    for subfolder in input_patient_dir.iterdir():
        if not subfolder.is_dir():
            continue

        output_subfolder = output_patient_dir / subfolder.name
        output_subfolder.mkdir(parents=True, exist_ok=True)

        # Process all NIfTI files in subfolder
        for nifti_file in subfolder.glob("*.nii*"):
            total += 1

            try:
                # Load the resampled volume
                img = nib.load(nifti_file)

                # Apply reverse resampling
                restored_img = reverse_resample_to_original(img, patient_id)

                if restored_img is not None:
                    # Generate output filename
                    output_filename = rename_file_for_output(nifti_file.name)
                    output_path = output_subfolder / output_filename

                    # Save
                    nib.save(restored_img, output_path)
                    successful += 1
                else:
                    print(f"  !WARNING! Could not restore {nifti_file.name} for {patient_id}")

            except Exception as e:
                print(f"  !WARNING! Error processing {nifti_file}: {e}")

    return successful, total


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Convert all volumes in 2resampledNifti back to original dimensions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 82resampled_to_original.py
  python 82resampled_to_original.py --output_dir /path/to/output
  python 82resampled_to_original.py --patients AB_1ABA009
  python 82resampled_to_original.py --patients AB_1ABA009 AB_1ABA010
  python 82resampled_to_original.py --patients AB_1ABA009,AB_1ABA010

Output Structure:
  <output_dir>/{patient_id}/
    CT_reg/{patient_id}_CT_reg_original_dim_reconstructed_alignment.nii.gz
    MR/{patient_id}_MR_original_dim_reconstructed_alignment.nii.gz
    new_masks/{patient_id}_mask_from_CT_treshold_original_dim_reconstructed_alignment.nii.gz
    totalsegmentator_masks/*.nii.gz

IMPORTANT: The "reconstructed_alignment" naming indicates that spatial alignment has been
           reconstructed from resampled volumes. Dimensions match original 1initNifti volumes.
        """
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: 2resampledNifti_reconstructed_dims in BASE_ROOT)"
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=None,
        help=(
            "Optional patient ID filter. Accepts space-separated and/or comma-separated "
            "values, e.g. --patients AB_1ABA009 AB_1ABA010 or --patients AB_1ABA009,AB_1ABA010"
        ),
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = BASE_ROOT / "2resampledNifti_reconstructed_dims"

    # Validate input
    if not RESAMPLED_DIR.exists():
        print(f"ERROR: 2resampledNifti directory not found: {RESAMPLED_DIR}")
        sys.exit(1)

    # Get all patient folders
    all_patient_folders = sorted([
        d for d in RESAMPLED_DIR.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])

    if not all_patient_folders:
        print(f"ERROR: No patient folders found in {RESAMPLED_DIR}")
        sys.exit(1)

    # Optional patient filtering
    requested_patients = None
    if args.patients:
        requested_patients = set()
        for token in args.patients:
            for part in token.split(","):
                pid = part.strip()
                if pid:
                    requested_patients.add(pid)

    if requested_patients:
        available_by_name = {p.name: p for p in all_patient_folders}
        patient_folders = [available_by_name[pid] for pid in sorted(requested_patients) if pid in available_by_name]
        missing = sorted(requested_patients - set(available_by_name.keys()))
        if missing:
            print(f"WARNING: {len(missing)} requested patient(s) not found in {RESAMPLED_DIR}:")
            for pid in missing:
                print(f"  - {pid}")
            print()
        if not patient_folders:
            print("ERROR: None of the requested patients were found. Nothing to process.")
            sys.exit(1)
    else:
        patient_folders = all_patient_folders

    print("=" * 80)
    print("Convert 2resampledNifti to Original Dimensions")
    print("=" * 80)
    print(f"Input:                  {RESAMPLED_DIR}")
    print(f"Reference data:         {INIT_DIR}")
    print(f"Output:                 {output_dir}")
    print(f"Patients to process:    {len(patient_folders)}")
    if requested_patients:
        print("Patient filter:         enabled")
    print("=" * 80)
    print()

    # Display alignment note
    print("-" * 80)
    print("ALIGNMENT NOTE:")
    print("-" * 80)
    print("Output files use 'reconstructed_alignment' naming to indicate that spatial")
    print("alignment has been reconstructed from resampled volumes. Dimensions match")
    print("the original 1initNifti volumes.")
    print("-" * 80)
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each patient
    total_successful = 0
    total_files = 0
    patient_success_count = 0

    for patient_dir in tqdm(patient_folders, desc="Processing patients"):
        patient_id = patient_dir.name
        output_patient_dir = output_dir / patient_id

        successful, total = process_patient_folder(
            patient_id,
            patient_dir,
            output_patient_dir
        )

        total_successful += successful
        total_files += total

        if successful > 0:
            patient_success_count += 1

    print()
    print("=" * 80)
    print(f"Conversion complete!")
    print(f"  Patients processed:   {patient_success_count}/{len(patient_folders)}")
    print(f"  Files converted:      {total_successful}/{total_files}")
    print(f"  Output:               {output_dir}")
    print()
    print("Output structure for each patient:")
    print("  {patient_id}/")
    print("    CT_reg/{patient_id}_CT_reg_original_dim_reconstructed_alignment.nii.gz")
    print("    MR/{patient_id}_MR_original_dim_reconstructed_alignment.nii.gz")
    print("    new_masks/{patient_id}_mask_from_CT_treshold_original_dim_reconstructed_alignment.nii.gz")
    print("    totalsegmentator_masks/")
    print("      liver_original_dim_reconstructed_alignment.nii.gz")
    print("      skeletal_muscle_original_dim_reconstructed_alignment.nii.gz")
    print("      subcutaneous_fat_original_dim_reconstructed_alignment.nii.gz")
    print("      torso_fat_original_dim_reconstructed_alignment.nii.gz")
    print("=" * 80)


if __name__ == "__main__":
    main()
