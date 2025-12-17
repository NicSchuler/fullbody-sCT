"""
N-Peaks Standardization using TotalSegmentator Masks

This script performs N-Peaks intensity normalization on MR images using 
pre-computed anatomical masks from TotalSegmentator (liver, torso fat, 
subcutaneous fat).

N-Peaks works by finding peaks in the intensity histogram of homogeneous 
tissue regions (defined by masks) and transforming the intensity scale to 
align those peaks to target values.

Prerequisites:
    - Run 20resampling.py to create resampled CT/MR images
    - Run 22resample_totalsegmentator_masks.py to create resampled masks

Usage:    
# Run on all abdomen patients (default)
python 34npeaks_standardization.py

# Run on single test patient (AB_1ABA005) with verbose output
python 34npeaks_standardization.py --test
# or
python 34npeaks_standardization.py -t

# Run on a specific patient with verbose output
python 34npeaks_standardization.py --patient AB_1ABA009
# or
python 34npeaks_standardization.py -p AB_1ABA009
"""

import os
import sys
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

# Add the npeaks_normalization folder to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'npeaks_normalization'))

from npeaks_normalization.npeaks_normalize import NPeaksNormalizer
from npeaks_normalization.npeaks_util import calculate_voxel_spacing_from_affine

# ============================================================================
# Configuration
# ============================================================================

# Data paths
DATA_BASE = Path("/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed")
RESAMPLED_DIR = DATA_BASE / "2resampledNifti"
OUTPUT_DIR = DATA_BASE / "34npeaks"

# Masks to use for normalization (from TotalSegmentator)
# These should be available in 2resampledNifti/{patient}/totalsegmentator_masks/
MASK_CONFIG = {
    # Mask name (without _cp256.nii.gz suffix) -> (target_intensity, peak_selection_strategy)
    # Fat is bright on T1, so we use "right" (highest peak)
    # Liver has intermediate intensity on T1
    "subcutaneous_fat": {
        "target_intensity": 0.8,
        "peak_strategy": "right",  # Fat is bright on T1
        "description": "Subcutaneous fat (bright on T1)"
    },
    "torso_fat": {
        "target_intensity": 0.75,
        "peak_strategy": "right",  # Fat is bright on T1  
        "description": "Torso/visceral fat (bright on T1)"
    },
    "liver": {
        "target_intensity": 0.5,
        "peak_strategy": "most",  # Most prominent peak in liver
        "description": "Liver (intermediate on T1)"
    },
}

# Which masks to actually use for normalization (can be subset of MASK_CONFIG)
ACTIVE_MASKS = ["subcutaneous_fat", "torso_fat"]  # Using fat masks for now

# Minimum number of voxels required in a mask
MIN_MASK_VOXELS = 1000

# Whether to save intermediate masks for debugging
SAVE_DEBUG_MASKS = True


def load_nifti(path):
    """Load a NIfTI file and return the data array, affine, and header."""
    nii = nib.load(str(path))
    return nii.get_fdata(), nii.affine, nii.header


def save_nifti(data, affine, header, output_path):
    """Save a numpy array as a NIfTI file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nii = nib.Nifti1Image(data.astype(np.float32), affine, header)
    nib.save(nii, str(output_path))


def load_totalsegmentator_mask(patient_dir: Path, mask_name: str) -> np.ndarray:
    """
    Load a resampled TotalSegmentator mask.
    
    Args:
        patient_dir: Path to patient directory in 2resampledNifti
        mask_name: Name of mask (e.g., 'liver', 'torso_fat', 'subcutaneous_fat')
    
    Returns:
        Boolean mask array, or None if not found
    """
    mask_path = patient_dir / "totalsegmentator_masks" / f"{mask_name}_cp256.nii.gz"
    
    if not mask_path.exists():
        return None
    
    mask_data, _, _ = load_nifti(mask_path)
    return mask_data.astype(bool)


def get_available_masks(patient_dir: Path) -> list:
    """Get list of available TotalSegmentator masks for a patient."""
    ts_dir = patient_dir / "totalsegmentator_masks"
    if not ts_dir.exists():
        return []
    
    available = []
    for mask_name in MASK_CONFIG.keys():
        mask_path = ts_dir / f"{mask_name}_cp256.nii.gz"
        if mask_path.exists():
            available.append(mask_name)
    
    return available


def normalize_patient(patient_id: str, verbose: bool = True) -> bool:
    """
    Perform N-Peaks normalization on a single patient.
    
    Args:
        patient_id: Patient identifier (e.g., 'AB_1ABA005')
        verbose: Whether to print detailed progress
    
    Returns:
        True if successful, False otherwise
    """
    patient_dir = RESAMPLED_DIR / patient_id
    
    if not patient_dir.exists():
        if verbose:
            print(f"[SKIP] {patient_id}: directory not found")
        return False
    
    # Build file paths
    mr_path = patient_dir / "MR" / f"{patient_id}_MR_cp256.nii.gz"
    body_mask_path = patient_dir / "new_masks" / f"{patient_id}_mask_from_CT_treshold_cp256.nii.gz"
    
    # Check files exist
    if not mr_path.exists():
        if verbose:
            print(f"[SKIP] {patient_id}: MR not found at {mr_path}")
        return False
    
    if not body_mask_path.exists():
        if verbose:
            print(f"[SKIP] {patient_id}: Body mask not found")
        return False
    
    # Check for TotalSegmentator masks
    available_masks = get_available_masks(patient_dir)
    masks_to_use = [m for m in ACTIVE_MASKS if m in available_masks]
    
    if not masks_to_use:
        if verbose:
            print(f"[SKIP] {patient_id}: No TotalSegmentator masks available")
            print(f"  Available: {available_masks}")
            print(f"  Required: {ACTIVE_MASKS}")
        return False
    
    if verbose:
        print(f"\nProcessing {patient_id}")
        print(f"  Using masks: {masks_to_use}")
    
    # Load MR image
    mr_data, mr_affine, mr_header = load_nifti(mr_path)
    voxel_spacing = calculate_voxel_spacing_from_affine(mr_affine)
    
    if verbose:
        print(f"  MR shape: {mr_data.shape}")
        print(f"  MR intensity range: [{mr_data.min():.2f}, {mr_data.max():.2f}]")
        print(f"  Voxel spacing: {voxel_spacing}")
    
    # Load body mask for applying to output
    body_mask, _, _ = load_nifti(body_mask_path)
    body_mask = body_mask.astype(bool)
    
    # Load anatomical masks
    mask_arr_list = []
    target_intensity_list = []
    peak_strategy_list = []
    mask_names_used = []
    
    for mask_name in masks_to_use:
        mask_data = load_totalsegmentator_mask(patient_dir, mask_name)
        
        if mask_data is None:
            if verbose:
                print(f"  [WARN] Could not load mask: {mask_name}")
            continue
        
        # Apply body mask intersection
        mask_data = mask_data & body_mask
        
        # Check minimum voxels
        n_voxels = mask_data.sum()
        if n_voxels < MIN_MASK_VOXELS:
            if verbose:
                print(f"  [WARN] {mask_name}: only {n_voxels} voxels (min: {MIN_MASK_VOXELS})")
            continue
        
        mask_arr_list.append(mask_data)
        config = MASK_CONFIG[mask_name]
        target_intensity_list.append(config["target_intensity"])
        peak_strategy_list.append(config["peak_strategy"])
        mask_names_used.append(mask_name)
        
        if verbose:
            print(f"  Loaded {mask_name}: {n_voxels} voxels")
    
    if not mask_arr_list:
        if verbose:
            print(f"[SKIP] {patient_id}: No valid masks after filtering")
        return False
    
    # Create normalizer
    normalizer = NPeaksNormalizer()
    
    try:
        # Find peaks in the masks
        if verbose:
            print(f"  Finding peaks in {len(mask_arr_list)} mask(s)...")
        
        peak_intensities = normalizer.find_peaks(
            intensity_arr=mr_data,
            mask_arr_list=mask_arr_list,
            voxel_spacing_arr=voxel_spacing,
            peak_detection_peak_selection_strategy_list=peak_strategy_list,
            image_name=patient_id
        )
        
        if verbose:
            for name, peak in zip(mask_names_used, peak_intensities):
                print(f"    {name} peak: {peak:.4f}")
        
        # Transform the intensity scale
        if verbose:
            print("  Transforming intensity scale...")
        
        normalized_data = normalizer.transform_intensity_scale(
            intensity_arr=mr_data,
            determined_peak_intensity_list=peak_intensities,
            target_peak_intensity_list=target_intensity_list
        )
        
        if verbose:
            print(f"  Normalized range: [{normalized_data.min():.4f}, {normalized_data.max():.4f}]")
        
        # Save the normalized image
        output_mr_dir = OUTPUT_DIR / "normalized" / patient_id / "MR"
        output_path = output_mr_dir / f"{patient_id}_MR_npeaks.nii.gz"
        save_nifti(normalized_data, mr_affine, mr_header, output_path)
        
        if verbose:
            print(f"  Saved: {output_path}")
        
        # Save debug masks if requested
        if SAVE_DEBUG_MASKS:
            debug_dir = OUTPUT_DIR / "debug_masks" / patient_id
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            for mask_name, mask_arr in zip(mask_names_used, mask_arr_list):
                mask_path = debug_dir / f"{patient_id}_{mask_name}_mask.nii.gz"
                save_nifti(mask_arr.astype(np.float32), mr_affine, mr_header, mask_path)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] {patient_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main entry point for N-Peaks normalization.
    Processes only abdomen (AB_*) patients.
    """
    print("=" * 60)
    print("N-Peaks Normalization using TotalSegmentator Masks")
    print("=" * 60)
    print(f"Input: {RESAMPLED_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Active masks: {ACTIVE_MASKS}")
    print()
    
    # Find only abdomen patients (AB_* prefix)
    patients = sorted([d.name for d in RESAMPLED_DIR.iterdir() 
                       if d.is_dir() and d.name.startswith("AB_")])
    print(f"Found {len(patients)} abdomen patients")
    
    # Process all patients
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for patient_id in tqdm(patients, desc="Normalizing"):
        result = normalize_patient(patient_id, verbose=False)
        if result:
            success_count += 1
        else:
            skip_count += 1
    
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Successful: {success_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Errors: {error_count}")


def test_single_patient(patient_id: str = "AB_1ABA005"):
    """
    Test normalization on a single patient with verbose output.
    """
    print("=" * 60)
    print(f"Testing N-Peaks Normalization on {patient_id}")
    print("=" * 60)
    
    result = normalize_patient(patient_id, verbose=True)
    
    if result:
        print("\n✓ Normalization completed successfully!")
    else:
        print("\n✗ Normalization failed or skipped")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="N-Peaks normalization using TotalSegmentator masks"
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Run on single test patient (AB_1ABA005) with verbose output"
    )
    parser.add_argument(
        "--patient", "-p",
        type=str,
        default=None,
        help="Run on a specific patient ID with verbose output"
    )
    
    args = parser.parse_args()
    
    if args.patient:
        # Run on specific patient
        test_single_patient(args.patient)
    elif args.test:
        # Run on default test patient
        test_single_patient("AB_1ABA005")
    else:
        # Run on all abdomen patients
        main()