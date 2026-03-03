#!/usr/bin/env python3
import argparse
import os
import sys
import shutil
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk

# Add npeaks module to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / 'npeaks_normalization'))

from npeaks_normalization.npeaks_normalize import NPeaksNormalizer
from npeaks_normalization.npeaks_util import calculate_voxel_spacing_from_affine
from npeaks_normalization.npeaks_visualize import NormVisualizer

# =============================================================================
# CONFIGURATION - MODIFY THESE SETTINGS
# =============================================================================

# Data paths
DATA_BASE = Path("/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed")
RESAMPLED_DIR = DATA_BASE / "2resampledNifti"

# Splits manifest file (defines train/val/test split)
SPLITS_MANIFEST = SCRIPT_DIR / "splits_manifest.csv"

# Body region filter
BODY_REGION = "AB"  # Abdomen patients

# Use only train patients for computing target intensities (from splits_manifest.csv)
# When True: targets are computed from train patients only, but ALL abdomen patients are normalized
USE_TRAIN_FOR_TARGETS = True

# -----------------------------------------------------------------------------
# N4 Bias Field Correction Settings
# -----------------------------------------------------------------------------
N4_PASSES = 2  # 0 = no correction, 1 = once, 2 = twice
N4_FITTING_LEVELS = 5
N4_ITERATIONS_PER_LEVEL = 75
N4_CONVERGENCE_THRESHOLD = 0.01

# -----------------------------------------------------------------------------
# N-Peaks Normalization Parameters
# -----------------------------------------------------------------------------
LOCAL_INTENSITY_CHANGE_RADIUS = 5.0  # mm
LIC_THRESHOLD_QUANTILE = 0.8
PEAK_PROMINENCE_FRACTION = 0.1

# Mask configuration
# NOTE: combined_fat will be created by combining torso_fat + subcutaneous_fat
MASK_CONFIG = {
    "combined_fat": {"target": None, "strategy": "right"},
    "liver": {"target": None, "strategy": "most"},
}

# Which masks to use
ACTIVE_MASKS = ["combined_fat", "liver"]

# -----------------------------------------------------------------------------
# Zero Anchor Setting
# -----------------------------------------------------------------------------
USE_ZERO_ANCHOR = True

# -----------------------------------------------------------------------------
# Center-Specific Normalization
# -----------------------------------------------------------------------------
CENTER_SPECIFIC = True  # Compute targets per center (A, B, C)
RECOMPUTE_TARGETS = True  # Set to True to recompute targets from all patients

FIXED_TARGETS = {
    "combined_fat": 800.0,
    "liver": 400.0,
}

CENTER_TARGET_INTENSITIES = {
    "A": {
        "combined_fat": 1246.34,
        "liver": 771.14,
    },
    "B": {
        "combined_fat": 249.39,
        "liver": 123.65,
    },
    "C": {
        "combined_fat": 1321.61,
        "liver": 246.05,
    },
}

# -----------------------------------------------------------------------------
# Visualization Settings
# -----------------------------------------------------------------------------
ENABLE_VISUALIZATION = True  # Save normalization visualization plots per patient

# Limit patients for testing (None = all)
MAX_PATIENTS = None

METHOD_CONFIGS = {
    "34npeaks": {
        "n4_passes": 0,
        "lic_threshold_quantile": 0.8,
        "center_specific": False,
    },
    "34normalized_n4_03LIC": {
        "n4_passes": 2,
        "lic_threshold_quantile": 0.3,
        "center_specific": False,
    },
    "34normalized_n4_08LIC": {
        "n4_passes": 2,
        "lic_threshold_quantile": 0.8,
        "center_specific": False,
    },
    "34normalized_n4_centerspecific_03LIC": {
        "n4_passes": 2,
        "lic_threshold_quantile": 0.3,
        "center_specific": True,
    },
    "34normalized_n4_centerspecific_08LIC": {
        "n4_passes": 2,
        "lic_threshold_quantile": 0.8,
        "center_specific": True,
    },
}


def method_folder_name(method: str) -> str:
    return f"3_{method}"

# =============================================================================
# FUNCTIONS
# =============================================================================

def get_train_patients_from_manifest(manifest_path, body_region=None):
    """
    Read train patient IDs from splits_manifest.csv.

    Args:
        manifest_path: Path to splits_manifest.csv
        body_region: Filter by body region (e.g., 'AB' for abdomen)

    Returns:
        List of patient folder names (e.g., ['AB_1ABA005', 'AB_1ABB118', ...])
    """
    import csv

    if not manifest_path.exists():
        print(f"Warning: Manifest file not found: {manifest_path}")
        return []

    train_patients = []
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['split'] == 'train':
                # Filter by body region if specified
                if body_region is None or row['body'] == body_region:
                    # Extract patient folder name from example_path
                    # e.g., '.../AB_1ABB118' -> 'AB_1ABB118'
                    patient_folder = Path(row['example_path']).name
                    train_patients.append(patient_folder)

    return train_patients


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



def apply_n4_correction(image_data, mask_data, affine, n_passes=1, 
                        fitting_levels=4, iterations_per_level=50, 
                        convergence_threshold=0.001):
    """Apply N4 bias field correction to MRI data."""
    sitk_image = sitk.GetImageFromArray(image_data.transpose(2, 1, 0).astype(np.float32))
    spacing = np.abs(np.diag(affine)[:3])
    sitk_image.SetSpacing(spacing.tolist())
    
    sitk_mask = sitk.GetImageFromArray(mask_data.transpose(2, 1, 0).astype(np.uint8))
    sitk_mask.CopyInformation(sitk_image)
    
    corrected = sitk_image
    
    for pass_num in range(n_passes):
        print(f"    N4 pass {pass_num + 1}/{n_passes}...")
        n4_filter = sitk.N4BiasFieldCorrectionImageFilter()
        n4_filter.SetMaximumNumberOfIterations([iterations_per_level] * fitting_levels)
        n4_filter.SetConvergenceThreshold(convergence_threshold)
        corrected = n4_filter.Execute(corrected, sitk_mask)
    
    return sitk.GetArrayFromImage(corrected).transpose(2, 1, 0)


def find_peaks_for_patient(patient_id, resampled_dir, active_masks, mask_config, n4_passes=0):
    """Find peak intensities for a single patient. Returns dict of {mask_name: peak_intensity}."""
    patient_dir = resampled_dir / patient_id

    # Load MR image
    mr_path = patient_dir / "MR" / f"{patient_id}_MR_cp256.nii.gz"
    if not mr_path.exists():
        return None

    mr_nii = nib.load(str(mr_path))
    mr_data = mr_nii.get_fdata()
    voxel_spacing = calculate_voxel_spacing_from_affine(mr_nii.affine)

    # Load body mask
    body_mask_path = patient_dir / "new_masks" / f"{patient_id}_mask_from_CT_treshold_cp256.nii.gz"
    if not body_mask_path.exists():
        return None
    body_mask = nib.load(str(body_mask_path)).get_fdata().astype(bool)

    # Apply N4 if configured (must match normalize_patient)
    if n4_passes > 0:
        mr_data = apply_n4_correction(
            mr_data, body_mask, mr_nii.affine,
            n_passes=n4_passes,
            fitting_levels=N4_FITTING_LEVELS,
            iterations_per_level=N4_ITERATIONS_PER_LEVEL,
            convergence_threshold=N4_CONVERGENCE_THRESHOLD
        )

    # Load masks and find peaks
    normalizer = NPeaksNormalizer()

    mask_arr_list = []
    strategies = []
    mask_names_used = []

    for mask_name in active_masks:
        # Special handling for combined_fat
        if mask_name == "combined_fat":
            torso_fat_path = patient_dir / "totalsegmentator_masks" / "torso_fat_cp256.nii.gz"
            subcut_fat_path = patient_dir / "totalsegmentator_masks" / "subcutaneous_fat_cp256.nii.gz"

            torso_fat = None
            subcut_fat = None

            if torso_fat_path.exists():
                torso_fat = nib.load(str(torso_fat_path)).get_fdata().astype(bool) & body_mask
            if subcut_fat_path.exists():
                subcut_fat = nib.load(str(subcut_fat_path)).get_fdata().astype(bool) & body_mask

            if torso_fat is not None and subcut_fat is not None:
                mask_data = torso_fat | subcut_fat
            elif torso_fat is not None:
                mask_data = torso_fat
            elif subcut_fat is not None:
                mask_data = subcut_fat
            else:
                continue
        else:
            mask_path = patient_dir / "totalsegmentator_masks" / f"{mask_name}_cp256.nii.gz"
            if not mask_path.exists():
                continue
            mask_data = nib.load(str(mask_path)).get_fdata().astype(bool) & body_mask

        if mask_data.sum() >= 1000:
            mask_arr_list.append(mask_data)
            strategies.append(mask_config[mask_name]["strategy"])
            mask_names_used.append(mask_name)

    if not mask_arr_list:
        return None

    try:
        peak_intensities = normalizer.find_peaks(
            intensity_arr=mr_data,
            mask_arr_list=mask_arr_list,
            voxel_spacing_arr=voxel_spacing,
            local_intensity_change_radius=LOCAL_INTENSITY_CHANGE_RADIUS,
            local_intensity_change_threshold_quantile_list=[LIC_THRESHOLD_QUANTILE] * len(mask_arr_list),
            peak_detection_prominence_fraction_list=[PEAK_PROMINENCE_FRACTION] * len(mask_arr_list),
            peak_detection_peak_selection_strategy_list=strategies,
        )
        return dict(zip(mask_names_used, peak_intensities))
    except Exception:
        return None


def normalize_patient(patient_id, resampled_dir, output_dir, target_intensities,
                      active_masks, mask_config, n4_passes, use_zero_anchor,
                      enable_visualization=False):
    """Normalize a single patient and save the result."""
    patient_dir = resampled_dir / patient_id
    
    # Load MR
    mr_path = patient_dir / "MR" / f"{patient_id}_MR_cp256.nii.gz"
    if not mr_path.exists():
        return False, "MR file not found"
    
    mr_nii = nib.load(str(mr_path))
    mr_data_raw = mr_nii.get_fdata()
    voxel_spacing = calculate_voxel_spacing_from_affine(mr_nii.affine)
    
    # Load body mask
    body_mask_path = patient_dir / "new_masks" / f"{patient_id}_mask_from_CT_treshold_cp256.nii.gz"
    if not body_mask_path.exists():
        return False, "Body mask not found"
    body_mask = nib.load(str(body_mask_path)).get_fdata().astype(bool)
    
    # Apply N4 if configured
    if n4_passes > 0:
        mr_data = apply_n4_correction(
            mr_data_raw, body_mask, mr_nii.affine,
            n_passes=n4_passes,
            fitting_levels=N4_FITTING_LEVELS,
            iterations_per_level=N4_ITERATIONS_PER_LEVEL,
            convergence_threshold=N4_CONVERGENCE_THRESHOLD
        )
    else:
        mr_data = mr_data_raw
    
    # Load masks
    mask_arr_list = []
    strategies = []
    mask_targets = []

    for mask_name in active_masks:
        # Special handling for combined_fat
        if mask_name == "combined_fat":
            torso_fat_path = patient_dir / "totalsegmentator_masks" / "torso_fat_cp256.nii.gz"
            subcut_fat_path = patient_dir / "totalsegmentator_masks" / "subcutaneous_fat_cp256.nii.gz"

            # Load both fat masks if they exist
            torso_fat = None
            subcut_fat = None

            if torso_fat_path.exists():
                torso_fat = nib.load(str(torso_fat_path)).get_fdata().astype(bool) & body_mask
            if subcut_fat_path.exists():
                subcut_fat = nib.load(str(subcut_fat_path)).get_fdata().astype(bool) & body_mask

            # Combine the masks (logical OR)
            if torso_fat is not None and subcut_fat is not None:
                mask_data = torso_fat | subcut_fat
            elif torso_fat is not None:
                mask_data = torso_fat
            elif subcut_fat is not None:
                mask_data = subcut_fat
            else:
                continue  # Skip if neither mask exists
        else:
            # Regular mask loading
            mask_path = patient_dir / "totalsegmentator_masks" / f"{mask_name}_cp256.nii.gz"
            if not mask_path.exists():
                continue
            mask_data = nib.load(str(mask_path)).get_fdata().astype(bool) & body_mask

        if mask_data.sum() >= 1000:
            mask_arr_list.append(mask_data)
            strategies.append(mask_config[mask_name]["strategy"])
            mask_targets.append(target_intensities[mask_name])
    
    if not mask_arr_list:
        return False, "No valid masks"

    # Setup output directory for this patient
    output_patient_dir = output_dir / patient_id / "MR"
    output_patient_dir.mkdir(parents=True, exist_ok=True)

    # Create visualizer if enabled
    visualizer = None
    if enable_visualization:
        visualizer = NormVisualizer(
            plot_folder=str(output_patient_dir),
            is_compensate_nifti_coords=True
        )

    # Find peaks
    normalizer = NPeaksNormalizer(visualizer=visualizer)
    peak_intensities = normalizer.find_peaks(
        intensity_arr=mr_data,
        mask_arr_list=mask_arr_list,
        voxel_spacing_arr=voxel_spacing,
        local_intensity_change_radius=LOCAL_INTENSITY_CHANGE_RADIUS,
        local_intensity_change_threshold_quantile_list=[LIC_THRESHOLD_QUANTILE] * len(mask_arr_list),
        peak_detection_prominence_fraction_list=[PEAK_PROMINENCE_FRACTION] * len(mask_arr_list),
        peak_detection_peak_selection_strategy_list=strategies,
        image_name=patient_id,
    )
    
    # Add zero anchor
    if use_zero_anchor:
        used_peak_list = [0.0] + list(peak_intensities)
        used_target_list = [0.0] + list(mask_targets)
    else:
        used_peak_list = list(peak_intensities)
        used_target_list = list(mask_targets)
    
    # Transform
    normalized_data = normalizer.transform_intensity_scale(
        intensity_arr=mr_data,
        determined_peak_intensity_list=used_peak_list,
        target_peak_intensity_list=used_target_list
    )

    # Clip negative values (can occur from piecewise linear extrapolation)
    normalized_data = np.clip(normalized_data, 0, None)

    # Apply body mask
    normalized_data = normalized_data * body_mask

    # Rescale to [0, 1] using a fixed maximum to preserve inter-patient alignment
    # Use 1.5x the highest target intensity — consistent for all patients sharing the same targets
    fixed_max = 1.5 * max(target_intensities.values())
    normalized_data = np.clip(normalized_data, 0, fixed_max) / fixed_max

    # Save MR
    output_path = output_patient_dir / f"{patient_id}_MR_cp256.nii.gz"
    nib.save(
        nib.Nifti1Image(normalized_data, mr_nii.affine, mr_nii.header),
        str(output_path)
    )

    # Process and save CT
    ct_path = patient_dir / "CT_reg" / f"{patient_id}_CT_reg_cp256.nii.gz"
    if ct_path.exists():
        ct_nii = nib.load(str(ct_path))
        ct_data = ct_nii.get_fdata()
        ct_normalized = normalize_ct(ct_data)

        # Save CT to output folder
        output_ct_dir = output_dir / patient_id / "CT_reg"
        output_ct_dir.mkdir(parents=True, exist_ok=True)
        ct_output_path = output_ct_dir / f"{patient_id}_CT_reg_cp256.nii.gz"
        nib.save(
            nib.Nifti1Image(ct_normalized, ct_nii.affine, ct_nii.header),
            str(ct_output_path)
        )

    # Copy mask folders (new_masks, totalsegmentator_masks) from source
    for mask_folder_name in ["new_masks", "totalsegmentator_masks"]:
        source_mask_folder = patient_dir / mask_folder_name
        target_mask_folder = output_dir / patient_id / mask_folder_name
        if source_mask_folder.exists() and not target_mask_folder.exists():
            shutil.copytree(source_mask_folder, target_mask_folder)

    return True, f"MR: [0, 1], CT: [-1024,1200]→[0,1]"


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run N-peaks MRI normalization.")
    parser.add_argument(
        "--method",
        choices=sorted(METHOD_CONFIGS),
        default="34normalized_n4_centerspecific_08LIC",
        help="N-peaks configuration to run.",
    )
    parser.add_argument(
        "--base-root",
        type=Path,
        default=DATA_BASE,
        help="Pipeline base root.",
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=None,
        help="Resampled input root (defaults to <base-root>/2resampledNifti).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Split manifest CSV (defaults to <base-root>/splits_manifest.csv).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Normalized output root (defaults to <base-root>/<method>/3normalized).",
    )
    parser.add_argument(
        "--patient-ids",
        nargs="+",
        default=None,
        help="Optional patient IDs to normalize.",
    )
    parser.add_argument(
        "--disable-visualization",
        action="store_true",
        help="Skip saving normalization plots.",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Optional limit for debugging.",
    )
    return parser.parse_args()


def main():
    global DATA_BASE, RESAMPLED_DIR, SPLITS_MANIFEST
    global N4_PASSES, LIC_THRESHOLD_QUANTILE, CENTER_SPECIFIC
    global ENABLE_VISUALIZATION, MAX_PATIENTS

    args = parse_args()
    config = METHOD_CONFIGS[args.method]
    DATA_BASE = args.base_root
    RESAMPLED_DIR = args.src_root if args.src_root is not None else DATA_BASE / "2resampledNifti"
    SPLITS_MANIFEST = args.manifest if args.manifest is not None else DATA_BASE / "splits_manifest.csv"
    N4_PASSES = config["n4_passes"]
    LIC_THRESHOLD_QUANTILE = config["lic_threshold_quantile"]
    CENTER_SPECIFIC = config["center_specific"]
    ENABLE_VISUALIZATION = not args.disable_visualization
    MAX_PATIENTS = args.max_patients

    print("=" * 70)
    print("N-PEAKS BATCH NORMALIZATION")
    print("=" * 70)

    OUTPUT_DIR = args.out_root if args.out_root is not None else DATA_BASE / method_folder_name(args.method) / "3normalized"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print(f"\nConfiguration:")
    print(f"  N4 passes: {N4_PASSES}")
    print(f"  Zero anchor: {USE_ZERO_ANCHOR}")
    print(f"  Center-specific: {CENTER_SPECIFIC}")
    print(f"  Active masks: {ACTIVE_MASKS}")
    print(f"  Visualization: {ENABLE_VISUALIZATION}")
    print(f"  Use train for targets: {USE_TRAIN_FOR_TARGETS}")
    print(f"  Body region: {BODY_REGION}")
    print(f"  Output directory: {OUTPUT_DIR}")

    # Get ALL abdomen patients for normalization
    all_patients = sorted(
        d.name for d in RESAMPLED_DIR.iterdir()
        if d.is_dir() and d.name.startswith(f"{BODY_REGION}_")
    )
    if args.patient_ids:
        selected = set(args.patient_ids)
        all_patients = [patient_id for patient_id in all_patients if patient_id in selected]
    print(f"\nFound {len(all_patients)} {BODY_REGION} patients to normalize")

    # Get train patients for computing targets (if needed)
    if USE_TRAIN_FOR_TARGETS:
        manifest_patients = get_train_patients_from_manifest(SPLITS_MANIFEST, BODY_REGION)
        train_patients = sorted([
            p for p in manifest_patients
            if (RESAMPLED_DIR / p).is_dir()
        ])
        if args.patient_ids:
            selected = set(args.patient_ids)
            train_patients = [patient_id for patient_id in train_patients if patient_id in selected]
        print(f"Using {len(train_patients)} train patients for computing targets")
    else:
        train_patients = all_patients
        print(f"Using all patients for computing targets")

    if MAX_PATIENTS:
        all_patients = all_patients[:MAX_PATIENTS]

    if not all_patients:
        raise SystemExit(f"No patients found under {RESAMPLED_DIR} for {args.method}")

    # Center-specific normalization
    if CENTER_SPECIFIC:
        from collections import defaultdict

        # Group ALL patients by center (for normalization)
        center_patients_all = defaultdict(list)
        for patient_id in all_patients:
            center = patient_id.split('_')[1][1:4][-1]  # Gets A, B, or C
            center_patients_all[center].append(patient_id)

        # Group TRAIN patients by center (for computing targets)
        center_patients_train = defaultdict(list)
        for patient_id in train_patients:
            center = patient_id.split('_')[1][1:4][-1]  # Gets A, B, or C
            center_patients_train[center].append(patient_id)

        print(f"\nPatients per center (all / train):")
        for center in sorted(center_patients_all.keys()):
            n_all = len(center_patients_all[center])
            n_train = len(center_patients_train[center])
            print(f"  Center {center}: {n_all} total, {n_train} train")

        # Use pre-computed targets or recompute
        if RECOMPUTE_TARGETS:
            # Collect peaks per center (using TRAIN patients only)
            print("\n" + "=" * 70)
            print("COLLECTING PEAKS PER CENTER (from train patients)")
            print("=" * 70)

            center_peaks = {center: {mask: [] for mask in ACTIVE_MASKS}
                           for center in center_patients_all.keys()}

            for center in sorted(center_patients_train.keys()):
                print(f"\nProcessing Center {center} ({len(center_patients_train[center])} train patients)...")
                for patient_id in tqdm(center_patients_train[center], desc=f"Center {center}"):
                    peaks = find_peaks_for_patient(patient_id, RESAMPLED_DIR, ACTIVE_MASKS, MASK_CONFIG, n4_passes=N4_PASSES)
                    if peaks:
                        for mask_name, peak_val in peaks.items():
                            if mask_name in center_peaks[center]:
                                center_peaks[center][mask_name].append(peak_val)

            # Calculate median targets per center
            print("\n" + "=" * 70)
            print("CENTER-SPECIFIC PEAK STATISTICS")
            print("=" * 70)

            targets = {}
            for center in sorted(center_patients_all.keys()):
                print(f"\n--- Center {center} ---")
                targets[center] = {}

                for mask_name in ACTIVE_MASKS:
                    peaks = center_peaks[center][mask_name]
                    if peaks:
                        median_peak = np.median(peaks)
                        mean_peak = np.mean(peaks)
                        std_peak = np.std(peaks)
                        targets[center][mask_name] = median_peak
                        print(f"  {mask_name}: n={len(peaks)}, median={median_peak:.2f}, mean={mean_peak:.2f}, std={std_peak:.2f}")
                    else:
                        print(f"  {mask_name}: No valid peaks found!")
                        targets[center][mask_name] = FIXED_TARGETS.get(mask_name, 0.5)
        else:
            # Use pre-computed targets
            print("\n" + "=" * 70)
            print("USING PRE-COMPUTED CENTER-SPECIFIC TARGETS")
            print("=" * 70)
            targets = CENTER_TARGET_INTENSITIES
            for center in sorted(targets.keys()):
                print(f"\n--- Center {center} ---")
                for mask_name in ACTIVE_MASKS:
                    print(f"  {mask_name}: {targets[center][mask_name]:.2f}")

        # Process each patient with center-specific targets
        print("\n" + "=" * 70)
        print("NORMALIZING PATIENTS")
        print("=" * 70)

        success_count = 0
        failed_patients = []

        for patient_id in tqdm(all_patients, desc="Normalizing"):
            try:
                center = patient_id.split('_')[1][1:4][-1]
                patient_targets = targets[center]

                success, msg = normalize_patient(
                    patient_id=patient_id,
                    resampled_dir=RESAMPLED_DIR,
                    output_dir=OUTPUT_DIR,
                    target_intensities=patient_targets,
                    active_masks=ACTIVE_MASKS,
                    mask_config=MASK_CONFIG,
                    n4_passes=N4_PASSES,
                    use_zero_anchor=USE_ZERO_ANCHOR,
                    enable_visualization=ENABLE_VISUALIZATION,
                )
                if success:
                    success_count += 1
                else:
                    failed_patients.append((patient_id, msg))
            except Exception as e:
                failed_patients.append((patient_id, str(e)))

        # Save center-specific log
        log_path = OUTPUT_DIR / "processing_log.txt"
        with open(log_path, 'w') as f:
            f.write(f"N-Peaks Batch Processing Log (Center-Specific)\n")
            f.write(f"{'=' * 50}\n")
            f.write(f"N4 passes: {N4_PASSES}\n")
            f.write(f"N4 iterations per level: {N4_ITERATIONS_PER_LEVEL}\n")
            f.write(f"LIC Threshold: {LIC_THRESHOLD_QUANTILE}\n")
            f.write(f"Zero anchor: {USE_ZERO_ANCHOR}\n")
            f.write(f"Center-specific: {CENTER_SPECIFIC}\n")
            f.write(f"Recompute targets: {RECOMPUTE_TARGETS}\n")
            f.write(f"Active masks: {ACTIVE_MASKS}\n")
            f.write(f"Visualization: {ENABLE_VISUALIZATION}\n")
            f.write(f"Use train for targets: {USE_TRAIN_FOR_TARGETS}\n")
            f.write(f"Body region: {BODY_REGION}\n")
            f.write(f"\nCenter-specific targets:\n")
            for center in sorted(targets.keys()):
                f.write(f"  Center {center}: {targets[center]}\n")
            f.write(f"\nSuccessful: {success_count}/{len(all_patients)}\n")
            if failed_patients:
                f.write(f"\nFailed:\n")
                for pid, err in failed_patients:
                    f.write(f"  {pid}: {err}\n")

    else:
        # Global normalization (original behavior)
        print(f"  Target intensities: {FIXED_TARGETS}")
        print("=" * 70)

        success_count = 0
        failed_patients = []

        for patient_id in tqdm(all_patients, desc="Normalizing"):
            try:
                success, msg = normalize_patient(
                    patient_id=patient_id,
                    resampled_dir=RESAMPLED_DIR,
                    output_dir=OUTPUT_DIR,
                    target_intensities=FIXED_TARGETS,
                    active_masks=ACTIVE_MASKS,
                    mask_config=MASK_CONFIG,
                    n4_passes=N4_PASSES,
                    use_zero_anchor=USE_ZERO_ANCHOR,
                    enable_visualization=ENABLE_VISUALIZATION,
                )
                if success:
                    success_count += 1
                else:
                    failed_patients.append((patient_id, msg))
            except Exception as e:
                failed_patients.append((patient_id, str(e)))

        # Save log
        log_path = OUTPUT_DIR / "processing_log.txt"
        with open(log_path, 'w') as f:
            f.write(f"N-Peaks Batch Processing Log\n")
            f.write(f"{'=' * 50}\n")
            f.write(f"N4 passes: {N4_PASSES}\n")
            f.write(f"N4 iterations per level: {N4_ITERATIONS_PER_LEVEL}\n")
            f.write(f"LIC Threshold: {LIC_THRESHOLD_QUANTILE}\n")
            f.write(f"Zero anchor: {USE_ZERO_ANCHOR}\n")
            f.write(f"Active masks: {ACTIVE_MASKS}\n")
            f.write(f"Visualization: {ENABLE_VISUALIZATION}\n")
            f.write(f"Use train for targets: {USE_TRAIN_FOR_TARGETS}\n")
            f.write(f"Body region: {BODY_REGION}\n")
            f.write(f"Targets: {FIXED_TARGETS}\n")
            f.write(f"\nSuccessful: {success_count}/{len(all_patients)}\n")
            if failed_patients:
                f.write(f"\nFailed:\n")
                for pid, err in failed_patients:
                    f.write(f"  {pid}: {err}\n")

    # Summary
    print("\n" + "=" * 70)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Successful: {success_count}/{len(all_patients)}")
    print(f"Output: {OUTPUT_DIR}")

    if failed_patients:
        print(f"\nFailed ({len(failed_patients)}):")
        for pid, err in failed_patients[:20]:
            print(f"  {pid}: {err}")
        if len(failed_patients) > 20:
            print(f"  ... and {len(failed_patients) - 20} more")

    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
