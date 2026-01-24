#!/usr/bin/env python

"""
Usage:
    python 40slice_creator.py [normalization_method]

Examples:
    python 40slice_creator.py 31baseline
    python 40slice_creator.py 32p99
    python 40slice_creator.py 33nyul
    python 40slice_creator.py 34npeaks

If no argument is provided, uses default: 32p99 (per-file p99)

This will:
    - Read from: {method}/3normalized/
    - Write to:  {method}/5slices/
"""

import os
import sys
import shutil
from glob import glob
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import nibabel as nib

# ===================== CONFIG =====================

# NORMALIZATION_METHOD: Choose which preprocessing pipeline to use
# Options: "31baseline", "32p99", "33nyul", "34npeaks"
# Can be overridden via command line argument: python 40slice_creator.py 32p99
NORMALIZATION_METHOD = "32p99"  # Default: per-file p99

# Base directory
BASE_ROOT = "/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2"

# Will be set dynamically based on NORMALIZATION_METHOD
CT_ROOT = None
MR_ROOT = None
SLICE_ROOT = None
OUT_ROOT = None
PATH_OUTPUT_MASK = None

# Use 2D slices
NN_INPUT_MODE = "2d"

# Slice file format: "nii.gz", "nii", or "png"
#SLICE_EXT = "nii.gz"
SLICE_EXT = "nii" #TODO: check if we can create png for pix2pix and nii otherwise

# Skip first/last slice to avoid edge artefacts
SKIP_FIRST_LAST = True

# Clamp Nyul MR values into [0,1]
CLAMP_MR = False

# ==================================================


def safe_rmtree_and_make(path: str):
    if os.path.exists(path):
        print(f"!IMPORTANT! path={path} exists. Removing it with all files inside")
        try:
            shutil.rmtree(path)
        except OSError as e:
            print("Error: %s : %s" % (path, e.strerror))
    os.makedirs(path)


def make_output_dirs(base_model: str, base_pix: str):
        """
        Create unified slice output directories (no train/test split here).
        Downstream splitting is handled by 50_dataset_split.py.
    
        Layouts created:
            - model (unpaired-style):  <base_model>/full/{A,B}
            - pix2pix (paired-style):  <base_pix>/full/{A,B}
        """
        # model: unpaired-style
        os.makedirs(os.path.join(base_model, "full", "A"))
        os.makedirs(os.path.join(base_model, "full", "B"))

        # pix2pix: paired-style
        os.makedirs(os.path.join(base_pix, "full", "A"))
        os.makedirs(os.path.join(base_pix, "full", "B"))


def find_ct_nifti_for_patient(patient_dir: str):
    ct_dir = os.path.join(patient_dir, "CT_reg")
    if not os.path.isdir(ct_dir):
        return None
    for pattern in ("*.nii.gz", "*.nii"):
        files = sorted(glob(os.path.join(ct_dir, pattern)))
        if files:
            return files[0]
    return None


def find_mr_nifti_for_patient(mr_root: str, patient_id: str):
    """
    Look for any MR file starting with patient_id under mr_root.
    Works for .nii or .nii.gz; flat or nested.
    """
    # Prefer names that contain 'MR'
    pattern = os.path.join(mr_root, "**", f"{patient_id}*MR*.nii*")
    files = sorted(glob(pattern, recursive=True))
    if files:
        return files[0]

    return None

def find_slice_nifti_for_patient(slice_root: str, patient_id: str):
    pattern = os.path.join(slice_root, "**", f"{patient_id}*mask*.nii*")
    files = sorted(glob(pattern, recursive=True))
    if files:
        return files[0]


def parse_region(patient_id: str) -> str:
    # e.g. "AB_1ABC100" -> "AB"
    return patient_id.split("_")[0] if "_" in patient_id else "UNK"


def parse_hospital(patient_id: str) -> str:
    """
    Hospital derived from the 3rd letter of the code part, e.g.:
      AB_1ABC100 -> letters in '1ABC100' are A,B,C -> hospital = 'C'
    If pattern breaks, returns 'UNK'.
    """
    if "_" not in patient_id:
        return "UNK"
    code = patient_id.split("_")[1]
    letters = [c for c in code if c.isalpha()]
    if len(letters) >= 3:
        return letters[2]
    return "UNK"


def save_slice(im_data: np.ndarray, affine, path_save: str, is_mr: bool):
    if os.path.exists(path_save):
        os.remove(path_save)

    data = im_data.copy()

    if is_mr and CLAMP_MR:
        data = np.clip(data, 0.0, 1.0)

    ext = SLICE_EXT.lower()
    if ext == "png":
        data = data.astype(np.float32)
        data = (data * 255.0).round().astype(np.uint8)
        if data.ndim == 3 and data.shape[-1] == 1:
            data = data[..., 0]
        import imageio
        imageio.imwrite(path_save, data)
    else:
        img_slice = nib.Nifti1Image(data, affine)
        nib.save(img_slice, path_save)


def create_slices_for_pair(
    patient_id: str,
    ct_path: str,
    mr_path: str,
    base_model: str,
    base_pix: str,
    mask_path: str
):
    """
    Creates 2D slices for one paired CT–MR volume:
      - MR -> domain A
      - CT -> domain B
    Note: This function does not perform any data split; it writes all slices
    into unified output folders. Use 50_dataset_split.py afterwards to create
    train/val/test splits.
    """
    # model (unpaired-style) dirs
    model_A_dir = os.path.join(base_model, "full", "A")
    model_B_dir = os.path.join(base_model, "full", "B")

    # pix2pix (paired) dirs
    pix_A_dir = os.path.join(base_pix, "full", "A")
    pix_B_dir = os.path.join(base_pix, "full", "B")

    os.makedirs(model_A_dir, exist_ok=True)
    os.makedirs(model_B_dir, exist_ok=True)
    os.makedirs(pix_A_dir, exist_ok=True)
    os.makedirs(pix_B_dir, exist_ok=True)

    # load volumes
    ct_img = nib.load(ct_path)
    mr_img = nib.load(mr_path)
    mask_img = nib.load(mask_path)

    ct = ct_img.get_fdata()
    mr = mr_img.get_fdata()
    mask = mask_img.get_fdata()

    if ct.shape != mr.shape != mask.shape:
        print(f"!WARNING! Shape mismatch for {patient_id}: CT{ct.shape} vs MR{mr.shape} vs Mask{mask.shape}, skipping")
        return

    if ct.ndim != 3:
        print(f"!WARNING! Non-3D volume for {patient_id}, skipping")
        return

    nz = ct.shape[2]

    if SKIP_FIRST_LAST and nz > 2:
        z_range = range(1, nz - 1)
    else:
        z_range = range(0, nz)

    for i in z_range:
        if NN_INPUT_MODE == "2d":
            mr_slice = mr[..., i:i+1]  # (x, y, 1)
            ct_slice = ct[..., i:i+1]
            mask_slice = mask[..., i:i+1]
        else:
            # pseudo3d: 3 slices as channels
            if i == 0 or i == nz - 1:
                continue
            mr_slice = mr[..., i-1:i+2]
            ct_slice = ct[..., i-1:i+2]
            mask_slice = mask[..., i-1:i+2]

        slice_name = f"{patient_id}-{i}.{SLICE_EXT}"

    # paired pix2pix-style
        save_slice(mr_slice, mr_img.affine, os.path.join(pix_A_dir, slice_name), is_mr=True)
        save_slice(ct_slice, ct_img.affine, os.path.join(pix_B_dir, slice_name), is_mr=False)
        save_slice(mask_slice, mask_img.affine, os.path.join(PATH_OUTPUT_MASK, slice_name), is_mr=False)

    # unpaired model dirs
        save_slice(mr_slice, mr_img.affine, os.path.join(model_A_dir, slice_name), is_mr=True)
        save_slice(ct_slice, ct_img.affine, os.path.join(model_B_dir, slice_name), is_mr=False)
        save_slice(mask_slice, mask_img.affine, os.path.join(PATH_OUTPUT_MASK, slice_name), is_mr=False)


def configure_paths(method: str):
    """Configure input/output paths based on normalization method."""
    global CT_ROOT, MR_ROOT, OUT_ROOT, SLICE_ROOT
    
    valid_methods = ["31baseline", "32p99", "33nyul", "34npeaks"]
    
    if method not in valid_methods:
        raise ValueError(
            f"Invalid normalization method: '{method}'\n"
            f"Valid options: {valid_methods}"
        )
    
    # Input: normalized data from 31-34 scripts
    CT_ROOT = os.path.join(BASE_ROOT, method, "3normalized")
    MR_ROOT = os.path.join(BASE_ROOT, method, "3normalized")
    SLICE_ROOT = os.path.join(BASE_ROOT, method, "3normalized")
    
    # Output: slices with matching suffix
    OUT_ROOT = os.path.join(BASE_ROOT, method, "5slices")
    
    # Verify input directories exist
    if not os.path.exists(CT_ROOT):
        raise FileNotFoundError(
            f"Input directory not found: {CT_ROOT}\n"
            f"Please run the corresponding normalization script first (e.g., {method}_standardization.py)"
        )
    
    print(f"=" * 60)
    print(f"Normalization method: {method}")
    print(f"CT_ROOT  = {CT_ROOT}")
    print(f"MR_ROOT  = {MR_ROOT}")
    print(f"OUT_ROOT = {OUT_ROOT}")
    print(f"SLICE_ROOT = {SLICE_ROOT}")
    print(f"=" * 60)


def main():
    global NORMALIZATION_METHOD
    
    # Parse command line argument if provided
    if len(sys.argv) > 1:
        NORMALIZATION_METHOD = sys.argv[1]
    
    # Configure paths based on normalization method
    configure_paths(NORMALIZATION_METHOD)
    
    print(f"Input mode: {NN_INPUT_MODE}")
    print(f"Skip first/last slice: {SKIP_FIRST_LAST}")
    print(f"Slice extension: {SLICE_EXT}")
    print()

    # 1) discover paired patients
    patients = []
    for entry in tqdm(sorted(os.listdir(CT_ROOT))):
        patient_dir = os.path.join(CT_ROOT, entry)
        if not os.path.isdir(patient_dir):
            continue

        ct_path = find_ct_nifti_for_patient(patient_dir)
        if ct_path is None:
            continue

        mr_path = find_mr_nifti_for_patient(MR_ROOT, entry)
        if mr_path is None:
            print(f"!WARNING! No MR found for patient {entry}, skipping")
            continue

        mask_path = find_slice_nifti_for_patient(SLICE_ROOT, entry)
        if mask_path is None:
            print(f"!WARNING! No mask slice found for patient {entry}, skipping")
            continue

        
        patients.append((entry, ct_path, mr_path, mask_path))

    if not patients:
        raise RuntimeError("No paired CT+MR patients found. Check CT_ROOT and MR_ROOT.")

    print(f"Found {len(patients)} paired patients")

    # 2) prepare output structure (no split here)
    path_model = os.path.join(OUT_ROOT, f"model_{NN_INPUT_MODE}")
    path_pix = os.path.join(OUT_ROOT, f"pix2pix_{NN_INPUT_MODE}")

    safe_rmtree_and_make(path_model)
    safe_rmtree_and_make(path_pix)
    make_output_dirs(path_model, path_pix)

    global PATH_OUTPUT_MASK

    PATH_OUTPUT_MASK = os.path.join(OUT_ROOT, "masks")
    safe_rmtree_and_make(PATH_OUTPUT_MASK)

    # 3) slice generation (all patients)
    for pid, ct_p, mr_p, mask_p in tqdm(patients):
        create_slices_for_pair(pid, ct_p, mr_p, path_model, path_pix, mask_p)

    print("Finished slice creation.")


if __name__ == "__main__":
    main()
