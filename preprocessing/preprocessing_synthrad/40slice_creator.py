#!/usr/bin/env python

import os
import shutil
from glob import glob
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import nibabel as nib

# ===================== CONFIG =====================

# Root with cropped CT & per-patient folders:
#   CT_ROOT/<PATIENT_ID>/CT_reg/*.nii.gz
CT_ROOT = "/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/2resampledNifti"
MR_ROOT = "/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/2resampledNifti"

# Root with Nyul-normalized MR:
#   MR_ROOT contains files starting with <PATIENT_ID>, e.g. AB_1ABC100_MR_...nii.gz
#MR_ROOT = "/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/4nyulNormalizedMRNifti"

# Where to write slice datasets
OUT_ROOT = "/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/5slicesOutputForModelsNonNormalized"

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

    # Fallback: any file with patient_id prefix
    pattern = os.path.join(mr_root, "**", f"{patient_id}*.nii*")
    files = sorted(glob(pattern, recursive=True))
    if files:
        return files[0]

    return None


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

    ct = ct_img.get_fdata()
    mr = mr_img.get_fdata()

    if ct.shape != mr.shape:
        print(f"!WARNING! Shape mismatch for {patient_id}: CT{ct.shape} vs MR{mr.shape}, skipping")
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
        else:
            # pseudo3d: 3 slices as channels
            if i == 0 or i == nz - 1:
                continue
            mr_slice = mr[..., i-1:i+2]
            ct_slice = ct[..., i-1:i+2]

        slice_name = f"{patient_id}-{i}.{SLICE_EXT}"

    # paired pix2pix-style
        save_slice(mr_slice, mr_img.affine, os.path.join(pix_A_dir, slice_name), is_mr=True)
        save_slice(ct_slice, ct_img.affine, os.path.join(pix_B_dir, slice_name), is_mr=False)

    # unpaired model dirs
        save_slice(mr_slice, mr_img.affine, os.path.join(model_A_dir, slice_name), is_mr=True)
        save_slice(ct_slice, ct_img.affine, os.path.join(model_B_dir, slice_name), is_mr=False)


def main():
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
        
        patients.append((entry, ct_path, mr_path))

    if not patients:
        raise RuntimeError("No paired CT+MR patients found. Check CT_ROOT and MR_ROOT.")

    print(f"Found {len(patients)} paired patients")

    # 2) prepare output structure (no split here)
    path_model = os.path.join(OUT_ROOT, f"model_{NN_INPUT_MODE}")
    path_pix = os.path.join(OUT_ROOT, f"pix2pix_{NN_INPUT_MODE}")

    safe_rmtree_and_make(path_model)
    safe_rmtree_and_make(path_pix)
    make_output_dirs(path_model, path_pix)

    # 3) slice generation (all patients)
    for pid, ct_p, mr_p in tqdm(patients):
        create_slices_for_pair(pid, ct_p, mr_p, path_model, path_pix)

    print("Finished slice creation.")


if __name__ == "__main__":
    main()
