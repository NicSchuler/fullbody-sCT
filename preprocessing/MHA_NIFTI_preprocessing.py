from pathlib import Path
import SimpleITK as sitk
import pandas as pd
import numpy as np
import shutil

# --- INPUTS ---
src_root = Path("/local/scratch/datasets/FullbodySCT/SynthRAD2025/task1/initDataTrainTask1")
base_path = Path("/local/scratch/datasets/FullbodySCT/SynthRAD2025/task1")  # must match resample.py
nifti_root = base_path / "nifti"
excel_dir = base_path / "excel"
excel_path = excel_dir / "data_CT_MR_TEMP_second_paper.xlsx"

rows = []  # rows for the Excel

AIR_HU_THRESH = -900
VAL_AIR = -1024
TISSUE_MIN, TISSUE_MAX = -30, 100
VAL_TISSUE = 7


def read_itk(p: Path) -> sitk.Image:
    return sitk.ReadImage(str(p))

def write_itk(img: sitk.Image, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(p))

def to_np(img: sitk.Image) -> np.ndarray:
    # (Z, Y, X)
    return sitk.GetArrayFromImage(img)

def to_img(arr: np.ndarray, ref: sitk.Image, pixel_type=None) -> sitk.Image:
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(ref)
    if pixel_type is not None:
        out = sitk.Cast(out, pixel_type)
    return out

def read_spacing_from_nifti(nifti_path: Path):
    img = sitk.ReadImage(str(nifti_path))
    sx, sy, sz = img.GetSpacing()  # (x, y, z)
    return sx, sy, sz

def convert_mha_to_nii(mha_path: Path, nii_path: Path):
    img = sitk.ReadImage(str(mha_path))
    sitk.WriteImage(img, str(nii_path))

def _find_one(patterns, root: Path):
    """Return first match for any pattern in patterns, or None."""
    for pat in patterns:
        m = next(root.glob(pat), None)
        if m is not None:
            return m
    return None

def main():
    # RECURSIVELY collect all unique parent folders that contain at least one .mha
    mha_dirs = sorted({p.parent for p in src_root.rglob("*.mha")})
    n_cases = len(mha_dirs)
    print(f"Found {n_cases} folders containing .mha files under {src_root}\n")
    
    for d in (excel_dir, nifti_root):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)


    processed = 0

    for i, case_dir in enumerate(mha_dirs, start=1):
        if not case_dir.is_dir():
            continue

        # Use leaf folder name as case_id (keep as in your original code)
        case_id = case_dir.name
        print(f"[{i}/{n_cases}] Processing {case_id}  ({case_dir})")

        out_dir = nifti_root / case_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # Source files (be a bit flexible with names)
        ct_mha = _find_one(["*ct*.mha", "*CT*.mha"], case_dir)
        mr_mha = _find_one(["*mr*.mha", "*mri*.mha", "*MR*.mha", "*MRI*.mha"], case_dir)
        body_mask_mha = _find_one(["*mask*.mha", "*body*mask*.mha"], case_dir)

        # Required target filenames for resample.py
        ct_out = out_dir / f"{case_id}_3D_CT_air_overwrite.nii"
        mr_out = out_dir / f"{case_id}_3D_body.nii"
        body_mask_out = out_dir / "3D_mask_body.nii"

        print(f"[{i}/{n_cases}] {case_id}")

        ct_itk = read_itk(ct_mha) if ct_mha and ct_mha.exists() else None
        mr_itk = read_itk(mr_mha) if mr_mha and mr_mha.exists() else None
        mask_itk = read_itk(body_mask_mha) if body_mask_mha and body_mask_mha.exists() else None

        if mask_itk is None:
            print(f"Missing body mask for {case_id} — skipping case (resampler requires 3D_mask_body.nii).")
            continue

        mask_np = (to_np(mask_itk) > 0).astype(np.uint8)
        # Write 3D_mask_body.nii with CT geometry if CT exists, else MR
        ref_for_mask = ct_itk if ct_itk is not None else (mr_itk if mr_itk is not None else mask_itk)
        write_itk(to_img(mask_np, ref_for_mask, pixel_type=sitk.sitkUInt8), body_mask_out)
        have_mr_row = False
        have_ct_row = False

        if mr_itk is not None:
            mr_np = to_np(mr_itk).astype(np.float32)
            mr_body_np = mr_np * mask_np
            write_itk(to_img(mr_body_np, mr_itk), mr_out)
            have_mr_row = True
            print(f"MR masked → {mr_out.name}")
        else:
            print(f"Missing MR for {case_id}")


        if ct_itk is not None:
            ct_np = to_np(ct_itk).astype(np.float32)
            body_bool = mask_np.astype(bool)

            air_np = ((ct_np <= AIR_HU_THRESH) & body_bool).astype(np.uint8)

            ct_over = ct_np.copy()
            ct_over[air_np.astype(bool)] = VAL_AIR
            tissue_win = (ct_np >= TISSUE_MIN) & (ct_np <= TISSUE_MAX) & body_bool & (~air_np.astype(bool))
            ct_over[tissue_win] = VAL_TISSUE
            write_itk(to_img(ct_over, ct_itk), ct_out)

            have_ct_row = True
            print(f"CT overwritten → {ct_out.name}")
        else:
            print(f"Missing CT for {case_id}")

        if have_mr_row and mr_out.exists():
            sx, sy, sz = read_spacing_from_nifti(mr_out)
            rows.append({
                "Patient": case_id,
                "TreatmentDay": "day0",
                "Folder": case_id,
                "ModalityFolder": "MR",
                "Modality": "MR",
                "PathNIFTI": str(out_dir),
                "PixelSpacing": f"({sx}, {sy})",
                "SliceThickness": float(sz),
                "TumourZcentrSlide": 0,
                "Treatment": "T0",
            })

        if have_ct_row and ct_out.exists():
            sx, sy, sz = read_spacing_from_nifti(ct_out)
            rows.append({
                "Patient": case_id,
                "TreatmentDay": "day0",
                "Folder": case_id,
                "ModalityFolder": "CT_reg",
                "Modality": "CT",
                "PathNIFTI": str(out_dir),
                "PixelSpacing": f"({sx}, {sy})",
                "SliceThickness": float(sz),
                "TumourZcentrSlide": 0,
                "Treatment": "T0",
            })

        processed += 1
        if processed % 50 == 0:
            print(f"Progress: {processed}/{n_cases} cases processed...")

    # Build DataFrame and save Excel
    df = pd.DataFrame(rows)
    df.to_excel(excel_path)  # keep index for compatibility with index_col=0 if you prefer
    print(f"Done processing. Patient folders with outputs: {processed} out of {n_cases} total.")
    print(f"Saved Excel: {excel_path}")
    print(f"Prepared NIfTI folders under: {nifti_root}")

if __name__ == "__main__":
    main()
