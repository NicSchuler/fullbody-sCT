from pathlib import Path
import SimpleITK as sitk
import pandas as pd
import numpy as np

# --- INPUTS ---
src_root = Path("/local/scratch/datasets/FullbodySCT/SynthRAD2025/synthRAD2025_Task1_Train")
base_path = Path("/local/scratch/datasets/FullbodySCT/SynthRAD2025/synthRAD2025_Task1_Train_Nifti")  # must match resample.py
nifti_root = base_path / "nifti"
excel_dir = base_path / "excel"
excel_path = excel_dir / "data_CT_MR_TEMP_second_paper.xlsx"

# --- CREATE OUTPUT DIRS ---
nifti_root.mkdir(parents=True, exist_ok=True)
excel_dir.mkdir(parents=True, exist_ok=True)

rows = []  # rows for the Excel

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

        # Convert if present (guard against None)
        if ct_mha and ct_mha.exists():
            convert_mha_to_nii(ct_mha, ct_out)
            print(f"  ✔ CT   → {ct_out.name}")
        else:
            print(f"  ⚠ Missing CT for {case_id}")

        if mr_mha and mr_mha.exists():
            convert_mha_to_nii(mr_mha, mr_out)
            print(f"  ✔ MR   → {mr_out.name}")
        else:
            print(f"  ⚠ Missing MR for {case_id}")

        if body_mask_mha and body_mask_mha.exists():
            convert_mha_to_nii(body_mask_mha, body_mask_out)
            print(f"  ✔ Mask → {body_mask_out.name}")
        else:
            print(f"  ⚠ Missing body mask for {case_id}")

        # Add Excel rows for CT + MR if their files exist
        # Columns used by resample.py: Patient, TreatmentDay, Folder, ModalityFolder, Modality, PathNIFTI,
        # PixelSpacing, SliceThickness, TumourZcentrSlide, Treatment
        # Defaults: TreatmentDay='day0', Treatment='T0', TumourZcentrSlide=0
        if mr_out.exists():
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

        if ct_out.exists():
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

    # Build DataFrame and save Excel
    df = pd.DataFrame(rows)
    # Keep index so it stays compatible with pd.read_excel(..., index_col=0) in resample.py
    df.to_excel(excel_path)
    print(f"\nSaved Excel: {excel_path}")
    print(f"Prepared NIfTI folders under: {nifti_root}")

if __name__ == "__main__":
    main()
