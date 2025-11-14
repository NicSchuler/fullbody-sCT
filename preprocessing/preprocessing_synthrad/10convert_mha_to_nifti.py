from pathlib import Path
import pathlib
import SimpleITK as sitk
import shutil
# --- INPUTS ---
RAW_ROOTS = [
    Path("/local/scratch/datasets/FullbodySCT/SynthRAD2025/task1/initDataTrainTask1/Task1"),
    Path("/local/scratch/datasets/FullbodySCT/SynthRAD2023/task_1/Task1"),
]

BASE_ROOT = Path("/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/1initNifti")

save_zipped = True
file_suffix = ".nii.gz" if save_zipped else ".nii"

def read_image(path: pathlib.Path) -> sitk.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    img = sitk.ReadImage(str(path))
    if img.GetDimension() != 3:
        raise ValueError(f"{path} is not 3D (dim={img.GetDimension()})")
    return img


def save_nifti(img: sitk.Image, out_path: pathlib.Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path), useCompression=True)


def find_input(path_base: pathlib.Path):
    """Return the existing file among mha, nii, nii.gz (or None)."""
    for ext in [".mha", ".nii.gz", ".nii"]:
        p = path_base.with_suffix(ext)
        if p.exists():
            return p
    return None


def convert_case(case_dir: pathlib.Path, site_name: str, out_root: pathlib.Path):
    """
    """
    patient_raw = case_dir.name
    patient_id = f"{site_name}_{patient_raw}"

    ct_in = find_input(case_dir / "ct")
    mr_in = find_input(case_dir / "mr")
    mask_in = find_input(case_dir / "mask")

    if ct_in is None or mr_in is None:
        print(f"[SKIP] {case_dir}: missing CT or MR")
        return

    # You can add checks here if you expect them to be already registered
    # (same size, spacing, direction, etc.)

    out_case = out_root / patient_id
    ct_out = out_case / "CT_reg" / f"{patient_id}_CT_reg{file_suffix}"
    mr_out = out_case / "MR" / f"{patient_id}_MR{file_suffix}"


    ct_out.parent.mkdir(parents=True, exist_ok=True)
    mr_out.parent.mkdir(parents=True, exist_ok=True)

    if ct_in.suffix == ".mha":
        ct_img = sitk.ReadImage(str(ct_in))
        sitk.WriteImage(ct_img, str(ct_out))
    else:
        shutil.copy(ct_in, ct_out)

    # --- MR ---
    if mr_in.suffix == ".mha":
        mr_img = sitk.ReadImage(str(mr_in))
        sitk.WriteImage(mr_img, str(mr_out))
    else:
        shutil.copy(mr_in, mr_out)

    # Optional mask (saved as label image)
    if mask_in is not None:
            mask_out = out_case / "masks" / f"{patient_id}_mask.nii.gz"
            mask_out.parent.mkdir(parents=True, exist_ok=True)

            if mask_in.suffix == ".mha":
                mask_img = sitk.ReadImage(str(mask_in))
                mask_img = sitk.Cast(mask_img, sitk.sitkUInt8)
                sitk.WriteImage(mask_img, str(mask_out))
            else:
                shutil.copy(mask_in, mask_out)
            mask_img = read_image(mask_in)
            mask_img = sitk.Cast(mask_img, sitk.sitkUInt8)
            mask_out = out_case / "masks" / f"{patient_id}_mask{file_suffix}"
            save_nifti(mask_img, mask_out)

    print(f"[OK] {case_dir} -> {out_case}")


def main():
    BASE_ROOT.mkdir(parents=True, exist_ok=True)
    for raw_root in RAW_ROOTS:
        if not raw_root.exists():
            print(f"[WARN] raw root missing: {raw_root}")
            continue
        for site_dir in sorted(raw_root.iterdir()):
            if not site_dir.is_dir():
                continue
            site_name = site_dir.name  # e.g. AB / HN / TH / brain / pelvis
            for case_dir in sorted(site_dir.iterdir()):
                if case_dir.is_dir():
                    try:
                        convert_case(case_dir, site_name, BASE_ROOT)
                    except Exception as e:
                        print(f"[ERR] {case_dir}: {e}")

if __name__ == "__main__":
    main()
