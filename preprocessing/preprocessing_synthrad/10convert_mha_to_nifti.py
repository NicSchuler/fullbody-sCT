from pathlib import Path
import pathlib
import SimpleITK as sitk

# --- INPUTS ---
src_root = Path("/local/scratch/datasets/FullbodySCT/SynthRAD2025/task1_backup/0initDataTrainTask1/Task1")
out_path = Path("/local/scratch/datasets/FullbodySCT/SynthRAD2025/task1_backup/1initNifti")
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


def convert_case(case_dir: pathlib.Path, site_name: str, out_root: pathlib.Path):
    """
    """
    patient_raw = case_dir.name
    patient_id = f"{site_name}_{patient_raw}"

    ct_path = case_dir / "ct.mha"
    mr_path = case_dir / "mr.mha"
    mask_path = case_dir / "mask.mha"

    if not (ct_path.exists() and mr_path.exists()):
        print(f"[SKIP] {case_dir}: missing ct.mha or mr.mha")
        return

    # Read images
    ct_img = read_image(ct_path)
    mr_img = read_image(mr_path)

    # You can add checks here if you expect them to be already registered
    # (same size, spacing, direction, etc.)

    out_case = out_root / patient_id
    ct_out = out_case / "CT_reg" / f"{patient_id}_CT_reg{file_suffix}"
    mr_out = out_case / "MR" / f"{patient_id}_MR{file_suffix}"

    save_nifti(ct_img, ct_out)
    save_nifti(mr_img, mr_out)

    # Optional mask (saved as label image)
    if mask_path.exists():
        mask_img = read_image(mask_path)
        mask_img = sitk.Cast(mask_img, sitk.sitkUInt8)
        mask_out = out_case / "masks" / f"{patient_id}_mask{file_suffix}"
        save_nifti(mask_img, mask_out)

    print(f"[OK] {case_dir} -> {out_case}")


def main():

    for site_dir in sorted(src_root.iterdir()):
        if not site_dir.is_dir():
            continue
        site_name = site_dir.name  # e.g. AB, HN, TH

        for case_dir in sorted(site_dir.iterdir()):
            if case_dir.is_dir():
                convert_case(case_dir, site_name, out_path)

if __name__ == "__main__":
    main()
