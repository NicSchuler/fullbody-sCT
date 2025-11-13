import numpy as np
import SimpleITK as sitk
from pathlib import Path

# ==========================
# Global config
# ==========================
TARGET_SIZE_XY = 256

# Unified combined input/output roots under final base
BASE_ROOT = Path("/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed")
src_root = BASE_ROOT / "1initNifti"
out_path = BASE_ROOT / "2resampledNifti"
save_zipped = True

# Background values
CT_BACKGROUND = -1024
MR_BACKGROUND = 0
MASK_BACKGROUND = 0
# ==========================


def crop_pad_xy(arr: np.ndarray, target: int, background: float) -> np.ndarray:
    """
    Center crop/pad a 3D volume (z, y, x) to (z, target, target) in x/y.
    TODO: 
        currently the images are cropped in spatial size. 
        Check if we better adjust voxel size in terms of
            1. check what most images look like and define voxel size
                e.g. if most images are 320x320x320@1mm
                instead of cropping define 256@1.25mm
                resample all images --> 1.25
            2. crop / pad images which are too big/small
    """
    z, y, x = arr.shape

    # ---- pad if smaller ----
    pad_y = max(0, target - y)
    pad_x = max(0, target - x)

    if pad_y > 0 or pad_x > 0:
        py_before = pad_y // 2
        py_after = pad_y - py_before
        px_before = pad_x // 2
        px_after = pad_x - px_before

        arr = np.pad(
            arr,
            ((0, 0),
             (py_before, py_after),
             (px_before, px_after)),
            mode="constant",
            constant_values=background,
        )
        z, y, x = arr.shape

    # ---- crop if larger ----
    if y > target:
        start_y = (y - target) // 2
        arr = arr[:, start_y:start_y + target, :]

    if x > target:
        start_x = (x - target) // 2
        arr = arr[:, :, start_x:start_x + target]

    # safety check
    assert arr.shape[1] == target and arr.shape[2] == target, arr.shape
    return arr


def process_image(
    in_path: Path,
    out_path: Path,
    background: float,
    is_label: bool = False,
):
    """
    Load NIfTI from in_path, center crop/pad x/y to TARGET_SIZE_XY,
    preserve spacing/origin/direction, and write to out_path.
    """
    img = sitk.ReadImage(str(in_path))
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)

    arr_out = crop_pad_xy(arr, TARGET_SIZE_XY, background)

    out_img = sitk.GetImageFromArray(arr_out)
    out_img.SetSpacing(img.GetSpacing())
    out_img.SetDirection(img.GetDirection())
    out_img.SetOrigin(img.GetOrigin())

    if is_label:
        out_img = sitk.Cast(out_img, sitk.sitkUInt8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out_img, str(out_path), useCompression=save_zipped)


def get_first_nifti(folder: Path):
    """
    Return first .nii.gz or .nii found in folder, or None.
    """
    if not folder.exists():
        return None
    for pattern in ("*.nii.gz", "*.nii"):
        files = sorted(folder.glob(pattern))
        if files:
            return files[0]
    return None


def make_out_name(in_path: Path) -> str:
    """
    Take input filename and append _cp{TARGET_SIZE_XY} with correct suffix.
    Handles .nii and .nii.gz cleanly.
    """
    name = in_path.name
    if name.endswith(".nii.gz"):
        base = name[:-7]
    elif name.endswith(".nii"):
        base = name[:-4]
    else:
        base = name
    suffix = ".nii.gz" if save_zipped else ".nii"
    return f"{base}_cp{TARGET_SIZE_XY}{suffix}"


def process_case(case_dir: Path, out_root: Path):
    """
    case_dir structure (current):
      case_dir/
        CT_reg/*.nii.gz
        MR/*.nii.gz
        masks/*.nii.gz   (optional)

    Output:
      out_root/case_id/CT_reg/<ct_name>_cp256.nii[.gz]
      out_root/case_id/MR/<mr_name>_cp256.nii[.gz]
      out_root/case_id/masks/<mask_name>_cp256.nii[.gz]
    """
    case_id = case_dir.name

    ct_in = get_first_nifti(case_dir / "CT_reg")
    mr_in = get_first_nifti(case_dir / "MR")
    mask_in = get_first_nifti(case_dir / "masks")

    if ct_in is None or mr_in is None:
        print(f"[SKIP] {case_id}: missing CT_reg or MR")
        return

    out_case = out_root / case_id

    ct_out = out_case / "CT_reg" / make_out_name(ct_in)
    mr_out = out_case / "MR" / make_out_name(mr_in)

    process_image(ct_in, ct_out, CT_BACKGROUND, is_label=False)
    process_image(mr_in, mr_out, MR_BACKGROUND, is_label=False)

    if mask_in is not None:
        mask_out = out_case / "masks" / make_out_name(mask_in)
        process_image(mask_in, mask_out, MASK_BACKGROUND, is_label=True)

    print(f"[OK] {case_id}")


def main():

    for case_dir in sorted(src_root.iterdir()):
        if case_dir.is_dir():
            process_case(case_dir, out_path)


if __name__ == "__main__":
    main()
