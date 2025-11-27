import numpy as np
import SimpleITK as sitk
from pathlib import Path

# ==========================
# Global config
# ==========================
TARGET_SIZE_XY = 256

# Unified combined input/output roots under final base
BASE_ROOT = Path("/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed")
#BASE_ROOT = Path("/local/scratch/datasets/FullbodySCT/SynthRAD2025/task1_backup")

src_root = BASE_ROOT / "1initNifti"
out_path = BASE_ROOT / "2resampledNifti"
save_zipped = True

# Background values
CT_BACKGROUND = -1024
MR_BACKGROUND = 0
MASK_BACKGROUND = 0
# ==========================

def normalize_ct(arr):
    """
    Normalize CT values to [0, 1] range using fixed HU window.
    Standard fullbody CT range: -1024 (air) to +1200 (dense bone)
    
    Mapping:
        -1024 HU (air/background) -> 0
        +1200 HU (dense bone) -> 1
    
    This ensures consistent scaling across all CT images.
    """
    arr = np.clip(arr, -1024, 1200)
    arr = (arr + 1024) / 2224.0  # Maps -1024→0, 1200→1
    return arr.astype(np.float32)

def normalize_mr(arr):
    """
    Normalize MRI values to [0, 1] range using 99th percentile of foreground.
    
    Mapping:
        0 (background) -> 0
        p99 (99th percentile of non-zero values) -> 1
        Values above p99 are clipped to 1
    
    This is per-image min-max rescaling with outlier removal.
    """
    # Calculate p99 only on non-zero (foreground) pixels
    foreground = arr[arr > 0]
    if foreground.size > 0:
        p99 = np.percentile(foreground, 99)
        if p99 > 0:
            arr = arr / p99  # Scale by p99
            arr = np.clip(arr, 0, 1)  # Clip to [0, 1] range
        else:
            arr = np.zeros_like(arr)
    else:
        arr = np.zeros_like(arr)
    return arr.astype(np.float32)


def crop_pad_xy(arr: np.ndarray,
                body_part: str,
                background: float,
                current_spacing: tuple,
                return_new_spacing=False,
                filename=None
        ) -> np.ndarray:
    bp = body_part.upper()

    # ---------------------------------------------
    #   Region-based target size BEFORE resampling
    # ---------------------------------------------
    print(f"[START CROP/PAD] {filename}: bodyPart={bp} > background: {background} > current_spacing: {current_spacing}")

    if bp in {"TH", "AB", "PELVIS"}:
        target_xy = 512
        need_downsample = True
    else:  # HN, BRAIN
        target_xy = 256
        need_downsample = False

    z, y, x = arr.shape

    # ---- pad if smaller ----
    pad_y = max(0, target_xy - y)
    pad_x = max(0, target_xy - x)

    if pad_y > 0 or pad_x > 0:
        py_before = pad_y // 2
        py_after  = pad_y - py_before
        px_before = pad_x // 2
        px_after  = pad_x - px_before

        arr = np.pad(
            arr,
            ((0, 0),
             (py_before, py_after),
             (px_before, px_after)),
            mode="constant",
            constant_values=background,
        )
        z, y, x = arr.shape
    # ---------------------------------------------
    #   Step 2: crop if larger
    # ---------------------------------------------
    if y > target_xy:
        start_y = (y - target_xy) // 2
        arr = arr[:, start_y:start_y + target_xy, :]

    if x > target_xy:
        start_x = (x - target_xy) // 2
        arr = arr[:, :, start_x:start_x + target_xy]

    # Safety
    assert arr.shape[1] == target_xy and arr.shape[2] == target_xy


    # ---------------------------------------------
    #   Step 3: Downsample (TH/AB/PELVIS only)
    # ---------------------------------------------
    if need_downsample:
        # Convert to SITK image
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing(current_spacing)

        sx, sy, sz = current_spacing
        
        # New spacing doubled
        new_spacing = (sx*2, sy*2, sz)

        # New size: half in x/y
        new_size = [
            arr.shape[2] // 2,   # x dimension (width)
            arr.shape[1] // 2,   # y dimension (height)
            arr.shape[0],        # z dimension (num slices)
        ]

        # Resample
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(background)
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetOutputOrigin(img.GetOrigin())
        resample_img = resampler.Execute(img)

        arr = sitk.GetArrayFromImage(resample_img)

        if return_new_spacing:
            return arr, new_spacing
        else:
            return arr

    # ---------------------------------------------
    #   Return for HN/BRAIN (no resampling)
    # ---------------------------------------------
    if return_new_spacing:
        return arr, current_spacing

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
    spacing = img.GetSpacing()

    body_part = in_path.name.split("_")[0]

    # Determine modality and normalization type - check filename only, not full path
    is_ct = "CT" in in_path.name
    is_mr = "MR" in in_path.name

    # First do geometric transformations with original values
    arr_out, new_spacing = crop_pad_xy(
        arr,
        body_part=body_part,
        background=background,
        current_spacing=spacing,
        return_new_spacing=True,
        filename=in_path.name
    )

    # Then normalize AFTER geometric transformations to avoid interpolation artifacts
    if is_ct:
        arr_out = normalize_ct(arr_out)
    elif is_mr:
        arr_out = normalize_mr(arr_out)

    out_img = sitk.GetImageFromArray(arr_out)
    out_img.SetSpacing(new_spacing)
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
    count = 0
    total = sum(1 for d in src_root.iterdir() if d.is_dir())

    for case_dir in sorted(src_root.iterdir()):
        if case_dir.is_dir():
            process_case(case_dir, out_path)
            count += 1

            if count % 25 == 0:
                print(f"Processed {count}/{total} items…")
        


if __name__ == "__main__":
    main()
