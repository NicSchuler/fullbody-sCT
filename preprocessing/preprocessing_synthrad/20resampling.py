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

# Note: Output is NOT normalized - just resampled/cropped/masked
# Use scripts 201-204 for different normalization techniques

# Background values
CT_BACKGROUND = -1024
MR_BACKGROUND = 0
MASK_BACKGROUND = 0
# ==========================

# Normalization functions removed - see 201-204 standardization scripts


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


def _compute_mask_bbox_xy(mask_arr: np.ndarray, margin: int = 10):
    """Compute 2D bounding box (y0:y1, x0:x1) over all z slices for a 3D mask.

    Parameters:
        mask_arr: np.ndarray (z, y, x) with binary mask values
        margin: int pixels to extend on all sides

    Returns:
        (y0, y1, x0, x1) inclusive-exclusive indices
        Returns None if mask has no foreground.
    """
    # Collapse over z to find any foreground at each (y, x)
    proj = (mask_arr > 0).any(axis=0)  # shape (y, x)
    if not proj.any():
        return None

    ys, xs = np.where(proj)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    # Add margin
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    y1 = min(mask_arr.shape[1], y1 + margin)
    x1 = min(mask_arr.shape[2], x1 + margin)
    return y0, y1, x0, x1


def _roi_crop_sitk(img: sitk.Image, x0: int, y0: int, w: int, h: int) -> sitk.Image:
    """Crop SimpleITK image by XY ROI, keeping all z slices. Updates origin automatically."""
    size = [int(w), int(h), int(img.GetSize()[2])]
    index = [int(x0), int(y0), 0]
    return sitk.RegionOfInterest(img, size=size, index=index)


def _resample_fit_xy(img: sitk.Image, target_xy: int, interpolator, background: float) -> sitk.Image:
    """Resample X/Y to fit within target_xy keeping aspect ratio; keep Z unchanged.

    Returns a new SimpleITK image with updated spacing reflecting the resampling.
    """
    size = img.GetSize()  # (x, y, z)
    spacing = img.GetSpacing()

    in_w, in_h, in_z = size[0], size[1], size[2]
    if in_w == 0 or in_h == 0:
        return img

    scale = min(target_xy / in_w, target_xy / in_h)
    new_w = max(1, int(round(in_w * scale)))
    new_h = max(1, int(round(in_h * scale)))

    out_size = [new_w, new_h, in_z]
    out_spacing = (
        spacing[0] * (in_w / new_w),
        spacing[1] * (in_h / new_h),
        spacing[2],
    )

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(out_size)
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(float(background))
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    return resampler.Execute(img)


def crop_letterbox_to_mask(
    img: sitk.Image,
    mask_img: sitk.Image,
    background: float,
    target_xy: int = 256,
    modality: str = "CT",
    margin: int = 10,
):
    """Crop the image to the mask's XY bounding box (+margin), then fit to target size with letterbox.

    Steps:
      1) Compute mask XY bbox across all slices and crop both image and mask using SITK ROI (Z kept).
      2) Resample both to fit within target_xy preserving aspect (linear for image, NN for mask).
      3) Pad both to exactly (target_xy, target_xy) centered.
      4) Apply mask: set outside-mask voxels to background.

    Returns:
      out_arr (np.ndarray z,y,x), out_spacing (tuple)
    """
    # Convert mask to array to compute bbox in array coordinates (z, y, x)
    mask_arr = sitk.GetArrayFromImage(mask_img)
    bbox = _compute_mask_bbox_xy(mask_arr, margin=margin)
    if bbox is None:
        # No foreground; return a fully background image, padded to target
        arr = sitk.GetArrayFromImage(img)
        z, y, x = arr.shape
        out = np.full((z, target_xy, target_xy), background, dtype=arr.dtype)
        return out, img.GetSpacing()

    y0, y1, x0, x1 = bbox
    w = x1 - x0
    h = y1 - y0

    # Crop both image and mask in SITK (order: x,y,z)
    cropped_img = _roi_crop_sitk(img, x0, y0, w, h)
    cropped_mask = _roi_crop_sitk(mask_img, x0, y0, w, h)

    # Resample to fit
    interp_img = sitk.sitkLinear if modality.upper() in {"CT", "MR", "MRI"} else sitk.sitkLinear
    img_fit = _resample_fit_xy(cropped_img, target_xy, interp_img, background)
    mask_fit = _resample_fit_xy(cropped_mask, target_xy, sitk.sitkNearestNeighbor, 0)

    # Convert to arrays for padding
    arr_img = sitk.GetArrayFromImage(img_fit)   # (z, y, x)
    arr_mask = sitk.GetArrayFromImage(mask_fit)

    z, y, x = arr_img.shape
    pad_y = max(0, target_xy - y)
    pad_x = max(0, target_xy - x)

    py_before = pad_y // 2
    py_after = pad_y - py_before
    px_before = pad_x // 2
    px_after = pad_x - px_before

    arr_img = np.pad(
        arr_img,
        ((0, 0), (py_before, py_after), (px_before, px_after)),
        mode="constant",
        constant_values=background,
    )
    arr_mask = np.pad(
        arr_mask,
        ((0, 0), (py_before, py_after), (px_before, px_after)),
        mode="constant",
        constant_values=0,
    )

    # If still larger than target (rare rounding), center-crop
    arr_img = arr_img[:, :target_xy, :target_xy]
    arr_mask = arr_mask[:, :target_xy, :target_xy]

    # Ensure binary mask
    arr_mask = (arr_mask > 0.5).astype(np.uint8)

    # Apply mask outside region
    arr_img[arr_mask == 0] = background

    # Spacing after resample stays from img_fit
    new_spacing = img_fit.GetSpacing()
    return arr_img, new_spacing


def process_image(
    in_path: Path,
    out_path: Path,
    background: float,
    is_label: bool = False,
    mask_path: Path = None,
):
    """
    Load NIfTI from in_path, center crop/pad x/y to TARGET_SIZE_XY,
    preserve spacing/origin/direction, and write to out_path.
    
    If mask_path is provided, apply mask to set background values.
    
    NOTE: This script does NOT normalize values - output retains original HU/intensity values.
          Use scripts 201-204 for different normalization techniques.
    """
    img = sitk.ReadImage(str(in_path))
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    spacing = img.GetSpacing()

    body_part = in_path.name.split("_")[0]

    # Determine modality - check filename only, not full path
    is_ct = "CT" in in_path.name.upper()
    is_mr = "MR" in in_path.name.upper()

    # If a mask is provided, use mask-based crop + letterbox fit to 256
    if mask_path is not None and Path(mask_path).exists() and not is_label:
        mask_img = sitk.ReadImage(str(mask_path))
        out_arr, new_spacing = crop_letterbox_to_mask(
            img,
            mask_img,
            background=background,
            target_xy=TARGET_SIZE_XY,
            modality="CT" if is_ct else "MR",
            margin=10,
        )
        arr_out = out_arr
        print(f"  → Mask-bbox crop+letterbox applied to {in_path.name}; output shape: {arr_out.shape}")
    else:
        # Fallback to legacy center crop/pad + (optional region downsample)
        arr_out, new_spacing = crop_pad_xy(
            arr,
            body_part=body_part,
            background=background,
            current_spacing=spacing,
            return_new_spacing=True,
            filename=in_path.name,
        )
        # If mask exists even in fallback, blank outside mask
        if mask_path is not None and Path(mask_path).exists():
            mask_img = sitk.ReadImage(str(mask_path))
            mask_arr = sitk.GetArrayFromImage(mask_img)
            mask_out, _ = crop_pad_xy(
                mask_arr,
                body_part=body_part,
                background=0,
                current_spacing=mask_img.GetSpacing(),
                return_new_spacing=True,
                filename=mask_path.name,
            )
            if is_ct:
                arr_out[mask_out == 0] = -1024
            elif is_mr:
                arr_out[mask_out == 0] = 0

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

    # Process CT and MR with mask applied (if available)
    process_image(ct_in, ct_out, CT_BACKGROUND, is_label=False, mask_path=mask_in)
    process_image(mr_in, mr_out, MR_BACKGROUND, is_label=False, mask_path=mask_in)

    # Process mask itself (without applying mask to mask)
    if mask_in is not None:
        mask_out = out_case / "masks" / make_out_name(mask_in)
        process_image(mask_in, mask_out, MASK_BACKGROUND, is_label=True, mask_path=None)

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
