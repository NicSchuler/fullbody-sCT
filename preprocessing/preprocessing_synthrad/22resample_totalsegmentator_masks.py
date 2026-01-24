"""
Resample TotalSegmentator masks to match the resampled CT/MR images.

This script takes the TotalSegmentator masks (liver, torso_fat, subcutaneous_fat, etc.)
from 1initNifti and applies the same transformations as 20resampling.py to produce
masks that are aligned with the resampled images in 2resampledNifti.
It prefers MR-derived masks when available and falls back to CT_reg masks otherwise.

The transformations include:
1. Crop to body mask bounding box
2. Pad/crop to target size (512x512 for TH/AB/PELVIS, 256x256 for HN/BRAIN)
3. Downsample by factor 2 for TH/AB/PELVIS regions

Usage:
    python 22resample_totalsegmentator_masks.py
"""

import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Optional, Tuple
from tqdm import tqdm

# ==========================
# Global config
# ==========================
TARGET_SIZE_XY = 256

# Unified combined input/output roots
BASE_ROOT = Path("/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed")

src_root = BASE_ROOT / "1initNifti"
out_root = BASE_ROOT / "2resampledNifti"
save_zipped = True

# Masks to process from totalsegmentator_output
TOTALSEGMENTATOR_MASKS = [
    "liver.nii.gz",
    "torso_fat.nii.gz", 
    "subcutaneous_fat.nii.gz",
    "skeletal_muscle.nii.gz",
]

# Prefer MR-derived masks when available, then fall back to CT_reg (order is priority)
TS_MODALITY_DIRS = ["MR", "CT_reg"]

# Background value for masks
MASK_BACKGROUND = 0


def crop_pad_xy(arr: np.ndarray,
                body_part: str,
                background: float,
                current_spacing: tuple
        ) -> tuple:
    """
    Pad/crop array to target size and optionally downsample.
    
    For TH/AB/PELVIS: pad/crop to 512x512 then downsample to 256x256
    For HN/BRAIN: pad/crop to 256x256 (no downsampling)
    
    Returns:
        (arr, new_spacing) tuple
    """
    # Region-based target size BEFORE resampling
    if body_part.upper() in {"TH", "AB", "PELVIS"}:
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
    if y > target_xy:
        start_y = (y - target_xy) // 2
        arr = arr[:, start_y:start_y + target_xy, :]

    if x > target_xy:
        start_x = (x - target_xy) // 2
        arr = arr[:, :, start_x:start_x + target_xy]

    # Safety check
    assert arr.shape[1] == target_xy and arr.shape[2] == target_xy

    # ---- Downsample (TH/AB/PELVIS only) ----
    if need_downsample:
        # Convert to SITK image
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing(current_spacing)

        sx, sy, sz = current_spacing
        
        # New spacing doubled
        new_spacing = (sx * 2, sy * 2, sz)

        # New size: half in x/y
        new_size = [
            arr.shape[2] // 2,   # x dimension (width)
            arr.shape[1] // 2,   # y dimension (height)
            arr.shape[0],        # z dimension (num slices)
        ]

        # Resample using nearest neighbor for masks
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(background)
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetOutputOrigin(img.GetOrigin())
        resample_img = resampler.Execute(img)

        arr = sitk.GetArrayFromImage(resample_img)
        return arr, new_spacing

    # Return for HN/BRAIN (no resampling)
    return arr, current_spacing


def _compute_mask_bbox_xy(mask_arr: np.ndarray, margin: int = 0):
    """
    Compute 2D bounding box (y0:y1, x0:x1) over all z slices for a 3D mask.

    Parameters:
        mask_arr: np.ndarray (z, y, x) with binary mask values
        margin: int pixels to extend on all sides (default 0)

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


def crop_to_body_mask_bbox(
    img: sitk.Image,
    body_mask_img: sitk.Image,
):
    """
    Crop image to the body mask XY bounding box.
    
    Returns:
        cropped_img (sitk.Image)
    """
    mask_arr = sitk.GetArrayFromImage(body_mask_img)
    bbox = _compute_mask_bbox_xy(mask_arr, margin=0)
    
    if bbox is None:
        # No foreground; return original image
        return img
    
    y0, y1, x0, x1 = bbox
    w = x1 - x0
    h = y1 - y0
    
    cropped_img = _roi_crop_sitk(img, x0, y0, w, h)
    return cropped_img


def get_first_nifti(folder: Path):
    """Return first .nii.gz or .nii found in folder, or None."""
    if not folder.exists():
        return None
    for pattern in ("*.nii.gz", "*.nii"):
        files = sorted(folder.glob(pattern))
        if files:
            return files[0]
    return None


def find_totalseg_dir(case_dir: Path) -> Tuple[Optional[Path], Optional[str]]:
    """
    Find the first available TotalSegmentator output folder by modality priority.

    Returns:
        (totalseg_dir, modality_name) or (None, None) if not found.
    """
    for modality in TS_MODALITY_DIRS:
        ts_dir = case_dir / modality / "totalsegmentator_output"
        if ts_dir.exists():
            return ts_dir, modality
    return None, None


def make_out_name(mask_name: str) -> str:
    """
    Create output filename for resampled mask.
    e.g. liver.nii.gz -> liver_cp256.nii.gz
    """
    if mask_name.endswith(".nii.gz"):
        base = mask_name[:-7]
    elif mask_name.endswith(".nii"):
        base = mask_name[:-4]
    else:
        base = mask_name
    
    suffix = ".nii.gz" if save_zipped else ".nii"
    return f"{base}_cp{TARGET_SIZE_XY}{suffix}"


def process_totalsegmentator_masks(case_dir: Path, out_case_dir: Path):
    """
    Process TotalSegmentator masks for a single case.
    
    Applies the same transformations as 20resampling.py:
    1. Load body mask from original new_masks folder
    2. Crop to body mask bounding box
    3. Pad/crop to target size
    4. Downsample if needed (TH/AB/PELVIS)
    
    Saves masks to: out_case_dir/totalsegmentator_masks/
    """
    case_id = case_dir.name
    body_part = case_id.split("_")[0]  # e.g., "AB" from "AB_1ABA005"
    
    # Find body mask in original folder
    body_mask_path = get_first_nifti(case_dir / "new_masks")
    if body_mask_path is None:
        print(f"[SKIP] {case_id}: no body mask found in new_masks/")
        return
    
    # Load body mask with error handling for non-orthonormal direction cosines
    try:
        body_mask_img = sitk.ReadImage(str(body_mask_path))
    except RuntimeError as e:
        if "orthonormal" in str(e).lower():
            print(f"[SKIP] {case_id}: body mask has non-orthonormal direction cosines")
            return
        raise
    
    # TotalSegmentator output directory (prefer MR, fallback CT_reg)
    ts_dir, ts_modality = find_totalseg_dir(case_dir)
    if ts_dir is None:
        print(f"[SKIP] {case_id}: no totalsegmentator_output directory")
        return
    
    # Output directory for resampled masks
    out_ts_dir = out_case_dir / "totalsegmentator_masks"
    out_ts_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each mask
    for mask_name in TOTALSEGMENTATOR_MASKS:
        mask_path = ts_dir / mask_name
        if not mask_path.exists():
            print(f"  [SKIP] {case_id}: {mask_name} not found in {ts_modality}")
            continue
        
        # Load mask
        mask_img = sitk.ReadImage(str(mask_path))
        
        # Step 1: Crop to body mask bounding box (same crop as images)
        cropped_mask = crop_to_body_mask_bbox(mask_img, body_mask_img)
        
        # Step 2 & 3: Pad/crop to target size and downsample if needed
        arr_mask, new_spacing = crop_pad_xy(
            sitk.GetArrayFromImage(cropped_mask),
            body_part=body_part,
            background=MASK_BACKGROUND,
            current_spacing=cropped_mask.GetSpacing()
        )
        
        # Create output image
        out_mask = sitk.GetImageFromArray(arr_mask)
        out_mask.SetSpacing(new_spacing)
        out_mask.SetDirection(mask_img.GetDirection())
        out_mask.SetOrigin(mask_img.GetOrigin())
        out_mask = sitk.Cast(out_mask, sitk.sitkUInt8)
        
        # Save
        out_path = out_ts_dir / make_out_name(mask_name)
        sitk.WriteImage(out_mask, str(out_path), useCompression=save_zipped)


def main():
    print("=" * 60)
    print("Resampling TotalSegmentator Masks")
    print("=" * 60)
    print(f"Source: {src_root}")
    print(f"Output: {out_root}")
    print(f"Masks to process: {TOTALSEGMENTATOR_MASKS}")
    print()
    
    # Process only abdomen cases (AB_* prefix)
    cases = sorted([d for d in src_root.iterdir() if d.is_dir() and d.name.startswith("AB_")])
    print(f"Found {len(cases)} abdomen cases to process")
    
    for case_dir in tqdm(cases, desc="Processing cases"):
        out_case_dir = out_root / case_dir.name
        process_totalsegmentator_masks(case_dir, out_case_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
