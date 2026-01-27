import argparse
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm

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

# Background values
CT_BACKGROUND = -1024
MR_BACKGROUND = 0
MASK_BACKGROUND = 0
# ==========================

def crop_pad_xy(arr: np.ndarray,
                body_part: str,
                background: float,
                current_spacing: tuple
        ) -> np.ndarray:
    # ---------------------------------------------
    #   Region-based target size BEFORE resampling
    # ---------------------------------------------
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

        return arr, new_spacing

    # ---------------------------------------------
    #   Return for HN/BRAIN (no resampling)
    # ---------------------------------------------
    return arr, current_spacing


def _compute_mask_bbox_xy(mask_arr: np.ndarray, margin: int = 0):
    """Compute 2D bounding box (y0:y1, x0:x1) over all z slices for a 3D mask.

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


def crop_img_to_mask_bbox(
    img: sitk.Image,
    mask_img: sitk.Image,
    background: float,
    target_xy: int = 256,
):
    """Crop to the mask XY bounding box, then pad/crop to (256x256) (resampling by factor 2 if necessary).

    Returns:
        out_arr (np.ndarray z,y,x), out_spacing (tuple)
    """
    # Convert mask to array to compute bbox in array coordinates (z, y, x)
    mask_arr = sitk.GetArrayFromImage(mask_img)
    bbox = _compute_mask_bbox_xy(mask_arr, margin=0)
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

    return cropped_img, cropped_mask


def process_image(
    in_path: Path,
    out_path: Path,
    background: float,
    mask_path: Path = None,
    save_mask: bool = False,
    mask_out_path: Path = None,
    crop_to_mask: bool = True,
):
    """
    Load NIfTI from in_path, center crop/pad x/y to TARGET_SIZE_XY,
    preserve spacing/origin/direction, and write to out_path.
    """
    # load image and mask
    img = sitk.ReadImage(str(in_path))
    mask_img = sitk.ReadImage(str(mask_path))

    body_part = in_path.name.split("_")[0]

    if crop_to_mask:
        # crop image to bbox of mask (smallest possible X/Y space without cropping mask)
        cropped_img, cropped_mask = crop_img_to_mask_bbox(
            img,
            mask_img,
            background=background,
            target_xy=TARGET_SIZE_XY,
        )
    else:
        cropped_img = img
        cropped_mask = mask_img

    # bring to correct final output size by padding and then cropping
    arr_img, new_spacing = crop_pad_xy(
        sitk.GetArrayFromImage(cropped_img),
        body_part=body_part,
        background=background,
        current_spacing=cropped_img.GetSpacing()
    )

    arr_mask, _ = crop_pad_xy(
        sitk.GetArrayFromImage(cropped_mask),
        body_part=body_part,
        background=0,
        current_spacing=cropped_mask.GetSpacing()
    )

    # Apply mask outside region
    arr_img[arr_mask == 0] = background

    # save output image
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = sitk.GetImageFromArray(arr_img)
    out_img.SetSpacing(new_spacing)
    out_img.SetDirection(img.GetDirection())
    out_img.SetOrigin(img.GetOrigin())
    sitk.WriteImage(out_img, str(out_path), useCompression=save_zipped)

    if save_mask and mask_out_path:
        mask_out_path.parent.mkdir(parents=True, exist_ok=True)
        out_mask = sitk.GetImageFromArray(arr_mask)
        out_mask.SetSpacing(new_spacing)
        out_mask.SetDirection(img.GetDirection())
        out_mask.SetOrigin(img.GetOrigin())
        out_mask = sitk.Cast(out_mask, sitk.sitkUInt8) # only for mask, not for 
        sitk.WriteImage(out_mask, str(mask_out_path), useCompression=save_zipped)


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


def process_case(case_dir: Path, out_root: Path, mr_only: bool = False, crop_to_mask: bool = True):
    """
    case_dir structure (current):
      case_dir/
        CT_reg/*.nii.gz
        MR/*.nii.gz
        new_masks/*.nii.gz   (optional, can also be in 'masks/' folder)

    Output:
      out_root/case_id/CT_reg/<ct_name>_cp256.nii[.gz]  (skipped if mr_only)
      out_root/case_id/MR/<mr_name>_cp256.nii[.gz]
      out_root/case_id/new_masks/<mask_name>_cp256.nii[.gz]

    Args:
        case_dir: Input case directory
        out_root: Output root directory
        mr_only: If True, skip CT processing (for inference)
        crop_to_mask: If True, crop to mask bounding box (default training behavior)
    """
    case_id = case_dir.name

    ct_in = get_first_nifti(case_dir / "CT_reg")
    mr_in = get_first_nifti(case_dir / "MR")
    # Check both 'new_masks' and 'masks' folders for backward compatibility
    mask_in = get_first_nifti(case_dir / "new_masks")
    if mask_in is None:
        mask_in = get_first_nifti(case_dir / "masks")

    # Check requirements based on mode
    if not mr_only and ct_in is None:
        print(f"[SKIP] {case_id}: missing CT_reg (use --mr-only for inference)")
        return

    if mr_in is None:
        print(f"[SKIP] {case_id}: missing MR")
        return

    if mask_in is None:
        print(f"[SKIP] {case_id}: missing mask (new_masks or masks folder)")
        return

    out_case = out_root / case_id

    # Process CT only if not in MR-only mode
    if not mr_only and ct_in is not None:
        ct_out = out_case / "CT_reg" / make_out_name(ct_in)
        process_image(ct_in, ct_out, CT_BACKGROUND, mask_path=mask_in,
                      save_mask=False, crop_to_mask=crop_to_mask)

    # Always process MR
    mr_out = out_case / "MR" / make_out_name(mr_in)
    mask_out = out_case / "new_masks" / make_out_name(mask_in)
    process_image(mr_in, mr_out, MR_BACKGROUND, mask_path=mask_in,
                  save_mask=True, mask_out_path=mask_out, crop_to_mask=crop_to_mask)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Resample NIfTI volumes to target size with optional mask cropping.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Original behavior (training data preprocessing):
    python 20resampling.py

    # Inference mode (MR only, no crop to bbox):
    python 20resampling.py --mr-only --skip-crop-to-bbox \\
        --src-root /path/to/input/1initNifti \\
        --out-root /path/to/output/2resampledNifti

    # Process specific patients:
    python 20resampling.py --patient-ids AB_1ABA005 AB_1ABA006
        """
    )

    parser.add_argument(
        "--src-root", type=str, default=None,
        help=f"Source directory (default: {src_root})"
    )
    parser.add_argument(
        "--out-root", type=str, default=None,
        help=f"Output directory (default: {out_path})"
    )
    parser.add_argument(
        "--mr-only", action="store_true",
        help="Process MR only, skip CT requirement (for inference)"
    )
    parser.add_argument(
        "--skip-crop-to-bbox", action="store_true",
        help="Skip crop to mask bounding box (use center crop/pad instead)"
    )
    parser.add_argument(
        "--patient-ids", nargs="+", default=None,
        help="Specific patient IDs to process (default: all)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Use command line args or defaults
    input_root = Path(args.src_root) if args.src_root else src_root
    output_root = Path(args.out_root) if args.out_root else out_path
    mr_only = args.mr_only
    crop_to_mask = not args.skip_crop_to_bbox

    print("=" * 60)
    print("Resampling NIfTI volumes")
    print("=" * 60)
    print(f"Source:          {input_root}")
    print(f"Output:          {output_root}")
    print(f"MR only:         {mr_only}")
    print(f"Crop to mask:    {crop_to_mask}")
    print(f"Target size XY:  {TARGET_SIZE_XY}")
    print("=" * 60)

    # Get case directories
    if args.patient_ids:
        case_dirs = [input_root / pid for pid in args.patient_ids if (input_root / pid).is_dir()]
        if not case_dirs:
            print(f"[ERROR] No valid patient directories found for: {args.patient_ids}")
            return
    else:
        case_dirs = [d for d in sorted(input_root.iterdir()) if d.is_dir()]

    print(f"Processing {len(case_dirs)} cases...")

    for case_dir in tqdm(case_dirs):
        process_case(case_dir, output_root, mr_only=mr_only, crop_to_mask=crop_to_mask)

    print("Resampling complete.")


if __name__ == "__main__":
    main()
