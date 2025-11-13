from pathlib import Path
import shutil

# ============= CONFIG =============
save_zipped = True  # ensure naming matches your current files
MR_SUBDIR = "MR"
MASK_SUBDIR = "masks"

# Unified combined paths
BASE_ROOT = Path("/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed")
in_path = BASE_ROOT / "2resampledNifti"
out_path = BASE_ROOT / "3resampledNiftiNyulReady"

# =================================




def get_first_nifti(folder: Path):
    if not folder.exists():
        return None
    for pattern in ("*.nii.gz", "*.nii"):
        files = sorted(folder.glob(pattern))
        if files:
            return files[0]
    return None


def prepare_case(case_dir: Path, out_root: Path):
    case_id = case_dir.name
    mr_in = get_first_nifti(case_dir / MR_SUBDIR)
    mask_in = get_first_nifti(case_dir / MASK_SUBDIR)

    if mr_in is None:
        print(f"[SKIP] {case_id}: no MR found")
        return

    # destination dirs
    mr_out_dir = out_root / "MR"
    mask_out_dir = out_root / "masks"
    mr_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    suffix = ".nii.gz" if save_zipped else ".nii"

    # copy MR
    mr_out = mr_out_dir / f"{case_id}_MR{suffix}"
    shutil.copy2(mr_in, mr_out)

    # copy mask if available
    if mask_in is not None:
        mask_out = mask_out_dir / f"{case_id}_mask{suffix}"
        shutil.copy2(mask_in, mask_out)

    print(f"[OK] {case_id}")


def main():

    for case_dir in sorted(in_path.iterdir()):
        if case_dir.is_dir():
            prepare_case(case_dir, out_path)


if __name__ == "__main__":
    main()
