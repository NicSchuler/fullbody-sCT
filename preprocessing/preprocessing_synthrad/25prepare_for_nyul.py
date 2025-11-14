"""Stage MR + mask volumes for Nyul fitting/applying.

Now manifest-aware: Only TRAIN and VAL patients are staged by default.
Test patients are excluded to avoid inadvertent leakage in normalization fitting.

Input layout (resampled):
  /.../2resampledNifti/<PATIENT_ID>/<MR_SUBDIR>/*.nii[.gz]
  /.../2resampledNifti/<PATIENT_ID>/<MASK_SUBDIR>/*.nii[.gz]

Output layout (Nyul ready):
  /.../3resampledNiftiNyulReady/MR/<PATIENT_ID>_MR.nii.gz
  /.../3resampledNiftiNyulReady/masks/<PATIENT_ID>_mask.nii.gz

Usage:
  python 25prepare_for_nyul.py \
    --base-root /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed \
    --manifest   /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/splits_manifest.csv

Optional:
  --include-test   (also stage TEST patients)
  --no-gzip        (write .nii instead of .nii.gz)

Manifest requirements:
  CSV with columns: split, patient_token
"""

from pathlib import Path
import shutil
import argparse
import csv

MR_SUBDIR = "MR"
MASK_SUBDIR = "masks"


def parse_args():
    p = argparse.ArgumentParser(description="Prepare MR + masks for Nyul (train+val only by default).")
    p.add_argument("--base-root", required=True, help="Pipeline base root containing 2resampledNifti")
    p.add_argument("--manifest", required=True, help="CSV manifest with split,patient_token columns")
    p.add_argument("--include-test", action="store_true", help="Also stage TEST patients")
    p.add_argument("--no-gzip", action="store_true", help="Write .nii instead of .nii.gz")
    return p.parse_args()


def load_tokens(manifest: Path, include_test: bool) -> set:
    wanted_splits = {"train", "val"}
    if include_test:
        wanted_splits.add("test")
    tokens = set()
    with manifest.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if "split" not in reader.fieldnames or "patient_token" not in reader.fieldnames:
            raise ValueError("Manifest must contain columns: split, patient_token")
        for row in reader:
            split = row["split"].strip().lower()
            token = row["patient_token"].strip()
            if split in wanted_splits and token:
                tokens.add(token)
    return tokens


def get_first_nifti(folder: Path):
    if not folder.exists():
        return None
    for pattern in ("*.nii.gz", "*.nii"):
        files = sorted(folder.glob(pattern))
        if files:
            return files[0]
    return None


def prepare_case(case_dir: Path, out_root: Path, zipped: bool):
    case_id = case_dir.name
    mr_in = get_first_nifti(case_dir / MR_SUBDIR)
    mask_in = get_first_nifti(case_dir / MASK_SUBDIR)
    if mr_in is None:
        print(f"[SKIP] {case_id}: no MR found")
        return False
    mr_out_dir = out_root / "MR"
    mask_out_dir = out_root / "masks"
    mr_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".nii.gz" if zipped else ".nii"
    shutil.copy2(mr_in, mr_out_dir / f"{case_id}_MR{suffix}")
    if mask_in is not None:
        shutil.copy2(mask_in, mask_out_dir / f"{case_id}_mask{suffix}")
    print(f"[OK] {case_id}")
    return True


def main():
    args = parse_args()
    base_root = Path(args.base_root)
    in_path = base_root / "2resampledNifti"
    out_path = base_root / "3resampledNiftiNyulReady"
    manifest = Path(args.manifest)
    if not in_path.exists():
        raise SystemExit(f"Input path missing: {in_path}")
    if not manifest.exists():
        raise SystemExit(f"Manifest missing: {manifest}")
    tokens = load_tokens(manifest, include_test=args.include_test)
    if not tokens:
        raise SystemExit("No tokens loaded (check manifest and splits).")
    staged = 0
    for case_dir in sorted(in_path.iterdir()):
        if not case_dir.is_dir():
            continue
        if case_dir.name not in tokens:
            continue
        if prepare_case(case_dir, out_path, zipped=not args.no_gzip):
            staged += 1
    print(f"[DONE] Staged {staged} patients -> {out_path}")


if __name__ == "__main__":
    main()
