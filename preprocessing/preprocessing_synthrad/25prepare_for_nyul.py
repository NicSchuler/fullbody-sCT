"""Stage MR + mask volumes for Nyul training and application.

Input (resampled):
    /.../2resampledNifti/<PATIENT_ID>/<MR_SUBDIR>/*.nii[.gz]
    /.../2resampledNifti/<PATIENT_ID>/<MASK_SUBDIR>/*.nii[.gz]

Outputs (Nyul-ready):
    /.../3resampledNiftiNyulReady/trainingforcalc/{MR,masks}/<PATIENT>_*.nii[.gz]
        - contains TRAIN only (used to fit Nyul)
    /.../3resampledNiftiNyulReady/valtest/{MR,masks}/<PATIENT>_*.nii[.gz]
        - contains VAL + TEST (to be normalized using train-fit mapping)

Usage:
    python 25prepare_for_nyul.py \
        --base-root /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed \
        --manifest   /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/splits_manifest.csv

Optional:
    --no-gzip        (write .nii instead of .nii.gz)

Manifest requirements:
    CSV with columns: split, patient_token
"""

from pathlib import Path
import shutil
import argparse
import csv
import re
from typing import Optional

MR_SUBDIR = "MR"
MASK_SUBDIR = "masks"

# Accept tokens like 1ABA005, 1HND012, 1BA123 even when prefixed in folder names
RE_TOKEN = re.compile(r"1(?:AB|HN|TH|B|P)[A-D][0-9]{3}")


def extract_token(text: str) -> Optional[str]:
    m = RE_TOKEN.search(text)
    return m.group(0) if m else None


DEFAULT_BASE = Path("/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed")


def parse_args():
    p = argparse.ArgumentParser(description="Prepare MR + masks for Nyul: train-only fitting set + all-data set.")
    p.add_argument("--base-root", default=str(DEFAULT_BASE), help="Pipeline base root containing 2resampledNifti")
    p.add_argument("--manifest", default=str(DEFAULT_BASE / "splits_manifest.csv"), help="CSV manifest with split,patient_token columns")
    p.add_argument("--no-gzip", action="store_true", help="Write .nii instead of .nii.gz")
    return p.parse_args()


def load_split_tokens(manifest: Path) -> dict:
    splits = {"train": set(), "val": set(), "test": set()}
    with manifest.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if "split" not in reader.fieldnames or "patient_token" not in reader.fieldnames:
            raise ValueError("Manifest must contain columns: split, patient_token")
        for row in reader:
            split = row["split"].strip().lower()
            token = row["patient_token"].strip()
            if split in splits and token:
                splits[split].add(token)
    return splits


def get_first_nifti(folder: Path):
    if not folder.exists():
        return None
    for pattern in ("*.nii.gz", "*.nii"):
        files = sorted(folder.glob(pattern))
        if files:
            return files[0]
    return None


def prepare_case(case_dir: Path, out_root: Path, zipped: bool, out_token: Optional[str] = None):
    case_id = out_token or case_dir.name
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
    out_base = base_root / "3resampledNiftiNyulReady"
    manifest = Path(args.manifest)
    if not in_path.exists():
        raise SystemExit(f"Input path missing: {in_path}")
    if not manifest.exists():
        raise SystemExit(f"Manifest missing: {manifest}")
    split_tokens = load_split_tokens(manifest)
    train_tokens = split_tokens.get("train", set())
    val_tokens = split_tokens.get("val", set())
    test_tokens = split_tokens.get("test", set())
    valtest_tokens = set().union(val_tokens, test_tokens)
    if not train_tokens and not valtest_tokens:
        raise SystemExit("No tokens loaded (check manifest and splits).")

    train_out = out_base / "trainingforcalc"
    valtest_out = out_base / "valtest"
    staged_train = 0
    staged_valtest = 0
    for case_dir in sorted(in_path.iterdir()):
        if not case_dir.is_dir():
            continue
        case_token = extract_token(case_dir.name) or case_dir.name
        did_any = False
        if case_token in valtest_tokens:
            if prepare_case(case_dir, valtest_out, zipped=not args.no_gzip, out_token=case_token):
                staged_valtest += 1
                did_any = True
        if case_token in train_tokens:
            if prepare_case(case_dir, train_out, zipped=not args.no_gzip, out_token=case_token):
                staged_train += 1
                did_any = True
        if not did_any:
            continue
    print(f"[DONE] Staged train-for-calc: {staged_train} -> {train_out}")
    print(f"[DONE] Staged val+test:       {staged_valtest} -> {valtest_out}")

    def count_in(dir_path: Path) -> int:
        if not dir_path.exists():
            return 0
        return sum(1 for _ in dir_path.glob("*.nii")) + sum(1 for _ in dir_path.glob("*.nii.gz"))

    tr_mr = count_in(train_out / "MR")
    tr_ms = count_in(train_out / "masks")
    vt_mr = count_in(valtest_out / "MR")
    vt_ms = count_in(valtest_out / "masks")

    print(f"[COUNT] trainingforcalc -> MR: {tr_mr}, masks: {tr_ms}")
    print(f"[COUNT] valtest         -> MR: {vt_mr}, masks: {vt_ms}")


if __name__ == "__main__":
    main()
