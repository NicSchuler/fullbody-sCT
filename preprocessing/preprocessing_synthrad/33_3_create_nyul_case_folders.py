#!/usr/bin/env python3
"""
Create per-patient folders for Nyul-normalized MR data to match 32p99 layout.

This script:
  - uses the case folder structure from a baseline 3normalized root
  - copies new_masks folders from baseline
  - optionally copies CT_reg from a separate root (e.g., resampled, unnormalized)
  - injects Nyul MR files from a flat folder into each case/MR

Default layout (experiment2):
  baseline: /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/31baseline/3normalized
  nyul:     /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/3_33nyul/3normalized
  output:   /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/3_33nyul/3normalized_by_case

Usage:
  python 35_create_nyul_case_folders.py
  python 35_create_nyul_case_folders.py \
    --baseline-root /path/to/31baseline/3normalized \
    --nyul-root /path/to/33nyul/3normalized \
    --out-root /path/to/33nyul/3normalized_by_case

  # Copy unnormalized CT from resampled root
  python 35_create_nyul_case_folders.py \
    --baseline-root /path/to/31baseline/3normalized \
    --nyul-root /path/to/33nyul/3normalized \
    --out-root /path/to/33nyul/3normalized_by_case \
    --copy-ct --ct-root /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/31baseline/3normalized
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

RE_TOKEN = re.compile(r"1(?:AB|HN|TH|B|P)[A-D][0-9]{3}")


def extract_token(text: str) -> Optional[str]:
    m = RE_TOKEN.search(text)
    return m.group(0) if m else None


def first_nifti(folder: Path) -> Optional[Path]:
    if not folder.exists():
        return None
    for pattern in ("*.nii.gz", "*.nii"):
        files = sorted(folder.glob(pattern))
        if files:
            return files[0]
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create per-patient Nyul folders from flat MR outputs.")
    default_base = Path("/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed")
    p.add_argument(
        "--baseline-root",
        default=str(default_base / "2resampledNifti"),
        help="Baseline root with per-patient folders",
    )
    p.add_argument(
        "--nyul-root",
        default=str(default_base / "experiment2" / "3_33nyul" / "3_2normalized"),
        help="Nyul normalized MR folder (flat files)",
    )
    p.add_argument(
        "--out-root",
        default=str(default_base / "experiment2" / "3_33nyul" / "3normalized"),
        help="Output root for per-patient layout",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing out-root (default: error if exists)",
    )
    p.add_argument(
        "--copy-ct",
        action="store_true",
        help="Copy CT_reg into output (default: skip CT_reg)",
    )
    p.add_argument(
        "--ct-root",
        default=None,
        help="Root containing CT_reg folders (defaults to baseline-root when --copy-ct is set)",
    )
    p.add_argument(
        "--ct-subdir",
        default="CT_reg",
        help="CT subdirectory name inside each case (default: CT_reg)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without copying files",
    )
    p.add_argument(
        "--abdomen-only",
        action="store_true",
        help="Process only AB_* cases from baseline-root (default: all cases)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    baseline_root = Path(args.baseline_root)
    nyul_root = Path(args.nyul_root)
    out_root = Path(args.out_root)
    ct_root = Path(args.ct_root) if args.ct_root else baseline_root

    if not baseline_root.exists():
        raise SystemExit(f"Baseline root missing: {baseline_root}")
    if not nyul_root.exists():
        raise SystemExit(f"Nyul root missing: {nyul_root}")
    if args.copy_ct and not ct_root.exists():
        raise SystemExit(f"CT root missing: {ct_root}")
    if out_root.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists (use --overwrite): {out_root}")

    if out_root.exists() and args.overwrite and not args.dry_run:
        shutil.rmtree(out_root)
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    cases = sorted(d for d in baseline_root.iterdir() if d.is_dir())
    if args.abdomen_only:
        cases = [d for d in cases if d.name.startswith("AB_")]
    missing = 0
    copied = 0

    for case_dir in cases:
        token = extract_token(case_dir.name) or case_dir.name
        nyul_file = next(nyul_root.glob(f"{token}*MR*nii*"), None)
        if nyul_file is None:
            print(f"[WARN] Missing Nyul MR for token {token}")
            missing += 1
            continue

        out_case = out_root / case_dir.name
        out_mr_dir = out_case / "MR"
        if not args.dry_run:
            out_mr_dir.mkdir(parents=True, exist_ok=True)

        # Use baseline MR filename for consistency if it exists
        baseline_mr = first_nifti(case_dir / "MR")
        out_name = baseline_mr.name if baseline_mr else nyul_file.name

        if not args.dry_run:
            out_path = out_mr_dir / out_name
            img = nib.load(str(nyul_file))
            data = img.get_fdata()
            data = np.clip(data, 0.0, 1.0)
            header = img.header.copy()
            header.set_data_dtype(np.float32)
            clipped = nib.Nifti1Image(data.astype(np.float32, copy=False), img.affine, header)
            nib.save(clipped, str(out_path))

        # Always copy masks to keep structure consistent
        masks_src = case_dir / "new_masks"
        if masks_src.exists():
            if args.dry_run:
                print(f"[DRY] copytree {masks_src} -> {out_case / 'new_masks'}")
            else:
                shutil.copytree(masks_src, out_case / "new_masks", dirs_exist_ok=True)

        # Optionally copy CT (can be sourced from a different root)
        if args.copy_ct:
            ct_src = ct_root / case_dir.name / args.ct_subdir
            if ct_src.exists():
                if args.dry_run:
                    print(f"[DRY] copytree {ct_src} -> {out_case / 'CT_reg'}")
                else:
                    shutil.copytree(ct_src, out_case / "CT_reg", dirs_exist_ok=True)
            else:
                print(f"[WARN] Missing CT source for {case_dir.name}: {ct_src}")

        copied += 1
        if copied % 25 == 0:
            print(f"[OK] {copied} cases processed...")

    print(f"[DONE] Cases written: {copied}")
    if missing:
        print(f"[WARN] Missing Nyul MR for {missing} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
