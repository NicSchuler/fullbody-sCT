#!/usr/bin/env python3
"""
Run TotalSegmentator on initial NIfTI cases.

Input:
  /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/1initNifti/<CASE>/{CT_reg,MR}/*.nii.gz

Output per modality:
  .../<CASE>/<MODALITY>/totalsegmentator_output/

This should be run after:
  10convert_mha_to_nifti.py
  11sanity_check.py
  12bodymasks_from_CT.py

and before:
  22resample_totalsegmentator_masks.py
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Optional

from tqdm import tqdm

try:
    from totalsegmentator.python_api import totalsegmentator
except Exception as exc:
    raise RuntimeError(
        "TotalSegmentator python API is required. Activate the environment with "
        "`totalsegmentator` installed."
    ) from exc


BASE_ROOT = Path("/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed")
DEFAULT_INPUT_ROOT = BASE_ROOT / "1initNifti"
DEFAULT_PREFIXES = ("AB_",)


def find_modality_nii(case_dir: Path, modality_dir: str, name_hint: str) -> Optional[Path]:
    modality_path = case_dir / modality_dir
    if not modality_path.is_dir():
        return None

    candidates = sorted(glob.glob(str(modality_path / f"*{name_hint}.nii*")))
    if not candidates:
        candidates = sorted(glob.glob(str(modality_path / "*.nii*")))
    return Path(candidates[0]) if candidates else None


def has_liver_mask(output_dir: Path) -> bool:
    return (output_dir / "liver.nii.gz").exists()


def has_fat_masks(output_dir: Path) -> bool:
    expected = ("subcutaneous_fat.nii.gz", "torso_fat.nii.gz", "skeletal_muscle.nii.gz")
    return all((output_dir / name).exists() for name in expected)


def run_totalseg_for_image(image_path: Path, modality_dir: str, device: str) -> None:
    output_dir = image_path.parent / "totalsegmentator_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    is_mr = modality_dir == "MR"
    liver_task = "total_mr" if is_mr else "total"
    fat_task = "tissue_types_mr" if is_mr else "tissue_types"

    if not has_liver_mask(output_dir):
        totalsegmentator(
            str(image_path),
            str(output_dir),
            task=liver_task,
            roi_subset=["liver"],
            fast=not is_mr,
            device=device,
        )

    if not has_fat_masks(output_dir):
        totalsegmentator(
            str(image_path),
            str(output_dir),
            task=fat_task,
            fast=False,
            device=device,
        )


def should_include_case(case_name: str, prefixes: tuple[str, ...]) -> bool:
    return any(case_name.startswith(prefix) for prefix in prefixes)


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 13 - run TotalSegmentator on init NIfTI cases")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--prefix", nargs="+", default=list(DEFAULT_PREFIXES))
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu", "mps"])
    args = parser.parse_args()

    input_root = Path(args.input_root)
    prefixes = tuple(args.prefix)

    case_dirs = [
        p
        for p in sorted(input_root.iterdir())
        if p.is_dir() and should_include_case(p.name, prefixes)
    ]

    print(f"Found {len(case_dirs)} case folders in {input_root}")
    for case_dir in tqdm(case_dirs):
        ct_path = find_modality_nii(case_dir, "CT_reg", "CT_reg")
        if ct_path is not None:
            run_totalseg_for_image(ct_path, "CT_reg", args.device)
        else:
            print(f"[SKIP] {case_dir.name}: no CT_reg NIfTI found")

        mr_path = find_modality_nii(case_dir, "MR", "MR")
        if mr_path is not None:
            run_totalseg_for_image(mr_path, "MR", args.device)
        else:
            print(f"[SKIP] {case_dir.name}: no MR NIfTI found")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
