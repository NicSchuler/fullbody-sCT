#!/usr/bin/env python3
"""Step 25: populate per-model sCT folders for download (manifest-free).

Expected final layout under <export-root>:
  test_patients_shared/<patient>/CT
  test_patients_shared/<patient>/TS_CT
  test_patients_shared/<patient>/reference_grid.json

  <model_name>/<patient>/<patient>_sCT.nii.gz
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from _pipeline_common import find_sct_nifti

DEFAULT_PREPROC_ROOT = Path("/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed")
DEFAULT_SCT_BASE = DEFAULT_PREPROC_ROOT / "9latestTestImages"
DEFAULT_EXPORT_ROOT = DEFAULT_PREPROC_ROOT / "11dvhEvalCases"


def read_patients(patients: list[str] | None, patients_file: Path | None, shared_root: Path) -> list[str]:
    if patients:
        return list(dict.fromkeys(patients))

    if patients_file:
        lines = [ln.strip() for ln in patients_file.read_text().splitlines()]
        return [ln for ln in lines if ln and not ln.startswith("#")]

    return sorted([p.name for p in shared_root.iterdir() if p.is_dir()]) if shared_root.is_dir() else []


def discover_models(sct_base: Path, epoch: int) -> list[str]:
    models = []
    if not sct_base.is_dir():
        return models
    for model_dir in sorted([p for p in sct_base.iterdir() if p.is_dir()]):
        recon_dir = model_dir / f"test_{epoch}" / "reconstruction"
        if recon_dir.is_dir():
            models.append(model_dir.name)
    return models


def copy_if_exists(src: Path, dst: Path, force: bool) -> bool:
    if not src.exists():
        return False
    if dst.exists() and not force:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 25 - copy per-model sCT into 11dvhEvalCases")
    p.add_argument("--sct-base", type=Path, default=DEFAULT_SCT_BASE)
    p.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    model_group = p.add_mutually_exclusive_group()
    model_group.add_argument("--model-name", default=None, help="Single model folder under --sct-base")
    model_group.add_argument("--model-names", nargs="*", default=None, help="Model folders under --sct-base")
    p.add_argument("--epoch", type=int, default=50, help="Evaluation epoch number used in test_<epoch>")
    p.add_argument("--patients", nargs="*", default=None)
    p.add_argument("--patients-file", type=Path, default=None)
    p.add_argument("--force", action="store_true", help="Overwrite existing sCT files")
    return p


def main() -> None:
    args = make_parser().parse_args()
    sct_base = args.sct_base.resolve()
    export_root = args.export_root.resolve()
    shared_root = export_root / "test_patients_shared"

    patients = read_patients(args.patients, args.patients_file, shared_root)
    if not patients:
        raise RuntimeError(f"No patients found under {shared_root}. Run step 10 first.")

    if args.model_name:
        model_names = [args.model_name]
    elif args.model_names:
        model_names = args.model_names
    else:
        model_names = discover_models(sct_base, args.epoch)
    if not model_names:
        raise RuntimeError("No model folders found. Provide --model-names or verify --sct-base.")

    total_ok = 0
    total_missing = 0

    for model_name in model_names:
        sct_root = sct_base / model_name / f"test_{args.epoch}" / "reconstruction"
        if not sct_root.is_dir():
            print(f"[WARN] Model reconstruction not found, skipping: {sct_root}")
            continue

        print(f"\nModel: {model_name}")
        model_ok = 0

        for patient in patients:
            try:
                sct_src = find_sct_nifti(sct_root, patient)
            except Exception:
                total_missing += 1
                print(f"[MISS] {patient} | no sCT under {sct_root}")
                continue

            sct_dst = export_root / model_name / patient / f"{patient}_sCT.nii.gz"
            copied = copy_if_exists(sct_src, sct_dst, args.force)
            if copied:
                model_ok += 1
                total_ok += 1
                print(f"[OK]   {patient} -> {sct_dst}")
            else:
                total_missing += 1
                print(f"[MISS] {patient} | source missing: {sct_src}")

        print(f"Model summary: {model_ok}/{len(patients)}")

    print("\nDone.")
    print(f"Copied sCT files: {total_ok}")
    print(f"Missing sCT files: {total_missing}")


if __name__ == "__main__":
    main()
