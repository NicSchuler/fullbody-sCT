#!/usr/bin/env python3
"""Step 10: prepare shared CT/reference-grid folders for DHV eval export.

This script creates a manifest-free, download-friendly structure under:
  <export-root>/test_patients_shared/<patient>/
    CT/<patient>_CT.nii.gz
    reference_grid.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from _pipeline_common import find_ct_nifti, load_grid_info

DEFAULT_PREPROC_ROOT = Path("/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed")
DEFAULT_CT_ROOT = DEFAULT_PREPROC_ROOT / "2resampledNifti_reconstructed_dims"
DEFAULT_OUTPUT_BASE = DEFAULT_PREPROC_ROOT / "11dvhEvalCases"


def read_patients(patients: list[str] | None, patients_file: Path | None, ct_root: Path) -> list[str]:
    if patients:
        return list(dict.fromkeys(patients))

    if patients_file:
        lines = [ln.strip() for ln in patients_file.read_text().splitlines()]
        return [ln for ln in lines if ln and not ln.startswith("#")]

    return sorted([p.name for p in ct_root.iterdir() if p.is_dir()]) if ct_root.is_dir() else []


def copy_ct(ct_src: Path, ct_dst: Path, force: bool) -> None:
    if ct_dst.exists() and not force:
        return
    ct_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ct_src, ct_dst)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 10 - prepare shared CT + reference_grid folders")
    p.add_argument("--ct-root", type=Path, default=DEFAULT_CT_ROOT)
    p.add_argument("--export-root", type=Path, default=DEFAULT_OUTPUT_BASE)
    p.add_argument("--ct-subdir", default="CT_reg", help="CT subfolder under patient folder")
    p.add_argument("--patients", nargs="*", default=None)
    p.add_argument("--patients-file", type=Path, default=None)
    p.add_argument("--force", action="store_true", help="Overwrite existing CT/reference_grid files")
    p.add_argument(
        "--clean-missing",
        action="store_true",
        help="Remove patient folders in test_patients_shared that are not in the selected patient list",
    )
    return p


def main() -> None:
    args = make_parser().parse_args()
    ct_root = args.ct_root.resolve()
    export_root = args.export_root.resolve()

    patients = read_patients(args.patients, args.patients_file, ct_root)
    if not patients:
        raise RuntimeError("No patients found. Provide --patients/--patients-file or verify --ct-root.")

    shared_root = export_root / "test_patients_shared"
    shared_root.mkdir(parents=True, exist_ok=True)

    selected = set(patients)
    if args.clean_missing and shared_root.exists():
        for child in shared_root.iterdir():
            if child.is_dir() and child.name not in selected:
                shutil.rmtree(child)
                print(f"[CLEAN] removed {child}")

    ok = 0
    failed = 0

    for patient in patients:
        try:
            ct_nifti = find_ct_nifti(ct_root, patient, args.ct_subdir)
            grid = load_grid_info(ct_nifti)

            patient_dir = shared_root / patient
            ct_dst = patient_dir / "CT" / f"{patient}_CT.nii.gz"
            grid_dst = patient_dir / "reference_grid.json"

            copy_ct(ct_nifti, ct_dst, args.force)

            if args.force or not grid_dst.exists():
                grid_dst.parent.mkdir(parents=True, exist_ok=True)
                grid_dst.write_text(json.dumps(grid, indent=2))

            ok += 1
            print(f"[OK] {patient} | CT -> {ct_dst}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {patient}: {exc}")

    print("\nDone.")
    print(f"Prepared: {ok} | Failed: {failed}")
    print(f"Shared root: {shared_root}")


if __name__ == "__main__":
    main()
