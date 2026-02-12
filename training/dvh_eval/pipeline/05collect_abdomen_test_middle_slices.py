#!/usr/bin/env python3
"""Collect one middle test slice per abdomen patient.

This script reads already-sliced NIfTI files from the materialized split folder,
groups by patient, picks the middle slice index, and copies it to an output folder.

Default source path targets abdomen test CT slices:
  .../7materialized_splits_31baselineBodyRegion/AB/pix2pix/test/B
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_PREPROC_ROOT = Path("/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed")
DEFAULT_INPUT_DIR = DEFAULT_PREPROC_ROOT / "7materialized_splits_31baselineBodyRegion" / "AB" / "pix2pix" / "test" / "B"
DEFAULT_OUTPUT_DIR = DEFAULT_PREPROC_ROOT / "11dvhEvalCases" / "abdomen_test_middle_slices"


def parse_patient_and_slice(stem: str) -> Optional[Tuple[str, int]]:
    """Parse patient id and slice index from file stem.

    Supported formats:
      - <patient>-<slice>
      - <patient>_<slice>
    where <slice> is an integer.
    """
    m = re.match(r"^(?P<patient>.+)-(?P<slice>\d+)$", stem)
    if m:
        return m.group("patient"), int(m.group("slice"))

    m = re.match(r"^(?P<patient>.+)_(?P<slice>\d+)$", stem)
    if m:
        return m.group("patient"), int(m.group("slice"))

    return None


def collect_nifti_files(input_dir: Path) -> List[Path]:
    files = sorted(input_dir.glob("*.nii")) + sorted(input_dir.glob("*.nii.gz"))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect middle abdomen test slice per patient")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Folder with abdomen test slices")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output folder for selected middle slices")
    parser.add_argument("--symlink", action="store_true", help="Create symlinks instead of copying files")
    parser.add_argument("--clean", action="store_true", help="Delete output dir before writing")
    parser.add_argument("--manifest-name", default="middle_slices_manifest.csv")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input dir not found: {input_dir}")

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, List[Tuple[int, Path]]] = {}
    for fpath in collect_nifti_files(input_dir):
        name = fpath.name
        stem = name[:-7] if name.endswith(".nii.gz") else fpath.stem
        parsed = parse_patient_and_slice(stem)
        if parsed is None:
            continue
        patient, slice_idx = parsed
        grouped.setdefault(patient, []).append((slice_idx, fpath))

    if not grouped:
        raise RuntimeError(f"No parseable NIfTI test slices found in {input_dir}")

    rows = []
    for patient, entries in sorted(grouped.items()):
        entries_sorted = sorted(entries, key=lambda x: x[0])
        mid = len(entries_sorted) // 2
        slice_idx, src = entries_sorted[mid]
        dst = output_dir / src.name

        if not args.symlink:
            shutil.copy2(src, dst)
        else:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)

        rows.append(
            {
                "patient": patient,
                "num_slices": len(entries_sorted),
                "selected_slice_idx": slice_idx,
                "source": str(src),
                "output": str(dst),
            }
        )
        print(f"[OK] {patient}: selected slice {slice_idx} ({src.name})")

    manifest_path = output_dir / args.manifest_name
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient", "num_slices", "selected_slice_idx", "source", "output"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Patients: {len(rows)}")
    print(f"Output: {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
