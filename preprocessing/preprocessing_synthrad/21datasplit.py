#!/usr/bin/env python3
"""
Patient-level Train/Val/Test split AFTER resampling and BEFORE Nyul.

Outputs an Excel-friendly CSV manifest at:
  /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/splits_manifest.csv

Columns: split, patient_token, year, body, center, label, example_path, n_files

Stratification: by body_center to preserve distribution across sites/regions.
"""

import argparse
import csv
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Sequence, Tuple

try:
    from sklearn.model_selection import train_test_split
except Exception:
    train_test_split = None


BASE_ROOT = Path("/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed")
INPUT_ROOT = BASE_ROOT / "2resampledNifti"
OUT_MANIFEST = BASE_ROOT / "splits_manifest.csv"

RE_2025 = re.compile(r"(?P<prefix>(?P<body>AB|HN|TH)_)?(?P<token>1(?P<body2>AB|HN|TH)?(?P<center>[A-D])(?P<num>\d{3}))")
RE_2023 = re.compile(r"(?P<prefix>(?P<body>B|P)_)?(?P<token>1(?P<body2>B|P)?(?P<center>[A-C])(?P<num>\d{3}))")


def find_patient_key(case_id: str) -> Optional[Tuple[str, str, str, int]]:
    m = RE_2025.search(case_id)
    if m:
        token = m.group("token")
        body = m.group("body") or m.group("body2") or "UNK"
        center = m.group("center")
        return token, body, center, 2025
    m = RE_2023.search(case_id)
    if m:
        token = m.group("token")
        body = m.group("body") or m.group("body2") or "UNK"
        center = m.group("center")
        return token, body, center, 2023
    return None


def collect_patients(root: Path) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for case_dir in sorted(root.iterdir()):
        if not case_dir.is_dir():
            continue
        key = find_patient_key(case_dir.name)
        if not key:
            continue
        token, body, center, year = key
        items.append({
            "patient_token": token,
            "case_id": case_dir.name,
            "body": body,
            "center": center,
            "year": str(year),
            "label": f"{body}_{center}",
        })
    return items


def stratified_split(items: List[Dict[str, str]], ratios: Tuple[float, float, float], seed: int):
    if train_test_split is None:
        raise RuntimeError("scikit-learn is required. Please install scikit-learn >= 1.1.")
    assert items, "No patients found to split"
    r_train, r_val, r_test = ratios
    if abs(r_train + r_val + r_test - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    X = list(range(len(items)))
    y = [it["label"] for it in items]
    X_tmp, X_test, y_tmp, _ = train_test_split(X, y, test_size=r_test, random_state=seed, stratify=y)
    val_rel = r_val / (r_train + r_val)
    X_train, X_val, _, _ = train_test_split(X_tmp, y_tmp, test_size=val_rel, random_state=seed, stratify=y_tmp)
    train = [items[i] for i in X_train]
    val = [items[i] for i in X_val]
    test = [items[i] for i in X_test]
    return train, val, test


def write_manifest(path: Path, splits: Dict[str, List[Dict[str, str]]]):
    fields = ["split", "patient_token", "year", "body", "center", "label", "example_path", "n_files"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for split, lst in splits.items():
            for it in lst:
                w.writerow({
                    "split": split,
                    "patient_token": it["patient_token"],
                    "year": it["year"],
                    "body": it["body"],
                    "center": it["center"],
                    "label": it["label"],
                    "example_path": str((INPUT_ROOT / it["case_id"]).resolve()),
                    "n_files": "",
                })
    print(f"[ok] Wrote manifest: {path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build patient splits manifest before Nyul")
    p.add_argument("--input-root", default=str(INPUT_ROOT), help="Root of resampled cases (2resampledNifti)")
    p.add_argument("--out-manifest", default=str(OUT_MANIFEST), help="Output CSV path")
    p.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    items = collect_patients(Path(args.input_root))
    train, val, test = stratified_split(items, tuple(args.ratios), args.seed)
    splits = {"train": train, "val": val, "test": test}
    write_manifest(Path(args.out_manifest), splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())