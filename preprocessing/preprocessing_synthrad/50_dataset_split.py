#!/usr/bin/env python3
"""
Dataset split utility for SynthRAD 2023 and 2025 datasets.

Goals:
- Build a patient-level manifest from directories of 3D volumes or 2D slices
- Parse patient id, body region, center from filenames/paths robustly
- Perform stratified splits (Train/Val/Test) preserving ratios across body region and center
- Save outputs as a CSV manifest with split labels and (optional) per-split file lists

Assumptions:
- Patient IDs follow patterns (core token anywhere in filename/path):
  * 2025: 1(AB|HN|TH)(A|B|C|D)\d{3} e.g. 1ABA005, 1HND012, 1THC123
  * 2023: 1(B|P)(A|B|C)\d{3} e.g. 1BA123, 1PC045
- Slice files may append extra suffixes (e.g., slice number), which we ignore by extracting the core token via regex.
- Splits must be at patient level to avoid leakage between train/val/test.

Usage examples:
    1) Build manifest and split lists only (no file moves):
        python 50_dataset_split.py \
            --input-root /local/scratch/datasets/FullbodySCT/SynthRAD2025/task1_backup/5slicesOutputForModels \
            --ratios 0.7 0.15 0.15 \
            --seed 42 \
            --out-manifest /tmp/splits_manifest.csv \
            --out-list-dir /tmp/split_lists

    2) Also materialize split folders for pix2pix and cyclegan (copy files):
        python 50_dataset_split.py \
            --input-root /local/scratch/datasets/FullbodySCT/SynthRAD2025/task1_backup/5slicesOutputForModels \
            --ratios 0.7 0.15 0.15 \
            --seed 42 \
            --out-manifest /tmp/splits_manifest.csv \
            --out-list-dir /tmp/split_lists \
            --materialize-dir /tmp/materialized_splits \
            --mode both \
            --link-mode copy

"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set

try:
    from sklearn.model_selection import train_test_split
except Exception as e:  # pragma: no cover
    train_test_split = None


# Regex patterns for patient tokens (core id embedded in filenames or dirnames)
RE_2025 = re.compile(r"(?P<prefix>(?P<body>AB|HN|TH)_)?(?P<token>1(?P<body2>AB|HN|TH)?(?P<center>[A-D])(?P<num>\d{3}))")
RE_2023 = re.compile(r"(?P<prefix>(?P<body>B|P)_)?(?P<token>1(?P<body2>B|P)?(?P<center>[A-C])(?P<num>\d{3}))")


@dataclass(frozen=True)
class PatientKey:
    token: str  # full token string like 1ABA005 or 1BA123
    body: str   # body region code (AB/HN/TH or B/P)
    center: str # center code (A-D or A-C)
    year: int   # 2025 or 2023

    @property
    def label(self) -> str:
        """Composite label for stratification, combining body and center."""
        return f"{self.body}_{self.center}"


@dataclass
class PatientRecord:
    key: PatientKey
    files: List[str]

    def to_row(self) -> Dict[str, str]:
        return {
            "patient_token": self.key.token,
            "year": str(self.key.year),
            "body": self.key.body,
            "center": self.key.center,
            "label": self.key.label,
            "n_files": str(len(self.files)),
            "example_path": self.files[0] if self.files else "",
        }


def find_patient_token(path: str) -> Optional[PatientKey]:
    """Extract patient token/body/center/year from a path or filename.

    Returns None if no known pattern is found.
    """
    name = os.path.basename(path)

    m = RE_2025.search(path)
    if m:
        token = m.group("token")
        # prefer explicit body from prefix, else body2 inside token
        body = m.group("body") or m.group("body2") or "UNK"
        center = m.group("center")
        return PatientKey(token=token, body=body, center=center, year=2025)

    m = RE_2023.search(path)
    if m:
        token = m.group("token")
        body = m.group("body") or m.group("body2") or "UNK"
        center = m.group("center")
        return PatientKey(token=token, body=body, center=center, year=2023)

    # Try also in the directory name hierarchy
    parts = Path(path).parts
    for p in reversed(parts):
        m = RE_2025.search(p)
        if m:
            token = m.group("token")
            body = m.group("body") or m.group("body2") or "UNK"
            center = m.group("center")
            return PatientKey(token=token, body=body, center=center, year=2025)
        m = RE_2023.search(p)
        if m:
            token = m.group("token")
            body = m.group("body") or m.group("body2") or "UNK"
            center = m.group("center")
            return PatientKey(token=token, body=body, center=center, year=2023)

    return None


def iter_files(roots: Sequence[str], include_ext: Sequence[str]) -> Iterable[str]:
    exts = tuple(e.lower() for e in include_ext)
    for root in roots:
        root_p = Path(root)
        if not root_p.exists():
            continue
        if root_p.is_file():
            if str(root_p).lower().endswith(exts):
                yield str(root_p)
            continue
        for p in root_p.rglob("*"):
            if p.is_file() and str(p).lower().endswith(exts):
                yield str(p)


def build_patient_records(roots: Sequence[str], include_ext: Sequence[str]) -> Dict[PatientKey, PatientRecord]:
    patients: Dict[PatientKey, PatientRecord] = {}
    missing: List[str] = []

    for f in iter_files(roots, include_ext):
        key = find_patient_token(f)
        if key is None:
            missing.append(f)
            continue
        if key not in patients:
            patients[key] = PatientRecord(key=key, files=[f])
        else:
            patients[key].files.append(f)

    if missing:
        print(f"[warn] {len(missing)} files had no recognizable patient token. First few:\n  - " + "\n  - ".join(missing[:10]))

    return patients


def stratified_split(
    items: List[PatientRecord],
    ratios: Tuple[float, float, float],
    seed: int,
) -> Tuple[List[PatientRecord], List[PatientRecord], List[PatientRecord]]:
    """Split items into train/val/test with stratification by composite label.

    Strategy: two-step stratified splitting using sklearn's train_test_split.
    """
    if train_test_split is None:
        raise RuntimeError("scikit-learn is required. Please install scikit-learn >= 1.1.")

    assert len(items) > 0, "No items to split"
    r_train, r_val, r_test = ratios
    if not (abs(r_train + r_val + r_test - 1.0) < 1e-6):
        raise ValueError("Ratios must sum to 1.0")

    X = list(range(len(items)))
    y = [rec.key.label for rec in items]

    # First: split off test
    test_size = r_test
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Then: split remaining into train/val
    val_size_relative = r_val / (r_train + r_val)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_size_relative, random_state=seed, stratify=y_tmp
    )

    train = [items[i] for i in X_train]
    val = [items[i] for i in X_val]
    test = [items[i] for i in X_test]
    return train, val, test


def write_manifest(
    path: str,
    splits: Dict[str, List[PatientRecord]],
):
    fields = ["split", "patient_token", "year", "body", "center", "label", "n_files", "example_path"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for split, items in splits.items():
            for rec in items:
                row = rec.to_row()
                row["split"] = split
                w.writerow(row)
    print(f"[ok] Wrote manifest: {path}")


def write_split_lists(out_dir: Optional[str], splits: Dict[str, List[PatientRecord]]):
    if not out_dir:
        return
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for split, items in splits.items():
        list_path = Path(out_dir) / f"{split}_files.txt"
        with open(list_path, "w") as f:
            for rec in items:
                for p in rec.files:
                    f.write(p + "\n")
        print(f"[ok] Wrote {split} file list: {list_path}")


def summarize_counts(items: List[PatientRecord], title: str):
    cnt = Counter(rec.key.label for rec in items)
    total = len(items)
    print(f"\n{title}: {total} patients")
    for label, n in sorted(cnt.items()):
        print(f"  {label}: {n}")


def default_slice_roots(input_root: str, mode: str) -> Dict[str, Dict[str, str]]:
    base = Path(input_root)
    return {
        "pix2pix": {
            "A": str(base / f"pix2pix_{mode}" / "full" / "A"),
            "B": str(base / f"pix2pix_{mode}" / "full" / "B"),
        },
        "cyclegan": {
            "A": str(base / f"model_{mode}" / "full" / "A"),
            "B": str(base / f"model_{mode}" / "full" / "B"),
        },
    }


def slice_pairs_for_pix2pix(dir_A: str, dir_B: str, include_ext: Sequence[str]) -> Dict[PatientKey, List[Tuple[str, str]]]:
    map_A = scan_slice_dir(dir_A, include_ext)
    map_B = scan_slice_dir(dir_B, include_ext)
    pairs: Dict[PatientKey, List[Tuple[str, str]]] = {}
    for key in set(map_A.keys()) | set(map_B.keys()):
        common = sorted(map_A.get(key, set()) & map_B.get(key, set()))
        if not common:
            continue
        lst: List[Tuple[str, str]] = []
        for fn in common:
            lst.append((os.path.join(dir_A, fn), os.path.join(dir_B, fn)))
        pairs[key] = lst
    return pairs


def scan_slice_dir(root: str, include_ext: Sequence[str]) -> Dict[PatientKey, Set[str]]:
    result: Dict[PatientKey, Set[str]] = {}
    for p in iter_files([root], include_ext):
        key = find_patient_token(p)
        if key is None:
            continue
        fn = os.path.basename(p)
        result.setdefault(key, set()).add(fn)
    return result


def slice_lists_for_unpaired(dir_A: str, dir_B: str, include_ext: Sequence[str]) -> Dict[str, Dict[PatientKey, List[str]]]:
    def scan(root: str) -> Dict[PatientKey, List[str]]:
        out: Dict[PatientKey, List[str]] = {}
        for p in iter_files([root], include_ext):
            key = find_patient_token(p)
            if key is None:
                continue
            out.setdefault(key, []).append(p)
        for k in out:
            out[k].sort()
        return out
    return {"A": scan(dir_A), "B": scan(dir_B)}


def copy_file(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    from shutil import copy2
    copy2(src, dst)


def materialize_splits(materialize_dir: str, splits: Dict[str, List[PatientRecord]], include_ext: Sequence[str], slice_roots: Dict[str, Dict[str, str]], mode: str = "both"):
    base = Path(materialize_dir)
    do_paired = mode in ("paired", "both")
    do_unpaired = mode in ("unpaired", "both")
    if do_paired:
        pixA = slice_roots["pix2pix"]["A"]
        pixB = slice_roots["pix2pix"]["B"]
        pairs = slice_pairs_for_pix2pix(pixA, pixB, include_ext)
        for split_name, items in splits.items():
            out_A = base / "pix2pix" / split_name / "A"
            out_B = base / "pix2pix" / split_name / "B"
            for rec in items:
                for srcA, srcB in pairs.get(rec.key, []):
                    copy_file(srcA, str(out_A / os.path.basename(srcA)))
                    copy_file(srcB, str(out_B / os.path.basename(srcB)))
    if do_unpaired:
        cycA = slice_roots["cyclegan"]["A"]
        cycB = slice_roots["cyclegan"]["B"]
        cyc_lists = slice_lists_for_unpaired(cycA, cycB, include_ext)
        for split_name, items in splits.items():
            out_A = base / "cyclegan" / split_name / "A"
            out_B = base / "cyclegan" / split_name / "B"
            for rec in items:
                for srcA in cyc_lists['A'].get(rec.key, []):
                    copy_file(srcA, str(out_A / os.path.basename(srcA)))
                for srcB in cyc_lists['B'].get(rec.key, []):
                    copy_file(srcB, str(out_B / os.path.basename(srcB)))

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stratified dataset splitter for SynthRAD with optional slice materialization")
    p.add_argument("--input-root", required=True, help="Root produced by 40slice_creator.py")
    p.add_argument("--nn-mode", choices=["2d", "3d"], default="2d", help="Subfolder mode name used by 40slice_creator")
    p.add_argument("--include-ext", nargs="+", default=[".nii", ".nii.gz"], help="Slice file extensions to include")
    p.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"), help="Split ratios that sum to 1.0")
    p.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    p.add_argument("--out-manifest", required=True, help="Path to write CSV manifest with split column")
    p.add_argument("--out-list-dir", default=None, help="Directory to write per-split file lists (optional)")
    p.add_argument("--materialize-dir", default=None, help="If set, copy slices into split folders under this directory")
    p.add_argument("--mode", choices=["paired", "unpaired", "both"], default="both", help="What to materialize if --materialize-dir is set")
    p.add_argument("--filter-year", type=int, choices=[2023, 2025], default=None, help="If set, include only this year")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    slice_roots = default_slice_roots(args.input_root, args.nn_mode)
    candidate_roots = [
        slice_roots["pix2pix"]["A"], slice_roots["pix2pix"]["B"],
        slice_roots["cyclegan"]["A"], slice_roots["cyclegan"]["B"],
    ]
    patients = build_patient_records(candidate_roots, args.include_ext)
    items = list(patients.values())
    if args.filter_year is not None:
        items = [r for r in items if r.key.year == args.filter_year]
    if len(items) == 0:
        print("[error] No patients found with the given roots/extensions.")
        return 2
    train, val, test = stratified_split(items, tuple(args.ratios), args.seed)
    summarize_counts(items, "ALL")
    summarize_counts(train, "TRAIN")
    summarize_counts(val, "VAL")
    summarize_counts(test, "TEST")
    splits = {"train": train, "val": val, "test": test}
    write_manifest(args.out_manifest, splits)
    write_split_lists(args.out_list_dir, splits)
    if args.materialize_dir:
        materialize_splits(args.materialize_dir, splits, args.include_ext, slice_roots, mode=args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
