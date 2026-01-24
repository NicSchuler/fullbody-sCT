#!/usr/bin/env python3
"""
Materialize dataset folder structure from a precomputed manifest.

This script is intentionally minimal: it does NOT compute splits.
It reads a CSV manifest with columns at least: `split`, `patient_token`.
It then copies 2D slice files into the folder layout expected by models:

  materialized_splits/
    pix2pix|cyclegan/
      train|val|test/
        A/
        B/

Sources are the outputs of 40slice_creator.py:
  - pix2pix:  <slices_root>/pix2pix_2d/full/{A,B}
  - cyclegan: <slices_root>/model_2d/full/{A,B}

Usage:
    # With normalization method (auto-configures paths):
    python 50_split_folderstructure.py 32p99

    # Or specify different method:
    python 50_split_folderstructure.py 31baseline
    python 50_split_folderstructure.py 33nyul
    python 50_split_folderstructure.py 34npeaks

    # Or override paths manually:
    python 50_split_folderstructure.py \
        --slices-root /.../32p99/5slices \
        --out-dir     /.../32p99/6materialized_splits \
        --modes both --include-ext .nii

Notes:
  - Patient tokens are inferred from filenames using simple regex patterns
    compatible with SynthRAD2023/2025 ids (e.g., 1ABA005, 1BA123).
  - Only files whose inferred token appears in the split's token set are copied.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from tqdm import tqdm

from shutil import copy2


# Regex: extract a patient token like 1ABA005, 1HND012, 1BA123, 1PC045
RE_TOKEN = re.compile(r"1(?:AB|HN|TH|B|P)?[A-D]?[0-9]{3}")


def extract_token(text: str) -> Optional[str]:
    m = RE_TOKEN.search(text)
    return m.group(0) if m else None


def load_manifest(path: Path) -> Dict[str, Set[str]]:
    """Return mapping: split -> set(patient_tokens)."""
    splits: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if "split" not in reader.fieldnames or "patient_token" not in reader.fieldnames:
            raise ValueError("Manifest must contain columns: split, patient_token")
        for row in reader:
            split = row["split"].strip().lower()
            token = row["patient_token"].strip()
            if not split or not token:
                continue
            if split not in splits:
                # allow custom split names but keep standard keys if present
                splits.setdefault(split, set())
            splits[split].add(token)
    # prune empty custom keys if any
    return {k: v for k, v in splits.items() if len(v) > 0}


def list_files(root: Path, include_ext: Sequence[str]) -> Iterable[Path]:
    exts = tuple(e.lower() for e in include_ext)
    if not root.exists():
        return []
    for p in root.iterdir() if root.is_dir() else [root]:
        if p.is_file() and str(p).lower().endswith(exts):
            yield p
    if root.is_dir():
        for p in root.rglob("*"):
            if p.is_file() and str(p).lower().endswith(exts):
                yield p


def index_by_token(files: Iterable[Path]) -> Dict[str, List[Path]]:
    idx: Dict[str, List[Path]] = {}
    for p in files:
        tok = extract_token(p.name) or extract_token(str(p))
        if not tok:
            continue
        idx.setdefault(tok, []).append(p)
    return idx


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def materialize_mode(
    mode: str,
    splits_tokens: Dict[str, Set[str]],
    slices_root: Path,
    out_root: Path,
    include_ext: Sequence[str],
) -> None:
    if mode == "pix2pix":
        srcA = slices_root / "pix2pix_2d" / "full" / "A"
        srcB = slices_root / "pix2pix_2d" / "full" / "B"
        dst_base = out_root / "pix2pix"
    elif mode == "cyclegan":
        srcA = slices_root / "model_2d" / "full" / "A"
        srcB = slices_root / "model_2d" / "full" / "B"
        dst_base = out_root / "cyclegan"
    else:
        raise ValueError("mode must be 'pix2pix' or 'cyclegan'")

    idxA = index_by_token(list_files(srcA, include_ext))
    idxB = index_by_token(list_files(srcB, include_ext))

    for split, tokens in tqdm(splits_tokens.items()):
        folderA = "A"
        folderB = "B"
        if mode == "cyclegan": ## for cycleGan we want to subfolders to be named like --> trainA, trainB, testA, testB etc.
            folderA = f"{split}A"
            folderB = f"{split}B"
        dstA = dst_base / split / folderA
        dstB = dst_base / split / folderB
        ensure_dir(dstA)
        ensure_dir(dstB)

        nA = nB = 0
        for t in tqdm(tokens):
            for src in idxA.get(t, []):
                copy2(src, dstA / src.name)
                nA += 1
            for src in idxB.get(t, []):
                copy2(src, dstB / src.name)
                nB += 1
        print(f"[ok] {mode}:{split} -> copied A:{nA} B:{nB}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Materialize split folder structure from manifest (no splitting)."
    )
    DEFAULT_BASE = Path("/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed")
    p.add_argument(
        "normalization_method",
        nargs="?",
        default="32p99",
        choices=["31baseline", "32p99", "33nyul", "34npeaks"],
        help="Normalization method (default: 32p99). Automatically configures slices-root and out-dir.",
    )
    p.add_argument(
        "--slices-root",
        default=None,
        help="Root produced by 40slice_creator.py (auto-configured from normalization_method if not provided)",
    )
    p.add_argument(
        "--manifest",
        default=str(DEFAULT_BASE / "splits_manifest.csv"),
        help="CSV with columns: split, patient_token",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Destination root for materialized_splits (auto-configured from normalization_method if not provided)",
    )
    p.add_argument(
        "--modes",
        choices=["pix2pix", "cyclegan", "both"],
        default="both",
        help="Which folder layout(s) to create",
    )
    p.add_argument(
        "--include-ext", nargs="+", default=[".nii"], help="Extensions to include"
    )
    
    args = p.parse_args(argv)
    
    # Auto-configure paths based on normalization method if not explicitly provided
    if args.slices_root is None:
        args.slices_root = str(DEFAULT_BASE / "experiment2" / args.normalization_method / "5slices")
    if args.out_dir is None:
        args.out_dir = str(DEFAULT_BASE / "experiment2" / args.normalization_method / "6materialized_splits")
    
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    slices_root = Path(args.slices_root)
    manifest = Path(args.manifest)
    out_root = Path(args.out_dir)

    print("=" * 60)
    print(f"Normalization method: {args.normalization_method}")
    print(f"Slices root: {slices_root}")
    print(f"Manifest:    {manifest}")
    print(f"Output root: {out_root}")
    print(f"Modes:       {args.modes}")
    print("=" * 60)

    if not slices_root.exists():
        print(f"[error] slices-root not found: {slices_root}")
        return 2
    if not manifest.exists():
        print(f"[error] manifest not found: {manifest}")
        return 2

    splits_tokens = load_manifest(manifest)
    if not splits_tokens:
        print("[error] Manifest contains no tokens.")
        return 2

    modes: List[str]
    if args.modes == "both":
        modes = ["pix2pix", "cyclegan"]
    else:
        modes = [args.modes]

    for m in modes:
        materialize_mode(m, splits_tokens, slices_root, out_root, args.include_ext)

    print(f"[done] Materialized splits at {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())