#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_ROOT = Path(
    "/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results"
)


def cleanup_file(csv_path: Path, backup: bool, dry_run: bool) -> tuple[int, int]:
    df = pd.read_csv(csv_path)
    total_rows = len(df)

    if "mask_voxels" not in df.columns:
        return total_rows, 0

    mask_vals = pd.to_numeric(df["mask_voxels"], errors="coerce")
    keep = ~(mask_vals == 0)
    removed_rows = int((~keep).sum())
    if removed_rows == 0:
        return total_rows, 0

    cleaned = df.loc[keep].copy()

    if dry_run:
        return total_rows, removed_rows

    if backup:
        backup_path = csv_path.with_suffix(csv_path.suffix + ".bak")
        csv_path.replace(backup_path)
        cleaned.to_csv(csv_path, index=False)
    else:
        cleaned.to_csv(csv_path, index=False)

    return total_rows, removed_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove rows with mask_voxels == 0 from all test_metrics_over_all.csv files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root folder to search recursively (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would change; do not modify files.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak backups before overwriting.",
    )
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root}")

    csv_paths = sorted(root.rglob("test_metrics_over_all.csv"))
    if not csv_paths:
        print(f"No test_metrics_over_all.csv files found under {root}")
        return

    print(f"Found {len(csv_paths)} files under {root}")
    total_removed = 0
    changed_files = 0

    for csv_path in csv_paths:
        total_rows, removed_rows = cleanup_file(
            csv_path=csv_path, backup=not args.no_backup, dry_run=args.dry_run
        )
        status = "unchanged"
        if removed_rows > 0:
            status = "would update" if args.dry_run else "updated"
            changed_files += 1
            total_removed += removed_rows
        print(
            f"{status:11} | removed={removed_rows:5d} | rows={total_rows:5d} | {csv_path}"
        )

    mode = "Would remove" if args.dry_run else "Removed"
    print(
        f"{mode} {total_removed} rows with mask_voxels == 0 across {changed_files} files."
    )
    if not args.dry_run and not args.no_backup:
        print("Backups written as '*.csv.bak'.")


if __name__ == "__main__":
    main()
