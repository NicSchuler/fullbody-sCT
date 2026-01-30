#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = ["MAE", "MSE", "PSNR", "SSIM"]
# Usage:
#   python preprocessing/preprocessing_synthrad/110compute_volume_metrics.py
#   python preprocessing/preprocessing_synthrad/110compute_volume_metrics.py --root /path/to/100results

DEFAULT_ROOT = Path(
    "/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results"
)


def _volume_id_from_name(name: str) -> str:
    base = name
    if base.endswith(".nii.gz"):
        base = base[:-7]
    elif base.endswith(".nii"):
        base = base[:-4]
    parts = base.rsplit("-", 1)
    return parts[0]


def load_slice_metrics(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "file_name" not in df.columns:
        df = df.iloc[:, 1:]

    if "file_name" not in df.columns:
        raise ValueError("Could not find 'file_name' column after dropping index column.")

    if df.shape[1] == 5:
        df.columns = ["file_name", "MAE", "MSE", "PSNR", "SSIM"]
    elif df.shape[1] == 6:
        df.columns = ["file_name", "MAE", "MSE", "PSNR", "SSIM", "mask_voxels"]

    for col in METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "mask_voxels" not in df.columns:
        raise ValueError("mask_voxels column missing; cannot compute volume metrics.")

    df["mask_voxels"] = pd.to_numeric(df["mask_voxels"], errors="coerce").fillna(0)
    return df


def compute_volume_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["volume_id"] = df["file_name"].astype(str).map(_volume_id_from_name)

    def _weighted_metrics(sub_df: pd.DataFrame) -> pd.Series:
        total = sub_df["mask_voxels"].sum()
        if total == 0:
            return pd.Series({**{m: None for m in METRICS}, "mask_voxels": 0, "slice_count": len(sub_df)})
        weights = sub_df["mask_voxels"] / total
        wmean = sub_df[METRICS].mul(weights, axis=0).sum()
        out = {m: wmean[m] for m in METRICS}
        out["mask_voxels"] = total
        out["slice_count"] = len(sub_df)
        return pd.Series(out)

    grouped = df[["volume_id", "mask_voxels", *METRICS]].groupby("volume_id")
    try:
        vol_df = grouped.apply(_weighted_metrics, include_groups=False)
    except TypeError:
        vol_df = grouped.apply(_weighted_metrics)

    vol_df = vol_df.reset_index()
    return vol_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-volume metrics for every test_metrics_over_all.csv under a root folder."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root folder to search (default: %(default)s)",
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
    for csv_path in csv_paths:
        df = load_slice_metrics(csv_path)
        vol_df = compute_volume_metrics(df)
        out_path = csv_path.parent / "test_metrics_over_volume.csv"
        vol_df.to_csv(out_path, index=False)
        print(f"Wrote volume metrics: {out_path}")


if __name__ == "__main__":
    main()
