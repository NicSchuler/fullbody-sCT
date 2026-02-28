#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_METRICS = ["MAE", "MSE", "PSNR", "SSIM"]
UNMASKED_METRICS = [f"{m}_unmasked" for m in BASE_METRICS]
MASKED_METRICS = [f"{m}_masked" for m in BASE_METRICS]
ALL_METRICS = [*UNMASKED_METRICS, *MASKED_METRICS]
# Usage:
#   python postprocessing/110compute_volume_metrics.py
#   python postprocessing/110compute_volume_metrics.py --root /path/to/100results

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

    # Backward compatibility: old files used bare metric names.
    if all(col in df.columns for col in BASE_METRICS):
        for metric in BASE_METRICS:
            df[f"{metric}_unmasked"] = pd.to_numeric(df[metric], errors="coerce")
            df[f"{metric}_masked"] = pd.to_numeric(df[metric], errors="coerce")
    else:
        for col in ALL_METRICS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_metrics = [col for col in ALL_METRICS if col not in df.columns]
    if missing_metrics:
        raise ValueError(f"Missing expected metric columns: {missing_metrics}")

    for col in ALL_METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "mask_voxels" not in df.columns:
        df["mask_voxels"] = 0

    df["mask_voxels"] = pd.to_numeric(df["mask_voxels"], errors="coerce").fillna(0)
    return df


def compute_volume_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["volume_id"] = df["file_name"].astype(str).map(_volume_id_from_name)

    def _masked_weighted_mean(values: pd.Series, weights: pd.Series) -> float:
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        if not valid.any():
            return np.nan
        return float(np.average(values[valid], weights=weights[valid]))

    def _volume_metrics(sub_df: pd.DataFrame) -> pd.Series:
        total_mask = float(sub_df["mask_voxels"].sum())
        out = {}
        for metric in BASE_METRICS:
            unmasked_col = f"{metric}_unmasked"
            masked_col = f"{metric}_masked"
            out[unmasked_col] = float(sub_df[unmasked_col].mean())
            out[masked_col] = _masked_weighted_mean(sub_df[masked_col], sub_df["mask_voxels"])
        out["mask_voxels"] = int(total_mask)
        out["slice_count"] = len(sub_df)
        return pd.Series(out)

    grouped = df[["volume_id", "mask_voxels", *ALL_METRICS]].groupby("volume_id")
    try:
        vol_df = grouped.apply(_volume_metrics, include_groups=False)
    except TypeError:
        vol_df = grouped.apply(_volume_metrics)

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
