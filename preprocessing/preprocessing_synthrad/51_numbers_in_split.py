#!/usr/bin/env python
"""
Aggregate patient counts from splits_manifest.csv.

Creates three aggregated tables with the number of unique patient_token,
with split values (train / val / test) as COLUMNS + percentage columns:

1) By split
   - 1 row ("total")
   - columns: train, val, test, train_pct, val_pct, test_pct

2) By [year, body]
   - rows: (year, body)
   - columns: year, body, train, val, test, train_pct, val_pct, test_pct

3) By [year, center]
   - rows: (year, center)
   - columns: year, center, train, val, test, train_pct, val_pct, test_pct

Outputs CSV files into an "aggregates" folder (created if it doesn't exist).
"""

from pathlib import Path
import pandas as pd

SPLITS = ["train", "val", "test"]


def load_manifest(csv_path):
    """Load the splits_manifest CSV and validate required columns."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError("Could not find CSV at: {}".format(csv_path))

    df = pd.read_csv(csv_path)

    expected_cols = {"split", "patient_token", "year", "body", "center"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError("CSV is missing required columns: {}".format(missing))

    return df


def add_split_percentages(df, split_cols):
    """
    Add percentage columns for each split in split_cols, per row.

    For each row:
        {split}_pct = 100 * {split} / (train + val + test)

    If the row total is 0, all percentages are set to 0.
    """
    totals = df[split_cols].sum(axis=1)

    for s in split_cols:
        pct_col = s + "_pct"
        df[pct_col] = 0.0  # default

        mask = totals > 0
        df.loc[mask, pct_col] = (
            df.loc[mask, s].astype(float) / totals[mask].astype(float) * 100.0
        )

    return df


def make_aggregates(df):
    """
    Build aggregated tables with split as columns plus percentage columns:

    - by split:          1 row "total", columns: train, val, test, *_pct
    - by [year, body]:   rows: (year, body), columns: train, val, test, *_pct
    - by [year, center]: rows: (year, center), columns: train, val, test, *_pct
    """

    # ---- 1) By split: 1 row "total" ----
    agg_split_series = (
        df.groupby("split")["patient_token"]
        .nunique()
        .reindex(SPLITS)      # ensure columns order and include missing splits
        .fillna(0)
        .astype(int)
    )

    # Make it a single-row DataFrame with index "total"
    agg_split = pd.DataFrame([agg_split_series.values], columns=SPLITS)
    agg_split.insert(0, "row", "total")

    # Add percentage columns
    agg_split = add_split_percentages(agg_split, SPLITS)

    # ---- 2) By [year, body] ----
    gb_year_body = (
        df.groupby(["year", "body", "split"])["patient_token"]
        .nunique()
        .reset_index()
    )

    agg_year_body = gb_year_body.pivot_table(
        index=["year", "body"],
        columns="split",
        values="patient_token",
        fill_value=0,
        aggfunc="sum",
    )

    # Ensure all split columns exist and in the right order
    for s in SPLITS:
        if s not in agg_year_body.columns:
            agg_year_body[s] = 0

    agg_year_body = agg_year_body[SPLITS].astype(int)

    # Bring index back to columns
    agg_year_body = agg_year_body.reset_index().sort_values(["year", "body"])

    # Add percentage columns
    agg_year_body = add_split_percentages(agg_year_body, SPLITS)

    # ---- 3) By [year, center] ----
    gb_year_center = (
        df.groupby(["year", "center", "split"])["patient_token"]
        .nunique()
        .reset_index()
    )

    agg_year_center = gb_year_center.pivot_table(
        index=["year", "center"],
        columns="split",
        values="patient_token",
        fill_value=0,
        aggfunc="sum",
    )

    # Ensure all split columns exist and in the right order
    for s in SPLITS:
        if s not in agg_year_center.columns:
            agg_year_center[s] = 0

    agg_year_center = agg_year_center[SPLITS].astype(int)

    # Bring index back to columns
    agg_year_center = agg_year_center.reset_index().sort_values(["year", "center"])

    # Add percentage columns
    agg_year_center = add_split_percentages(agg_year_center, SPLITS)

    return {
        "by_split": agg_split,
        "by_year_body": agg_year_body,
        "by_year_center": agg_year_center,
    }


def save_aggregates(aggregates, output_dir):
    """Save each aggregated table as a separate CSV file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_map = {
        "by_split": "agg_by_split.csv",
        "by_year_body": "agg_by_year_body.csv",
        "by_year_center": "agg_by_year_center.csv",
    }

    for key, table in aggregates.items():
        filename = file_map.get(key, "{}.csv".format(key))
        out_path = output_dir / filename
        table.to_csv(out_path, index=False)
        print("Saved {} -> {}".format(key, out_path))


def main():
    # Adjust this path if the CSV is in a different location
    base_path = "/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/"
    csv_path = base_path + "splits_manifest.csv"
    output_dir = base_path + "datastats/"

    print("Loading manifest from: {}".format(csv_path))
    df = load_manifest(csv_path)

    print("Creating aggregated tables...")
    aggregates = make_aggregates(df)

    # print("Saving aggregated tables...")
    # save_aggregates(aggregates, output_dir)

    # Optionally: print them to stdout
    print("\n=== Patients by split (columns train/val/test) ===")
    print(aggregates["by_split"])

    print("\n=== Patients by [year, body] (columns train/val/test) ===")
    print(aggregates["by_year_body"])

    print("\n=== Patients by [year, center] (columns train/val/test) ===")
    print(aggregates["by_year_center"])


if __name__ == "__main__":
    main()

