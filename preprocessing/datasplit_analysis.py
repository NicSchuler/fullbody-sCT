import pandas as pd

# 1) Load your manifest (adapt path if needed)
df = pd.read_csv("/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/splits_manifest.csv")

# 2) Count patients per (body, center, split)
#    (unique patient_token in case it appears multiple times)
counts = (
    df.groupby(["body", "center", "split"])["patient_token"]
      .nunique()
      .reset_index(name="n_patients")
)

# 3) Pivot to get columns train / val / test
pivot = (
    counts.pivot(index=["body", "center"],
                 columns="split",
                 values="n_patients")
          .fillna(0)
          .astype(int)
)

# Ensure columns exist even if one split is missing in the data
for col in ["train", "val", "test"]:
    if col not in pivot.columns:
        pivot[col] = 0

# 4) Compute row totals
pivot["total"] = pivot["train"] + pivot["val"] + pivot["test"]

# 5) Compute percentages per row
pivot["train_pct"] = 100 * pivot["train"] / pivot["total"]
pivot["val_pct"]   = 100 * pivot["val"]   / pivot["total"]
pivot["test_pct"]  = 100 * pivot["test"]  / pivot["total"]

# 6) Optional: sort and reset index for nice printing
pivot = (
    pivot[["train", "val", "test", "total", "train_pct", "val_pct", "test_pct"]]
          .sort_index()
          .reset_index()
)

# 7) Show result
pd.set_option("display.float_format", "{:.3f}".format)
print(pivot)
