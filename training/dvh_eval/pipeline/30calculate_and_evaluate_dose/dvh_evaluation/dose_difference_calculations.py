import pandas as pd

def compute_dose_difference(df_dvh_real, df_dvh_fake, dose_metrics, method="standard_percent_difference"):

    # Merge on patient and name (RTstruct) columns
    df_merged = pd.merge(df_dvh_fake, df_dvh_real, on=["patient", "name"], suffixes=("_fake", "_real"))

    if method == "standard_percent_difference":

        # Omit rows with low dose in mean_fake or D_2_fake
        valid_rows = (df_merged["mean_fake"] >= 8) & (df_merged["D_2_fake"] >= 8)
        df_merged_valid = df_merged.loc[valid_rows].copy()

        # Compute % difference: (fake - real) / real * 100
        df_diff = pd.DataFrame()
        df_diff["patient"] = df_merged_valid["patient"]
        df_diff["name"] = df_merged_valid["name"]

        for metric in dose_metrics:
            fake_col = f"{metric}_fake"
            real_col = f"{metric}_real"

            df_diff[metric] = (df_merged_valid[fake_col] - df_merged_valid[real_col]) / df_merged_valid[real_col] * 100
            #df_diff[metric] = (df_merged[fake_col] - df_merged[real_col]) / df_merged[fake_col] * 100

        return df_diff

def calculate_mean_dose_differences(df_long, output_path, research_question):

    # Group by DVH label and model to compute mean difference
    mean_dvh_diff_per_model = (
        df_long
        .groupby(['dvh_label', 'comparison'])['dvh_difference']
        .mean()
        .reset_index()
        .rename(columns={'dvh_difference': 'mean_difference'})
    )

    # Pivot so that models are rows, and DVH labels are columns
    mean_diff_table = mean_dvh_diff_per_model.pivot(
        index='comparison',
        columns='dvh_label',
        values='mean_difference'
    )

    # Display the table
    print("\nMean DVH dose differences (rows = model, columns = DVH label):")
    print(mean_diff_table.round(2))

    if output_path:
        mean_diff_table.round(2).to_excel(f"{output_path}/mean_dvh_differences_{research_question}.xlsx")

    return mean_dvh_diff_per_model

def calculate_mean_dose_differences_absolute(df_long, output_path, research_question):

    df_long = df_long.copy()
    df_long["dvh_difference"] = df_long["dvh_difference"].abs()

    # Group by DVH label and model to compute mean difference
    mean_dvh_diff_per_model = (
        df_long
        .groupby(['dvh_label', 'comparison'])['dvh_difference']
        .mean()
        .reset_index()
        .rename(columns={'dvh_difference': 'mean_difference'})
    )

    # Pivot so that models are rows, and DVH labels are columns
    mean_diff_table = mean_dvh_diff_per_model.pivot(
        index='comparison',
        columns='dvh_label',
        values='mean_difference'
    )

    # Display the table
    print("\nMean DVH dose differences (rows = model, columns = DVH label):")
    print(mean_diff_table.round(2))

    if output_path:
        mean_diff_table.round(2).to_excel(f"{output_path}/mean_dvh_differences_abs_{research_question}.xlsx")

    return mean_dvh_diff_per_model

def save_dvh_differences(df_long):

    # Pivot so that models are rows, and DVH labels are columns
    dvh_all = df_long.pivot_table(
        index=["comparison", "patient"],
        columns="dvh_label",
        values="dvh_difference",
        aggfunc="first"
    ).sort_index()

    dvh_all.round(2).to_excel("./dvh_difference_per_model_patient.xlsx")
    dvh_stats = dvh_all.groupby(level="comparison").agg(["min", "max", "mean"]).round(2)

    # Print row by row
    for comparison, row in dvh_stats.iterrows():
        print(f"\nComparison: {comparison}")
        for dvh_label in dvh_stats.columns.levels[0]:
            vals = row[dvh_label]
            print(f"  {dvh_label:<20} {vals['mean']:.2f} ({vals['min']:.2f}, {vals['max']:.2f})")

