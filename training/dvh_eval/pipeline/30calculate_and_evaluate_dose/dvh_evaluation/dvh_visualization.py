import os
import sys

import pandas as pd

from dose_difference_calculations import compute_dose_difference, calculate_mean_dose_differences, save_dvh_differences, calculate_mean_dose_differences_absolute
from dosimetric_calculation.statistic_calculations import identify_outliers_patient
from dosimetric_calculation.visualization_plots import plot_boxplot_with_stats
from statistic_calculations import check_ANOVA_assumpations, calculate_friedman_test, identify_outliers, calculate_wilcoxon_pratt_test

if __name__ == '__main__':
    # Sorry, sehr ugly code wege last minute Master Thesis Abgab ;-)

    # Variables
    path_excel = "./" # Path to excel file with experiments to compare
    path_results = "/media/nico/Extreme SSD/USZ/dosimetric_calculation/dosimetry_results/" # folder with results: (dvh_table_fake.csv, dvh_table_real.csv)
    output_path_excel = None # Path to save excel with evaluation per patient
    resolution = "2mm" # in mm

    # Flavian: I always named folders like: path_results/DATASET/MODEL/MODEL_NAME/dvh_RESOLUTION/ ->containing: dvh_table_fake.csv, dvh_table_real.csv

    plot_with_p_values = True
    plot_with_effect_sizes = True

    research_question = "paper_MR_in_MR_opp" # ending of excel-file name where experiments to compare defined: eg. experiments_Q1.xlsx

    # Indicators / Dose metrics
    dose_metrics = ['mean', 'std', 'max', 'min', 'D_2', 'D_5', 'D_95', 'D_98']

    # Information about Experiments
    df_experiments_info = pd.read_excel(os.path.join(path_excel, f"experiments_{research_question}.xlsx"), engine="openpyxl")

    # Dataframe to save all processed data
    difference_method = "standard_percent_difference" #"prescribed_percent_difference"
    df_dvh_diff_all = pd.DataFrame()  # empty

    # Loop through experiments
    for exp_idx, exp_row in df_experiments_info.iterrows():

        dataset = exp_row.Dataset
        model = exp_row.Model
        model_name = exp_row.Name
        plot_name = exp_row.PlotName


        print(f"Processing: {dataset} {model} {model_name}")

        # Load DVH data
        csv_file_dvh_real = os.path.join(path_results, dataset, model, model_name, f"dvh_{resolution}", "dvh_table_real.csv")
        csv_file_dvh_fake = os.path.join(path_results, dataset, model, model_name, f"dvh_{resolution}", "dvh_table_fake.csv")

        df_dvh_real = pd.read_csv(csv_file_dvh_real)
        df_dvh_fake = pd.read_csv(csv_file_dvh_fake)

        # Calculate DVH difference
        df_dvh_diff = compute_dose_difference(df_dvh_real, df_dvh_fake, dose_metrics, method=difference_method)

        # Modify Columns -> TBD: make dynamic for different levels of comparison
        df_dvh_diff["comparison"] = plot_name #model ###
        df_dvh_diff['name'] = df_dvh_diff['name'].apply(lambda x: 'PTV' if 'PTV' in x and x != 'Ring PTV' else x) # -> rename all PTV1_V1_1a variations to PTV

        # Concatenate current experiment to total dataframe
        df_dvh_diff_all = pd.concat([df_dvh_diff_all, df_dvh_diff], ignore_index=True)

    ###########################################################################

    # Transform Dataframe to long format for visualization purposes
    df_dvh_diff_all_long = df_dvh_diff_all.melt(id_vars=['name', 'patient', 'comparison'],
                                                value_vars=dose_metrics,
                                                var_name='dvh_indicator',
                                                value_name='dvh_difference')

    # Filter only target indicators
    target_combos = [
        ('PTV', 'D2'),
        ('PTV', 'D95'),
        ('PTV', 'mean'),
        ('Bladder', 'D2'),
        ('Bladder', 'mean'),
        ('Rectum', 'D2'),
        ('Rectum', 'mean')
    ]

    # Modify labels
    df_dvh_diff_all_long['dvh_indicator'] = df_dvh_diff_all_long['dvh_indicator'].str.replace('_', '', regex=False)
    df_dvh_diff_all_long = df_dvh_diff_all_long[df_dvh_diff_all_long.apply(lambda row: (row['name'], row['dvh_indicator']) in target_combos, axis=1)]
    df_dvh_diff_all_long['dvh_difference'] = pd.to_numeric(df_dvh_diff_all_long['dvh_difference'], errors='coerce')

    # Create ordered dvh_label list
    ordered_dvh_labels = [f"{name}_{indicator}" for name, indicator in target_combos]

    # Generate dvh_label column
    def format_dvh_name(row):
        return f"{row['name']}_{row['dvh_indicator']}"

    df_dvh_diff_all_long["dvh_label"] = df_dvh_diff_all_long.apply(format_dvh_name, axis=1)

    # Apply the correct categorical order
    df_dvh_diff_all_long["dvh_label"] = pd.Categorical(
        df_dvh_diff_all_long["dvh_label"],
        categories=ordered_dvh_labels,
        ordered=True
    )

    df_plot = df_dvh_diff_all_long.dropna(subset=["dvh_label"])

    ###########################################################################

    # Count outliers per DVH label & model
    outlier_counts = (
        df_plot.groupby(['dvh_label', 'comparison'])['dvh_difference']
        .agg(outliers_above_2=lambda x: (x > 2).sum(),
             outliers_below_m2=lambda x: (x < -2).sum())
        .reset_index()
    )

    ###########################################################################

    # Identify patient-level outliers
    identify_outliers_patient(df_plot, "dvh_difference", 2)

    # Save patient level dvhs
    save_dvh_differences(df_dvh_diff_all_long)

    ###########################################################################
    # Mean DVH
    mean_dvh_diff_per_model = calculate_mean_dose_differences(df_dvh_diff_all_long, output_path_excel, research_question)
    mean_dvh_diff_per_model_abs = calculate_mean_dose_differences_absolute(df_dvh_diff_all_long, output_path_excel, research_question)

    # Identify Outliers (Mean DVH)
    identify_outliers(mean_dvh_diff_per_model, "mean_difference", 1)

    ###########################################################################

    # STATISTICAL TESTS

    groups_to_compare = df_dvh_diff_all_long['comparison'].unique()
    number_of_groups_to_compare = len(groups_to_compare)

    if number_of_groups_to_compare > 2:
        print("Apply Friedman Test")
        stats_df = calculate_friedman_test(df_dvh_diff_all_long)
        statistical_test = "Friedman"
    else:
        print("Apply Wilcoxon Test")
        stats_df = calculate_wilcoxon_pratt_test(df_dvh_diff_all_long, groups_to_compare[0], groups_to_compare[1])
        statistical_test = "Wilcoxon-Pratt"

    # Create plot
    plot_boxplot_with_stats(
        df_plot,
        outlier_counts,
        stats_df=stats_df,
        statistical_test=statistical_test,
        ordered_labels=ordered_dvh_labels,
        with_p_value=plot_with_p_values,
        with_effect_size=plot_with_effect_sizes
    )






