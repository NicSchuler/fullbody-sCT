#!/usr/bin/env python3
"""
DVH Evaluation Script

Crawls outputs/dvh_results/<model>/<case>/ for dvh_metrics_ct.csv and
dvh_metrics_sct.csv, computes per-case dose differences, then runs
statistical tests and produces a boxplot using the dvh_evaluation module.

Usage:
    python scripts/run_dvh_evaluation.py

Configuration is at the top of this file.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# -- Path setup ---------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'dvh_evaluation'))

from dose_difference_calculations import (
    compute_dose_difference,
    calculate_mean_dose_differences,
    calculate_mean_dose_differences_absolute,
)
from statistic_calculations import (
    calculate_friedman_test,
    calculate_wilcoxon_pratt_test,
    identify_outliers,
    identify_outliers_patient,
)
from visualization_plots import plot_boxplot_with_stats

# -- Configuration ------------------------------------------------------------

DVH_RESULTS_ROOT = os.path.join(PROJECT_ROOT, 'outputs', 'dvh_results')

# Set to a directory path to save Excel summaries, or None to skip.
OUTPUT_PATH = None

# Metrics that exist as columns in dvh_metrics_ct/sct.csv
DOSE_METRICS = ['mean', 'D_2', 'D_5', 'D_95', 'D_98']

# ROI grouping:
# - PTV: kidney left
# - OARs: selected clinically relevant structures only
PTV_ROI_NAME = 'kidney left'
PTV_METRICS = {'mean', 'D2', 'D95'}
OAR_METRICS = {'mean', 'D2'}
OAR_ROI_NAMES = {'kidney right', 'liver', 'spinal cord'}
GROUP_BY_STRUCTURE_TYPE = True

RESEARCH_QUESTION = 'abdomen_synthrad'  # suffix for saved Excel files
PLOT_P_VALUES     = True
PLOT_EFFECT_SIZES = True

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DVH evaluation across all models or selected models."
    )
    parser.add_argument(
        "--model",
        action="append",
        help=(
            "Model name under outputs/dvh_results (repeatable). "
            "If omitted, all models are evaluated."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    available_models = sorted(
        m for m in os.listdir(DVH_RESULTS_ROOT)
        if os.path.isdir(os.path.join(DVH_RESULTS_ROOT, m))
    )
    requested_models = args.model or available_models

    missing_models = [m for m in requested_models if m not in available_models]
    if missing_models:
        print(f"Requested model(s) not found in {DVH_RESULTS_ROOT}: {missing_models}")
        return 1

    models_to_run = [m for m in requested_models if m in available_models]
    if not models_to_run:
        print("No models selected for evaluation.")
        return 1

    print(f"Evaluating models: {models_to_run}")

    # -- Load all per-case CSVs ---------------------------------------------------
    df_all_diffs = pd.DataFrame()

    for model in models_to_run:
        model_path = os.path.join(DVH_RESULTS_ROOT, model)
        for case in sorted(os.listdir(model_path)):
            case_path = os.path.join(model_path, case)
            ct_csv  = os.path.join(case_path, 'dvh_metrics_ct.csv')
            sct_csv = os.path.join(case_path, 'dvh_metrics_sct.csv')

            if not (os.path.exists(ct_csv) and os.path.exists(sct_csv)):
                print(f"  Skipping {model}/{case}: missing CSVs")
                continue

            df_ct  = pd.read_csv(ct_csv).rename(columns={'caseId': 'patient'})
            df_sct = pd.read_csv(sct_csv).rename(columns={'caseId': 'patient'})

            df_diff = compute_dose_difference(df_ct, df_sct, DOSE_METRICS)
            df_diff['comparison'] = model

            print(f"  Loaded {model} / {case}  ({len(df_diff)} ROIs)")
            df_all_diffs = pd.concat([df_all_diffs, df_diff], ignore_index=True)

    if df_all_diffs.empty:
        print("No data loaded – check DVH_RESULTS_ROOT.")
        return 1

    print(f"\nTotal rows before filtering: {len(df_all_diffs)}")

    # -- Reshape to long format ---------------------------------------------------
    df_long = df_all_diffs.melt(
        id_vars=['patient', 'name', 'comparison'],
        value_vars=DOSE_METRICS,
        var_name='dvh_indicator',
        value_name='dvh_difference',
    )

    # Strip underscores: D_2 -> D2, D_95 -> D95, mean stays mean
    df_long['dvh_indicator'] = df_long['dvh_indicator'].str.replace('_', '', regex=False)

    # Filter to PTV + OAR logic:
    # - PTV (kidney left): mean, D2, D95
    # - OARs (selected list): mean, D2
    is_ptv = df_long['name'] == PTV_ROI_NAME
    is_oar = df_long['name'].isin(OAR_ROI_NAMES)
    ptv_keep = is_ptv & df_long['dvh_indicator'].isin(PTV_METRICS)
    oar_keep = is_oar & df_long['dvh_indicator'].isin(OAR_METRICS)
    df_long = df_long[ptv_keep | oar_keep].copy()

    # Build dvh_label and enforce order
    if GROUP_BY_STRUCTURE_TYPE:
        df_long['structure_type'] = np.where(df_long['name'] == PTV_ROI_NAME, 'PTV', 'OAR')
        df_long['dvh_label'] = df_long.apply(
            lambda r: f"{r['structure_type']}_{r['dvh_indicator']}", axis=1
        )
        ordered_labels = ['PTV_mean', 'PTV_D2', 'PTV_D95', 'OAR_mean', 'OAR_D2']
    else:
        present_names = sorted(df_long['name'].dropna().unique().tolist())
        oar_names = [n for n in present_names if n != PTV_ROI_NAME]
        ordered_labels = (
            [f"{PTV_ROI_NAME}_mean", f"{PTV_ROI_NAME}_D2", f"{PTV_ROI_NAME}_D95"] +
            [f"{n}_mean" for n in oar_names] +
            [f"{n}_D2" for n in oar_names]
        )
        df_long['dvh_label'] = df_long.apply(
            lambda r: f"{r['name']}_{r['dvh_indicator']}", axis=1
        )
    df_long['dvh_label'] = pd.Categorical(
        df_long['dvh_label'], categories=ordered_labels, ordered=True
    )
    df_long = df_long.dropna(subset=['dvh_label'])
    df_long['dvh_difference'] = pd.to_numeric(df_long['dvh_difference'], errors='coerce')

    print(f"Rows after filtering to PTV/OAR combos: {len(df_long)}\n")

    # -- Outlier annotation counts (used by the plot) ----------------------------
    outlier_counts = (
        df_long
        .groupby(['dvh_label', 'comparison'])['dvh_difference']
        .agg(
            outliers_above_2=lambda x: (x > 2).sum(),
            outliers_below_m2=lambda x: (x < -2).sum(),
        )
        .reset_index()
    )

    # -- Print patient-level outliers --------------------------------------------
    identify_outliers_patient(df_long, 'dvh_difference', 2)

    # -- Mean differences (prints table, optionally saves Excel) -----------------
    mean_dvh_diff = calculate_mean_dose_differences(
        df_long, OUTPUT_PATH, RESEARCH_QUESTION
    )
    calculate_mean_dose_differences_absolute(
        df_long, OUTPUT_PATH, RESEARCH_QUESTION
    )
    identify_outliers(mean_dvh_diff, 'mean_difference', confidence_level=1)

    # -- Drop labels filtered out by the dose threshold --------------------------
    # compute_dose_difference drops ROIs with mean < 8 Gy, so some selected labels
    # may produce no rows (e.g. low-dose OARs like liver). Warn and remove them.
    present_labels = set(df_long['dvh_label'].dropna().unique())
    dropped = [l for l in ordered_labels if l not in present_labels]
    if dropped:
        print(f"\nWarning: the following labels had no data after the dose "
              f"threshold filter and will be skipped: {dropped}")
        ordered_labels = [l for l in ordered_labels if l in present_labels]
        df_long = df_long[df_long['dvh_label'].isin(ordered_labels)].copy()
        df_long['dvh_label'] = pd.Categorical(
            df_long['dvh_label'], categories=ordered_labels, ordered=True
        )
        outlier_counts = outlier_counts[outlier_counts['dvh_label'].isin(ordered_labels)]

    # -- Statistical tests --------------------------------------------------------
    groups = df_long['comparison'].unique()
    stats_df = None
    test_name = None
    plot_p_values = PLOT_P_VALUES
    plot_effect_sizes = PLOT_EFFECT_SIZES

    if len(groups) > 2:
        print(f"\nApplying Friedman test across {len(groups)} models ...")
        stats_df = calculate_friedman_test(df_long)
        test_name = 'Friedman'
    elif len(groups) == 2:
        print(f"\nApplying Wilcoxon-Pratt test: {groups[0]} vs {groups[1]} ...")
        stats_df = calculate_wilcoxon_pratt_test(df_long, groups[0], groups[1])
        test_name = 'Wilcoxon-Pratt'
    else:
        print("\nOnly one model selected. Skipping statistical test panels.")
        plot_p_values = False
        plot_effect_sizes = False

    if stats_df is not None:
        print(stats_df)

    # -- Plot ---------------------------------------------------------------------
    plot_boxplot_with_stats(
        df_long,
        outlier_counts,
        stats_df=stats_df,
        statistical_test=test_name,
        ordered_labels=ordered_labels,
        with_p_value=plot_p_values,
        with_effect_size=plot_effect_sizes,
        grouped=GROUP_BY_STRUCTURE_TYPE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
