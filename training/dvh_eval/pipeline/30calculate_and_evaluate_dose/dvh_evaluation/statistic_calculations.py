import pandas as pd
from scipy.stats import shapiro
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pingouin as pg

###########################################################################

# STATISTICAL TESTS

###########################################################################

def check_ANOVA_assumpations(df_long):
    # Repeated measures ANOVA
    # Parametric test -> Assumptions: Normality, sphericity, no severe outliers

    # 1. Check Normality of residuals (per group): Shapiro-Wilk test
    print("Normality test:")
    show_qq_plot = False

    for model in df_long['comparison'].unique():
        subset = df_long[df_long['comparison'] == model]['dvh_difference'].dropna()
        if len(subset) >= 3:
            stat, p = shapiro(subset)
            print(f"{model}: W = {stat:.3f}, p = {p:.4f}")
            # QQ-plot
            if show_qq_plot:
                sm.qqplot(subset, line='45')
                plt.title(f"Q-Q plot for {model}")
                plt.show()

    # 2. Sphericity: Mauchly's test

    print("Sphericity test:")

    sphericity = pg.sphericity(df_long, dv='dvh_difference', subject='patient', within='comparison')
    print(sphericity)

    # 3. Outliers

    print("Outlier test: IQR method")

    for model in df_long['comparison'].unique():
        group = df_long[df_long['comparison'] == model]['dvh_difference']
        q1 = group.quantile(0.25)
        q3 = group.quantile(0.75)
        iqr = q3 - q1
        outliers = group[(group < q1 - 3 * iqr) | (group > q3 + 3 * iqr)]
        print(f"{model}: {len(outliers)} extreme outliers")


def calculate_friedman_test(df_long):
    # Friedman Test -> comparing multiple experiments (eg. diff models) across same patients
    # Non-parametric test -> few assumptions: Paired data, ordinal data, same number of measurements, independent observations
    # Kendall's W: each patient "ranks" models based on performance for metric -> Kendall's W quantifies how much rankings agree across patients

    results = []

    for metric in df_long['dvh_label'].unique():
        subset = df_long[df_long['dvh_label'] == metric]
        result = pg.friedman(data=subset, dv='dvh_difference', within='comparison', subject='patient')
        stat = result['Q'].iloc[0]
        df = result['ddof1'].iloc[0]
        p = result['p-unc'].iloc[0]
        kendall_w = result['W'].iloc[0] if 'W' in result.columns else None

        results.append({
            "dvh_metric": metric,
            "chi2": stat,
            "df": df,
            "p_value": p,
            "effect_size": kendall_w #kendall W
        })

    # Convert Friedman test results to DataFrame
    friedman_df = pd.DataFrame(results).set_index('dvh_metric')
    return friedman_df

def calculate_wilcoxon_pratt_test(df_long, group1, group2):

    results = []

    for metric in df_long['dvh_label'].unique():
        subset = df_long[df_long['dvh_label'] == metric]

        # Pivot to align paired patient values for both groups
        wide = subset.pivot_table(index="patient", columns="comparison", values="dvh_difference", aggfunc="mean")

        # Run Wilcoxon signed-rank test (Pratt method handles zero differences)
        # Two-sided -> don't check direction, only if the two groups differ
        # method: "auto" -> computes exact distribution of T, used for small sample sizes
        result = pg.wilcoxon(wide[group1], wide[group2], alternative="two-sided", zero_method="pratt", method="auto")
        T = result["W-val"].iloc[0]
        p = result["p-val"].iloc[0]
        effsize = result["RBC"].iloc[0] if "RBC" in result.columns else None


        results.append({
            "dvh_metric": metric,
            "T": T,
            "p_value": p,
            "effect_size": effsize, #RBC
            "n": len(wide)
        })

    wilcoxon_df = pd.DataFrame(results).set_index("dvh_metric")
    return wilcoxon_df


def identify_outliers(df, difference_col, confidence_level=2):

    upper_outliers = df[df[difference_col] > confidence_level]
    lower_outliers = df[df[difference_col] < -confidence_level]

    print(f"Outliers ABOVE +{confidence_level}%:")
    for _, row in upper_outliers.iterrows():
        print(f"Label: {row['dvh_label']}, Model: {row['comparison']}, Diff: {row[difference_col]:.2f}%")

    print(f"Outliers BELOW -{confidence_level}%:")
    for _, row in lower_outliers.iterrows():
        print(f"Label: {row['dvh_label']}, Model: {row['comparison']}, Diff: {row[difference_col]:.2f}%")


def identify_outliers_patient(df, difference_col, confidence_level=2):

    upper_outliers = df[df[difference_col] > confidence_level].sort_values(by=["comparison", "patient"])
    lower_outliers = df[df[difference_col] < -confidence_level].sort_values(by="comparison")

    print(f"Outliers ABOVE +{confidence_level}%:")
    for _, row in upper_outliers.iterrows():
        print(f"Patient: {row['patient']}, Label: {row['dvh_label']}, Model: {row['comparison']}, Diff: {row[difference_col]:.2f}%")

    print(f"Outliers BELOW -{confidence_level}%:")
    for _, row in lower_outliers.iterrows():
        print(f"Patient: {row['patient']}, Label: {row['dvh_label']}, Model: {row['comparison']}, Diff: {row[difference_col]:.2f}%")