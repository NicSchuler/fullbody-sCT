import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns


def _short_model_label(model_name):
    name = str(model_name)
    if "abdomen_" in name:
        return name.split("abdomen_", 1)[1]
    return name


def _infer_family_title(model_names):
    names = [str(n).lower() for n in model_names]
    if names and all("pix2pix" in n for n in names):
        return "Pix2Pix"
    if names and all("cyclegan" in n for n in names):
        return "CycleGAN"
    if names and all("cut" in n for n in names):
        return "CUT"
    return None


def plot_boxplot_with_stats(
    df_plot,
    outlier_counts,
    stats_df=None,
    statistical_test=None,
    ordered_labels=None,
    with_p_value=True,
    with_effect_size=True,
    grouped = False,
    fig_height = 8
):

    # Color palette (expanded for many-model comparisons)
    custom_hex_colors = [
        "#4DBBD5", "#00A087", "#3C5488", "#E64B35", "#F39B7F", "#8491B4",
        "#91D1C2", "#7E6148", "#B09C85", "#DC0000", "#008B8B", "#8A2BE2",
        "#FF8C00", "#2E8B57", "#6A5ACD", "#CD5C5C", "#20B2AA", "#D2691E",
        "#1E90FF", "#228B22", "#FF1493", "#708090", "#BC8F8F", "#4169E1",
    ]

    model_palette = generate_model_palette(df_plot, custom_hex_colors)

    # Determine how many columns the plot should have
    cols = 1 + int(with_p_value) + int(with_effect_size)
    width_ratios = [4] + [1] * (cols - 1)
    gs = gridspec.GridSpec(1, cols, width_ratios=width_ratios, wspace=0.1)

    #fig = plt.figure(figsize=(14 if cols == 1 else 16, fig_height)) #14
    fig = plt.figure(figsize=(10, 12)) #14
    fig = plt.figure(figsize=(20, 12)) #14

    family_title = _infer_family_title(df_plot["comparison"].unique())
    if family_title:
        fig.suptitle(family_title, fontsize=20, y=0.995)

    # --- Boxplot (Main Panel) ---
    ax0 = plt.subplot(gs[0])
    # same width as in your sns.boxplot call
    BOX_WIDTH = 0.75 #####0.6
    hue_order = list(model_palette.keys())

    # draw the boxplot with fixed ordering (important!)
    sns.boxplot(
        data=df_plot,
        y="dvh_label",
        x="dvh_difference",
        hue="comparison",
        palette=model_palette,
        ax=ax0,
        width=BOX_WIDTH,
        fliersize=2,
        order=ordered_labels,
        hue_order=hue_order,
    )

    ax0.axvline(x=2, color='red', linestyle='--', linewidth=1)
    ax0.axvline(x=-2, color='red', linestyle='--', linewidth=1)

    # map each label -> its category center on the y-axis
    label_to_tick = {lbl: y for lbl, y in zip(ordered_labels, ax0.get_yticks())}

    # annotate using exact dodge positions
    for label in ordered_labels:
        # which hue levels actually exist for this label?
        present_hues = [h for h in hue_order
                        if ((df_plot['dvh_label'] == label) & (df_plot['comparison'] == h)).any()]
        k = max(1, len(present_hues))  # number of boxes in this row
        base_y = label_to_tick[label]

        for j, model in enumerate(present_hues):
            counts = outlier_counts[
                (outlier_counts['dvh_label'] == label) &
                (outlier_counts['comparison'] == model)
                ]
            if counts.empty:
                continue

            # center of the j-th hue box within this row:
            # spans [base_y - BOX_WIDTH/2, base_y + BOX_WIDTH/2] split into k equal bands
            y_pos = base_y - BOX_WIDTH / 2 + (j + 0.5) * (BOX_WIDTH / k)

            above = counts['outliers_above_2'].values[0]
            below = counts['outliers_below_m2'].values[0]
            ax0.text(2.05, y_pos, str(above), va='center', fontsize=13, color='black') #2.05, fontsize=9
            ax0.text(-2.25, y_pos, str(below), va='center', fontsize=13, color='black') #-2.1

    # Set axis limits
    max_diff = df_plot["dvh_difference"].abs().max()
    x_lim = max(2.5, round(max_diff + 0.5))
    ax0.set_xlim(-x_lim, x_lim)
    #ax0.set_xlim(-6.5, 6.5) ####

    # Styling
    ax0.set_xlabel("Diff., % = (sCTdose - dCTdose)/dCTdose × 100", fontsize=16)
    ax0.set_ylabel("DVH Indicator", fontsize=18) #12
    if grouped:
        ax0.set_ylabel("Structure Group", fontsize=12) #####show
    legend_cols = min(3, max(1, len(model_palette)))
    handles, labels = ax0.get_legend_handles_labels()
    display_labels = [_short_model_label(lbl) for lbl in labels]
    ax0.legend(
        handles=handles,
        labels=display_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.00),
        ncol=legend_cols,
        frameon=False,
        fontsize=14
    )
    ax0.grid(axis='x', linestyle=':', linewidth=0.8)

    # --- P-value Plot ---
    if with_p_value:
        ax1 = plt.subplot(gs[1], sharey=ax0)
        pvals = stats_df.loc[ordered_labels]

        ax1.barh(pvals.index, pvals["p_value"], color='gray')
        ax1.axvline(0.05, color='red', linestyle='--', linewidth=1)

        if statistical_test == "Friedman":
            ax1.set_xlim(0, 0.5)
        else:
            ax1.set_xlim(0, 1.0)

        ax1.set_xlabel(f"p-value\n{statistical_test} test", fontsize=16)
        ax1.tick_params(axis='y', left=False, labelleft=False)

        for i, (label, row) in enumerate(pvals.iterrows()):
            ax1.text(row["p_value"] + 0.01, label, f"{row['p_value']:.3f}", va='center', fontsize=12)

    # --- Effect Size Plot ---
    if with_effect_size:
        ax2 = plt.subplot(gs[2 if with_p_value else 1], sharey=ax0)
        effects = stats_df.loc[ordered_labels]
        ax2.barh(effects.index, effects["effect_size"], color='steelblue', alpha=0.7)

        if statistical_test == "Friedman":
            ax2.axvline(0.5, color='red', linestyle='--', linewidth=1)
            ax2.set_xlim(0, 1)
            ax2.set_xlabel("Effect Size\nKendall's W", fontsize=12)
            ax2.tick_params(axis='y', left=False, labelleft=False)
            ax2.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        else:
            ax2.axvline(0.5, color='red', linestyle='--', linewidth=1)
            ax2.axvline(-0.5, color='red', linestyle='--', linewidth=1)
            ax2.axvline(0.0, color='black', linestyle='-', linewidth=1)

            ax2.set_xlim(-1, 1)
            ax2.set_xlabel("Effect Size\nRank-biserial correlation", fontsize=10)
            ax2.tick_params(axis='y', left=False, labelleft=False)
            ax2.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])

        for i, (label, row) in enumerate(effects.iterrows()):
            ax2.text(row["effect_size"] + 0.02, label, f"{row['effect_size']:.2f}", va='center', fontsize=12)

    def format_label(lbl):
        organ, metric = lbl.rsplit("_", 1)
        if metric.lower() == "mean":
            return f"{organ.upper()} mean" if organ.upper() in {"PTV", "OAR"} else f"{organ} mean"
        else:
            return f"{organ.upper()} {metric}" if organ.upper() in {"PTV", "OAR"} else f"{organ} {metric}"

    labels = [tick.get_text() for tick in ax0.get_yticklabels()]
    pretty_labels = [format_label(lbl) for lbl in labels]

    ax0.set_yticks(range(len(pretty_labels)))
    ax0.set_yticklabels(pretty_labels)

    ax0.tick_params(axis='x', labelsize=14)
    ax0.tick_params(axis='y', labelsize=16)
    #ax1.tick_params(axis='x', labelsize=12)
    plt.subplots_adjust(left=0.22)

    plt.tight_layout()
    plt.show()

def generate_model_palette(df_plot, hex_colors):
    unique_models = df_plot['comparison'].unique()

    if len(unique_models) <= len(hex_colors):
        return dict(zip(unique_models, hex_colors[:len(unique_models)]))

    # Fallback: generate a larger distinct palette when model count exceeds the fixed list.
    auto_colors = sns.color_palette("husl", n_colors=len(unique_models)).as_hex()
    return dict(zip(unique_models, auto_colors))
