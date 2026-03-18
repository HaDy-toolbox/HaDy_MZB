import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal
import scikit_posthocs as sp
import numpy as np
import os
import matplotlib.ticker as mtick

from variables_from_config import HABITAT_TARGETS, OUTPUT_FOLDER, METRICS_TO_COMPUTE, NUMBER_OF_HABITATS, STATIC_HABITAT_CSV_PATH, FINAL_CSV_PATH


HABITAT_TARGETS = HABITAT_TARGETS[0] # the plots are only done for one habitat type in focus (the first one of the list, if a list is provided)

csv_path = FINAL_CSV_PATH
csv_path_static = STATIC_HABITAT_CSV_PATH

# csv_path = r"C:\Users\lecrivau\Documents\00_Research_Assistant\Toolbox\QGIS_to_share\Metrics_small_zone\Summer_2019\metrics.csv"
# csv_path_static = r"C:\Users\lecrivau\source\repos\HaDy_MZB\data\output\Small_zone\Summer_2019_small_scale\Desiccation_files\mesh_habitat_egg_bancs_gravier.csv" 

habitat_labels_dict = {
    0: "Dry",
    1: "Stagnant",
    2: "Slow",
    3: "Slow to fast",
    4: "Fast to very fast",
    5: "Very fast",
    6: "Violent"
}


habitat_colors_dict = {
    0: "#fde725",
    1: "#009d61",
    2: "#8c3c77",
    3: "#d57628",
    4: "#92a8d1",
    5: "#31688e",
    6: "#d43d35"
}

# SUPPORT FUNCTIONS -------------------------------------------------------------------------------------------------------------------------------------------------
# ==========================================================
# FUNCTION: Extract intensity column (vectorized, fast)
# ==========================================================
def build_intensity_column(df, habitat_col):
    """
    For each row, extracts the probability from column prob_hab_X
    where X = mostProb value.
    """
    prob_cols = [c for c in df.columns if c.startswith("prob_hab_")]

    # Create empty intensity column
    intensity = pd.Series(index=df.index, dtype=float)

    for col in prob_cols:
        hab_value = int(col.split("_")[-1])
        mask = df[habitat_col] == hab_value
        intensity.loc[mask] = df.loc[mask, col]

    return intensity

# ==========================================================
# FUNCTION: Compact Letter Assignment
# ==========================================================
def compact_letter_assignment(pmat, alpha=0.05):
    groups = pmat.index.tolist()
    letters = {g: "" for g in groups}
    current_letter = "a"

    while "" in letters.values():
        for g in groups:
            if letters[g] != "":
                continue
            ok = True
            for h in groups:
                if g == h or letters[h] != current_letter:
                    continue
                if pmat.loc[g, h] < alpha:
                    ok = False
                    break
            if ok:
                letters[g] = current_letter
        current_letter = chr(ord(current_letter) + 1)

    return letters

# ==========================================================
# FUNCTION: Filter meshes experiencing the target habitat
# ==========================================================
def filter_meshes_with_target_habitat(df, habitat_target):
    """
    Keep only meshes that experience the target habitat
    at least once during the time series.
    """
    target_column = f"prob_hab_{habitat_target}"

    if target_column not in df.columns:
        raise ValueError(f"Column {target_column} not found in dataframe.")

    return df[df[target_column] > 0].copy()

# PLOT FUNCTIONS ----------------------------------------------------------------------------------------------------------------------------------------------------
# ==========================================================
# FUNCTION: ALL MESHES IN SIMULATED DOMAIN - Plot most probable habitat type and related intensity
# ==========================================================
def plot_most_prob_intensity(
    csv_path,
    habitat_col,
    habitat_labels_dict,
    habitat_colors_dict,
    number_of_habitats,
    save_path
):

    df = pd.read_csv(csv_path)
    df[habitat_col] = df[habitat_col].astype(int)

    # -----------------------------
    # Habitat range definition
    # -----------------------------
    habitat_range = list(range(0, number_of_habitats))
    habitat_labels_dict = habitat_labels_dict
        
    habitat_labels = {k: habitat_labels_dict[k] for k in habitat_range}

    # -----------------------------
    # Build intensity column
    # -----------------------------
    df["intensity"] = build_intensity_column(df, habitat_col)


    # -----------------------------
    # Filtering
    # -----------------------------
    df = df[df[habitat_col].isin(habitat_range)]

    # Remove rows with missing intensity
    df = df.dropna(subset=["intensity"])

    # -----------------------------
    # Prepare labels
    # -----------------------------
    df["habitat_label"] = df[habitat_col].map(habitat_labels)
    label_order = [habitat_labels[i] for i in habitat_range]

    # Colors (editable)
    colors_dict = habitat_colors_dict
    palette = {habitat_labels[i]: colors_dict[i] for i in habitat_range}

    # -----------------------------
    # Kruskal–Wallis
    # -----------------------------
    groups = [
        df[df["habitat_label"] == lbl]["intensity"]
        for lbl in label_order
        if len(df[df["habitat_label"] == lbl]) > 0
    ]

    H, p = kruskal(*groups)
    print(f"Kruskal–Wallis: H={H:.3f}, p={p:.4f}")

    # -----------------------------
    # Dunn posthoc
    # -----------------------------
    posthoc_df = sp.posthoc_dunn(
        df,
        val_col="intensity",
        group_col="habitat_label",
        p_adjust="holm"
    )

    letters = compact_letter_assignment(posthoc_df)

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(12, 8))

    ax = sns.boxplot(
        x="habitat_label",
        y="intensity",
        data=df,
        palette=palette,
        order=label_order,
        showfliers=False,
        saturation=1
    )

    plt.xlabel("Most probable habitat type", fontsize=20)
    plt.ylabel("Associated Probability", fontsize=20)
    plt.title("Probability distribution per most probable habitat type\n for the entire simulated domain", fontsize=22)

    plt.xticks(rotation=45, ha="right", fontsize=16)
    plt.yticks(fontsize=18)

    # Add letters
    """
    for i, lbl in enumerate(label_order):
        group_vals = df[df["habitat_label"] == lbl]["intensity"]
        if len(group_vals) == 0:
            continue

        Q3 = group_vals.quantile(0.75)

        plt.text(
            i, Q3 + 0.02,
            letters.get(lbl, ""),
            ha='center',
            va='bottom',
            fontsize=20,
            fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8)
        )
    """
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show()

# ==========================================================
# FUNCTION: TARGET HABITAT - Plot most probable habitat type and related intensity
# ==========================================================
def plot_most_prob_intensity_target_hab(
    csv_path,
    habitat_col,
    habitat_labels_dict,
    habitat_colors_dict,
    number_of_habitats,
    habitat_in_focus,
    save_path
):

    df = pd.read_csv(csv_path)
    df[habitat_col] = df[habitat_col].astype(int)

    # -----------------------------
    # Habitat range definition
    # -----------------------------
    habitat_range = list(range(0, number_of_habitats))
    habitat_labels_dict = habitat_labels_dict
        
    habitat_labels = {k: habitat_labels_dict[k] for k in habitat_range}

    # -----------------------------
    # Build intensity column
    # -----------------------------
    df["intensity"] = build_intensity_column(df, habitat_col)

    # -----------------------------
    # Filtering
    # -----------------------------
    focus_column = f"prob_hab_{habitat_in_focus}"

    df = df[df[focus_column] > 0]
    df = df[df[habitat_col].isin(habitat_range)]

    # Remove rows with missing intensity
    df = df.dropna(subset=["intensity"])

    # -----------------------------
    # Prepare labels
    # -----------------------------
    df["habitat_label"] = df[habitat_col].map(habitat_labels)
    label_order = [habitat_labels[i] for i in habitat_range]

    # Colors (editable)
    colors_dict = habitat_colors_dict
    palette = {habitat_labels[i]: colors_dict[i] for i in habitat_range}

    # -----------------------------
    # Kruskal–Wallis
    # -----------------------------
    groups = [
        df[df["habitat_label"] == lbl]["intensity"]
        for lbl in label_order
        if len(df[df["habitat_label"] == lbl]) > 0
    ]

    H, p = kruskal(*groups)
    print(f"Kruskal–Wallis: H={H:.3f}, p={p:.4f}")

    # -----------------------------
    # Dunn posthoc
    # -----------------------------
    posthoc_df = sp.posthoc_dunn(
        df,
        val_col="intensity",
        group_col="habitat_label",
        p_adjust="holm"
    )

    letters = compact_letter_assignment(posthoc_df)

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(12, 8))

    ax = sns.boxplot(
        x="habitat_label",
        y="intensity",
        data=df,
        palette=palette,
        order=label_order,
        showfliers=False,
        saturation=1
    )

    plt.xlabel("Most probable habitat type", fontsize=20)
    plt.ylabel("Associated Probability", fontsize=20)
    plt.title("Probability distribution per most probable habitat type\nfor patches experiencing the target habitat conditions \nduring the flow time-series", fontsize=22)

    plt.xticks(rotation=45, ha="right", fontsize=16)
    plt.yticks(fontsize=18)

    # Add letters
    """
    for i, lbl in enumerate(label_order):
        group_vals = df[df["habitat_label"] == lbl]["intensity"]
        if len(group_vals) == 0:
            continue

        Q3 = group_vals.quantile(0.75)

        plt.text(
            i, Q3 + 0.02,
            letters.get(lbl, ""),
            ha='center',
            va='bottom',
            fontsize=20,
            fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8)
        )
    """
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show()

# ==========================================================
# FUNCTION: ALL MESHES IN SIMULATED DOMAIN - Percentage of most probable habitat types
# ==========================================================
def plot_most_prob_percentages(
    csv_path,
    habitat_col,
    habitat_labels_dict,
    habitat_colors_dict,
    number_of_habitats,
    save_path
):

    df = pd.read_csv(csv_path)
    df[habitat_col] = df[habitat_col].astype(int)

    # -----------------------------
    # Habitat range definition
    # -----------------------------
    habitat_range = list(range(0, number_of_habitats))
    habitat_labels_dict = habitat_labels_dict

    # Filter to selected habitat range
    df = df[df[habitat_col].isin(habitat_range)]

    # -----------------------------
    # Compute percentages
    # -----------------------------
    habitat_counts = df[habitat_col].value_counts().sort_index()
    total_elements = habitat_counts.sum()
    habitat_percentages = (habitat_counts / total_elements) * 100

    # Ensure all habitats appear (even if 0%)
    habitat_percentages = habitat_percentages.reindex(habitat_range, fill_value=0)

    # -----------------------------
    # Prepare plotting elements
    # -----------------------------
    x = habitat_percentages.index
    y = habitat_percentages.values

    labels = [habitat_labels_dict[i] for i in x]
    bar_colors = [habitat_colors_dict[i] for i in x]

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(12, 6))

    plt.bar(x, y, color=bar_colors)

    plt.xlabel("Most probable habitat type", fontsize=18)
    plt.ylabel("Percentage of patches (%)", fontsize=18)
    plt.title("Percentage of most probable habitat types across all patches \n for the entire simulated domain", fontsize=20)

    plt.xticks(ticks=x, labels=labels, rotation=30, ha='right', fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show()

# ==========================================================
# FUNCTION: TARGET HABITAT - Percentage of most probable habitat types
# ==========================================================
def plot_most_prob_percentages_target_hab(
    csv_path,
    habitat_col,
    habitat_labels_dict,
    habitat_colors_dict,
    number_of_habitats,
    habitat_in_focus,
    save_path
):

    df = pd.read_csv(csv_path)
    df[habitat_col] = df[habitat_col].astype(int)

    # -----------------------------
    # Habitat range definition
    # -----------------------------
    habitat_range = list(range(0, number_of_habitats))
    habitat_labels_dict = habitat_labels_dict

    # -----------------------------
    # Filtering
    # -----------------------------
    focus_column = f"prob_hab_{habitat_in_focus}"

    df = df[df[focus_column] > 0]
    df = df[df[habitat_col].isin(habitat_range)]

    # -----------------------------
    # Compute percentages
    # -----------------------------
    habitat_counts = df[habitat_col].value_counts().sort_index()
    total_elements = habitat_counts.sum()
    habitat_percentages = (habitat_counts / total_elements) * 100

    # Ensure all habitats appear (even if 0%)
    habitat_percentages = habitat_percentages.reindex(habitat_range, fill_value=0)

    # -----------------------------
    # Prepare plotting elements
    # -----------------------------
    x = habitat_percentages.index
    y = habitat_percentages.values

    labels = [habitat_labels_dict[i] for i in x]
    bar_colors = [habitat_colors_dict[i] for i in x]

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(12, 6))

    plt.bar(x, y, color=bar_colors)

    plt.xlabel("Most probable habitat type", fontsize=18)
    plt.ylabel("Percentage of patches (%)", fontsize=18)
    plt.title("Percentage of most probable habitat types across all patches \nfor patches experiencing the target habitat conditions \nduring the flow time-series", fontsize=20)

    plt.xticks(ticks=x, labels=labels, rotation=30, ha='right', fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show()

# ==========================================================
# FUNCTION: ALL MESHES IN SIMULATED DOMAIN - Horizontal stacked bar of dominant habitat types
# ==========================================================
def plot_most_prob_horizontal(
    csv_path,
    habitat_col,
    habitat_labels_dict,
    habitat_colors_dict,
    number_of_habitats,
    save_path
):

    df = pd.read_csv(csv_path)
    df[habitat_col] = df[habitat_col].astype(int)

    # -----------------------------
    # Habitat range and labels
    # -----------------------------
    habitat_range = list(range(0, number_of_habitats))
    habitat_labels_dict = habitat_labels_dict

    df = df[df[habitat_col].isin(habitat_range)]

    # -----------------------------
    # Compute percentages
    # -----------------------------
    habitat_counts = df[habitat_col].value_counts().sort_index()
    total = habitat_counts.sum()
    habitat_percentages = (habitat_counts / total) * 100
    habitat_percentages = habitat_percentages.reindex(habitat_range, fill_value=0)

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(14, 4))
    left = 0
    for i in habitat_range:
        pct = habitat_percentages[i]
        ax.barh(
            0, pct,
            left=left,
            color=habitat_colors_dict[i],
            label=f"{habitat_labels_dict[i]}: {pct:.1f}%",
            edgecolor='white'
        )
        left += pct

    # -----------------------------
    # Aesthetics
    # -----------------------------
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Percentage of the entire simulated domain", fontsize=18)
    ax.set_title("Dominant habitat type distribution", fontsize=20)
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        frameon=False,
        title="Habitat type (%)",
        fontsize=14,
        title_fontsize=16
    )
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')
    ax.tick_params(axis='x', labelsize=14, colors='#333333')

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, transparent=True)
    plt.show()

# ==========================================================
# FUNCTION: TARGET HABITAT - Horizontal stacked bar of dominant habitat types
# ==========================================================
def plot_most_prob_horizontal_target_hab(
    csv_path,
    habitat_col,
    habitat_labels_dict,
    habitat_colors_dict,
    number_of_habitats,
    habitat_in_focus,
    save_path
):

    df = pd.read_csv(csv_path)
    df[habitat_col] = df[habitat_col].astype(int)

    # -----------------------------
    # Habitat range and labels
    # -----------------------------
    habitat_range = list(range(0, number_of_habitats))
    habitat_labels_dict = habitat_labels_dict

    # -----------------------------
    # Filtering
    # -----------------------------
    focus_column = f"prob_hab_{habitat_in_focus}"

    df = df[df[focus_column] > 0]
    df = df[df[habitat_col].isin(habitat_range)]

    # -----------------------------
    # Compute percentages
    # -----------------------------
    habitat_counts = df[habitat_col].value_counts().sort_index()
    total = habitat_counts.sum()
    habitat_percentages = (habitat_counts / total) * 100
    habitat_percentages = habitat_percentages.reindex(habitat_range, fill_value=0)

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(14, 4))
    left = 0
    for i in habitat_range:
        pct = habitat_percentages[i]
        ax.barh(
            0, pct,
            left=left,
            color=habitat_colors_dict[i],
            label=f"{habitat_labels_dict[i]}: {pct:.1f}%",
            edgecolor='white'
        )
        left += pct

    # -----------------------------
    # Aesthetics
    # -----------------------------
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Percentage of patches experiencing the target habitat conditions \nduring the flow time-series", fontsize=18)
    ax.set_title("Dominant habitat type distribution", fontsize=20)
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        frameon=False,
        title="Habitat type (%)",
        fontsize=14,
        title_fontsize=16
    )
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')
    ax.tick_params(axis='x', labelsize=14, colors='#333333')

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, transparent=True)
    plt.show()

# ==========================================================
# FUNCTION: ALL MESHES IN SIMULATED DOMAIN - IN PERCENTAGE OF MESHES - Habitat availability per discharge (stacked %)
# ==========================================================
def plot_habitat_availability_per_discharge(
    csv_path,
    habitat_labels_dict,
    habitat_colors_dict,
    number_of_habitats,
    save_path
):

    df = pd.read_csv(csv_path)

    # -----------------------------
    # Habitat range and labels
    # -----------------------------
    habitat_range = list(range(0, number_of_habitats))
    habitat_labels_dict = habitat_labels_dict

    # -----------------------------
    # Extract habitat columns and discharges
    # -----------------------------
    habitat_columns = [col for col in df.columns if col.startswith("Hab_")]

    if len(habitat_columns) == 0:
        raise ValueError("No habitat columns starting with 'Hab_' found.")

    # Extract discharge values while keeping dataframe order
    discharges = [float(col.replace("Hab_", "")) for col in habitat_columns]

    # -----------------------------
    # Exclude meshes that are -1 in all habitat columns
    # -----------------------------
    valid_rows = ~(df[habitat_columns] == -1).all(axis=1)
    df_valid = df.loc[valid_rows]

    total_meshes = len(df_valid)

    habitat_data = {
        ht: [df[col][df[col] == ht].count() for col in habitat_columns]
        for ht in habitat_range
    }

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(16, 8))

    x = discharges
    bottom = np.zeros(len(x))

    for ht in habitat_range:

        values = habitat_data[ht]
        percent_values = [v / total_meshes * 100 for v in values]

        plt.bar(
            range(len(x)),
            percent_values,
            bottom=bottom,
            width=0.8,
            label=habitat_labels_dict[ht],
            color=habitat_colors_dict[ht]
        )

        # Add labels
        for i, (pv, b) in enumerate(zip(percent_values, bottom)):
            if pv > 1:  # avoid clutter for tiny values
                plt.text(
                    i,
                    b + pv / 2,
                    f"{pv:.1f}",
                    ha='center',
                    va='center',
                    fontsize=15,
                    color='black'
                )

        bottom += np.array(percent_values)

    # -----------------------------
    # Formatting
    # -----------------------------
    plt.xticks(
        range(len(x)),
        [f"{val:g}" for val in x],
        rotation=90,
        fontsize=16
    )

    plt.yticks(fontsize=16)

    plt.xlabel("Discharge [m³/s]", fontsize=18, labelpad=15)
    plt.ylabel("Habitat availability\n[% of the simulated domain]", fontsize=18, labelpad=15)

    plt.title("Habitat availability per discharge", fontsize=20, pad=20)

    plt.legend(
        title="Habitat Type",
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=14,
        title_fontsize=16
    )

    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

    plt.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()

    plt.savefig(save_path, dpi=600)

    plt.show()

# ==========================================================
# FUNCTION: TARGET HABITAT - IN PERCENTAGE OF MESHES - Habitat availability per discharge (stacked %)
# ==========================================================
def plot_habitat_availability_per_discharge_target_hab(
    csv_path,
    habitat_labels_dict,
    habitat_colors_dict,
    number_of_habitats,
    target_habitat,
    save_path
):

    df = pd.read_csv(csv_path)

    # -----------------------------
    # Habitat range and labels
    # -----------------------------
    habitat_range = list(range(0, number_of_habitats))
    habitat_labels_dict = habitat_labels_dict

    # -----------------------------
    # Extract habitat columns
    # -----------------------------
    habitat_columns = [col for col in df.columns if col.startswith("Hab_")]

    if len(habitat_columns) == 0:
        raise ValueError("No habitat columns starting with 'Hab_' found.")

    # Extract discharge values
    discharges = [float(col.replace("Hab_", "")) for col in habitat_columns]

    # -----------------------------
    # Remove meshes that are -1 everywhere
    # -----------------------------
    valid_rows = ~(df[habitat_columns] == -1).all(axis=1)
    df_valid = df.loc[valid_rows]

    # -----------------------------
    # Keep only meshes that contain the target habitat
    # at least once in the time series
    # -----------------------------
    has_target = (df_valid[habitat_columns] == target_habitat).any(axis=1)
    df_valid = df_valid.loc[has_target]

    total_meshes = len(df_valid)

    # -----------------------------
    # Compute habitat counts
    # -----------------------------
    habitat_data = {
        ht: [df_valid[col][df_valid[col] == ht].count() for col in habitat_columns]
        for ht in habitat_range
    }

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(16, 8))

    x = discharges
    bottom = np.zeros(len(x))

    for ht in habitat_range:

        values = habitat_data[ht]
        percent_values = [v / total_meshes * 100 for v in values]

        plt.bar(
            range(len(x)),
            percent_values,
            bottom=bottom,
            width=0.8,
            label=habitat_labels_dict[ht],
            color=habitat_colors_dict[ht]
        )

        # Add labels
        for i, (pv, b) in enumerate(zip(percent_values, bottom)):
            if pv > 1:  # avoid clutter for tiny values
                plt.text(
                    i,
                    b + pv / 2,
                    f"{pv:.1f}",
                    ha='center',
                    va='center',
                    fontsize=15,
                    color='black'
                )

        bottom += np.array(percent_values)

    # -----------------------------
    # Formatting
    # -----------------------------
    plt.xticks(
        range(len(x)),
        [f"{val:g}" for val in x],
        rotation=90,
        fontsize=16
    )

    plt.yticks(fontsize=16)

    plt.xlabel("Discharge [m³/s]", fontsize=18, labelpad=15)
    plt.ylabel("Habitat availability\n[% of the patches experiencing the target habitat conditions \nduring the flow time-series]", fontsize=18, labelpad=15)

    plt.title("Habitat availability per discharge", fontsize=20, pad=20)

    plt.legend(
        title="Habitat Type",
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=14,
        title_fontsize=16
    )

    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show()


# ==========================================================
# FUNCTION: TARGET HABITAT - Habitat probability for each habitat type
# Among all patches that ever experience habitat 2, what is the distribution of time spent in each habitat?
# ==========================================================
def plot_probability_distribution_target_hab(
    csv_path,
    habitat_labels_dict,
    habitat_colors_dict,
    number_of_habitats,
    target_habitat,
    save_path
):
    df = pd.read_csv(csv_path)

    # -----------------------------
    # Habitat range
    # -----------------------------
    habitat_range = list(range(0, number_of_habitats))
    habitat_labels_dict = habitat_labels_dict

    # -----------------------------
    # Probability columns
    # -----------------------------
    prob_cols = [f"prob_hab_{i}" for i in habitat_range if f"prob_hab_{i}" in df.columns]

    # -----------------------------
    # Filter patches experiencing target habitat
    # -----------------------------
    target_col = f"prob_hab_{target_habitat}"

    if target_col not in df.columns:
        raise ValueError(f"{target_col} not found in dataset")

    df = df[df[target_col] > 0]

    # -----------------------------
    # Reshape to long format
    # -----------------------------
    df_long = df[prob_cols].melt(
        var_name="habitat",
        value_name="probability"
    )

    df_long["habitat"] = df_long["habitat"].str.replace("prob_hab_", "").astype(int)

    df_long["habitat_label"] = df_long["habitat"].map(habitat_labels_dict)

    # -----------------------------
    # Color palette
    # -----------------------------
    palette = {
        habitat_labels_dict[i]: habitat_colors_dict[i]
        for i in habitat_range
    }

    label_order = [habitat_labels_dict[i] for i in habitat_range]

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(12, 8))

    sns.boxplot(
        x="habitat_label",
        y="probability",
        data=df_long,
        order=label_order,
        palette=palette,
        showfliers=False
    )

    plt.xlabel("Habitat type", fontsize=18)
    plt.ylabel("Habitat probability", fontsize=18)

    plt.title(
        f"Distribution of habitat probabilities\n"
        f"(patches experiencing habitat target {target_habitat} \nat least once during the flow time-series)", 
        fontsize=20
    )

    plt.xticks(rotation=45, ha="right", fontsize=16)
    plt.yticks(fontsize=16)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show()


def plot_histograms_target_habitat(csv_path, target_habitat, METRICS_TO_COMPUTE, save_dir):
    """
    Plot percentage histograms for selected metrics of a target habitat.

    Parameters
    ----------
    csv_path : str
        Path to the CSV containing habitat metrics per mesh.
    target_habitat : int
        Target habitat type (e.g., 2).
    METRICS_TO_COMPUTE : dict
        Dictionary of metrics with True/False values for selection.
        Keys should match the metric names: 'shift_all_daily', 'shift_targ_daily', 
        'shift_dry_daily', 'dry_max', 'desiccation_risk', 'drift_percentile', etc.
    save_dir : str
        Directory where plots will be saved.
    """
    
    df = pd.read_csv(csv_path)
    filtered_df = df[df[f'prob_hab_{target_habitat}'] > 0]

    # Colors for each metric
    colors = {
        "drift": ['#c9eac2', '#7bc77c', '#2a924b', '#00441b'],        # green
        "probability": ['#ffe1e1', '#ffb0ac', '#ff8080', '#ff4040'],  # red
        "shifts": ['#d7e6f5', '#afd1e7', '#3e8ec4', '#1663aa'],       # blue
        "desiccation": ['#fdd2a5', '#fd9243', '#df5005', '#7f2704'],  # orange
    }

    # -------------------- Habitat Probability --------------------
    bin_edges = [0, 0.25, 0.50, 0.75, 1.0]
    counts, _ = np.histogram(filtered_df[f'prob_hab_{target_habitat}'].dropna(), bins=bin_edges)
    percentages = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(5, 4))
    for i in range(len(percentages)):
        center = (bin_edges[i] + bin_edges[i+1]) / 2
        width = bin_edges[i+1] - bin_edges[i]
        ax.bar(center, percentages[i], width=width, color=colors["probability"][i], edgecolor='black', align='center')
        ax.text(center, percentages[i]+1, f"{percentages[i]:.1f}%", ha='center', va='bottom', fontsize=14)

    plt.ylim(0, 100)
    ax.set_xlabel('Habitat probability', fontsize=20)
    ax.set_ylabel('Percentage of \npatches', fontsize=20)
    ax.set_xticks(bin_edges)
    ax.set_xlim(0, 1.0)
    ax.tick_params(labelsize=20)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/histogram_prob{target_habitat}_percentage.png", dpi=600, transparent=True)
    plt.show()

    # -------------------- Drift Risk --------------------
    if METRICS_TO_COMPUTE.get("drift_max", False):
        bin_edges = [0.5, 1.5, 2.5, 3.5, 4.5]
        bin_centers = [1, 2, 3, 4]
        counts, _ = np.histogram(filtered_df[f'hab{target_habitat}_DriftPerc'], bins=bin_edges)  
        percentages = counts / counts.sum() * 100

        fig, ax = plt.subplots(figsize=(5, 4))
        for i in range(len(percentages)):
            ax.bar(bin_centers[i], percentages[i], width=1.0, color=colors["drift"][i], edgecolor='black')
            ax.text(bin_centers[i], percentages[i]+1, f"{percentages[i]:.1f}%", ha='center', va='bottom', fontsize=14)

        plt.ylim(0, 100)
        ax.set_xlabel('Drift risk', fontsize=20)
        ax.set_ylabel('Percentage of \npatches', fontsize=20)
        ax.set_xticks(bin_centers)
        ax.set_xlim(0.5, 4.5)
        ax.tick_params(labelsize=20)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/histogram_drift{target_habitat}_percentage.png", dpi=600, transparent=True)
        plt.show()


    # -------------------- Habitat Shifts --------------------
    if METRICS_TO_COMPUTE.get("shift_targ_daily", False):
        
        shift_col = f'hab{target_habitat}_shift_targ_daily'
        unique_vals = np.sort(filtered_df[shift_col].dropna().unique())
        
        if len(unique_vals) > 4:
            # More than 4 values → split into 4 equal integer bins from 0 to max+1
            max_val = int(filtered_df[shift_col].max())
            bin_edges = np.linspace(0, max_val+1, 5, dtype=int)
        else:
            # 4 or fewer values → bins centered on the unique values
            bin_edges = np.concatenate(([unique_vals[0]-0.5], unique_vals + 0.5))
        
        # Histogram counts
        counts = []
        for i in range(len(bin_edges)-1):
            lower, upper = bin_edges[i], bin_edges[i+1]
            if i == 0:
                count = filtered_df[(filtered_df[shift_col] >= lower) & (filtered_df[shift_col] <= upper)].shape[0]
            else:
                count = filtered_df[(filtered_df[shift_col] > lower) & (filtered_df[shift_col] <= upper)].shape[0]
            counts.append(count)
        
        percentages = np.array(counts) / np.sum(counts) * 100

        # Plot
        fig, ax = plt.subplots(figsize=(5, 4))
        x = np.arange(len(percentages))
        for i in range(len(percentages)):
            ax.bar(x[i], percentages[i], width=1.0, color=colors["shifts"][i], edgecolor='black', linewidth=1)
            if percentages[i] > 2:
                ax.text(x[i], percentages[i]+1, f"{percentages[i]:.1f}%", ha='center', va='bottom', fontsize=14)
        
        ax.set_ylim(0, 100)
        ax.set_xlabel('Habitat shifts', fontsize=20)
        ax.set_ylabel('Percentage of \npatches', fontsize=20)
        
        # Set x-axis labels
        if len(unique_vals) > 4:
            ax.set_xticks(x)
            ax.set_xticklabels([f"{bin_edges[i]}-{bin_edges[i+1]-1}" for i in range(len(bin_edges)-1)], rotation=45, ha='right')
        else:
            ax.set_xticks(x)
            ax.set_xticklabels([str(int(val)) for val in unique_vals], rotation=45, ha='right')
        
        ax.tick_params(labelsize=20)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/histogram_shifts{target_habitat}_percentage.png", dpi=600, transparent=True)
        plt.show()

    # -------------------- Desiccation Risk --------------------
    if METRICS_TO_COMPUTE.get("desiccation_risk", False):
        bin_edges = [0.5, 1.5, 2.5, 3.5, 4.5]
        bin_centers = [1, 2, 3, 4]
        counts, _ = np.histogram(filtered_df[f'hab{target_habitat}_DesicRisk'], bins=bin_edges)
        percentages = counts / counts.sum() * 100

        fig, ax = plt.subplots(figsize=(5, 4))
        for i in range(len(percentages)):
            ax.bar(bin_centers[i], percentages[i], width=1.0, color=colors["desiccation"][i], edgecolor='black', align='center')
            if percentages[i] > 2:
                ax.text(bin_centers[i], percentages[i]+1, f"{percentages[i]:.1f}%", ha='center', va='bottom', fontsize=14)

        plt.ylim(0, 100)
        ax.set_xlabel('Desiccation risk', fontsize=20)
        ax.set_ylabel('Percentage of \npatches', fontsize=20)
        ax.set_xticks(bin_centers)
        ax.set_xlim(0.5, 4.5)
        ax.tick_params(labelsize=20)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/histogram_desicc{target_habitat}_percentage.png", dpi=600, transparent=True)
        plt.show()



# ==========================================================
# MAIN
# ==========================================================
def main():
    OUTPUT_FOLDER_PLOTS = os.path.join(OUTPUT_FOLDER, "Plots")
    os.makedirs(OUTPUT_FOLDER_PLOTS, exist_ok=True)

    
    output_hist_most_prob_and_intensity = os.path.join(OUTPUT_FOLDER_PLOTS, "hist_most_prob_and_associated_intensity_simulated_zone.png")
    plot_most_prob_intensity(
        csv_path=csv_path,
        habitat_col="mostProb",
        habitat_labels_dict=habitat_labels_dict,
        habitat_colors_dict=habitat_colors_dict,
        number_of_habitats=NUMBER_OF_HABITATS,
        save_path=output_hist_most_prob_and_intensity
    )

    output_hist_most_prob_and_intensity_target_hab = os.path.join(OUTPUT_FOLDER_PLOTS, "hist_most_prob_and_associated_intensity_target_habitat.png")
    plot_most_prob_intensity_target_hab(
        csv_path=csv_path,
        habitat_col="mostProb",
        habitat_labels_dict=habitat_labels_dict,
        habitat_colors_dict=habitat_colors_dict,
        number_of_habitats=NUMBER_OF_HABITATS,
        habitat_in_focus=HABITAT_TARGETS,
        save_path=output_hist_most_prob_and_intensity_target_hab
    )
    # ---- Second plot (percentage bar chart)
    output_percentages = os.path.join(OUTPUT_FOLDER_PLOTS, "most_prob_percentages_simulated_zone.png")
    plot_most_prob_percentages(
        csv_path=csv_path,
        habitat_col="mostProb",
        habitat_labels_dict=habitat_labels_dict,
        habitat_colors_dict=habitat_colors_dict,
        number_of_habitats=NUMBER_OF_HABITATS,
        save_path=output_percentages
    )

    output_percentages_target_hab = os.path.join(OUTPUT_FOLDER_PLOTS, "most_prob_percentages_target_hab.png")
    plot_most_prob_percentages_target_hab(
        csv_path=csv_path,
        habitat_col="mostProb",
        habitat_labels_dict=habitat_labels_dict,
        habitat_colors_dict=habitat_colors_dict,
        number_of_habitats=NUMBER_OF_HABITATS,
        habitat_in_focus=HABITAT_TARGETS,
        save_path=output_percentages_target_hab
    )

    # ---- Third plot: horizontal stacked bar
    output_horizontal = os.path.join(OUTPUT_FOLDER_PLOTS, "most_prob_horizontal_simulated_zone.png")
    plot_most_prob_horizontal(
        csv_path=csv_path,
        habitat_col="mostProb",
        habitat_labels_dict=habitat_labels_dict,
        habitat_colors_dict=habitat_colors_dict,
        number_of_habitats=NUMBER_OF_HABITATS,
        save_path=output_horizontal
    )
    
    output_horizontal_target_hab = os.path.join(OUTPUT_FOLDER_PLOTS, "most_prob_horizontal_target_hab.png")
    plot_most_prob_horizontal_target_hab(
        csv_path=csv_path,
        habitat_col="mostProb",
        habitat_labels_dict=habitat_labels_dict,
        habitat_colors_dict=habitat_colors_dict,
        number_of_habitats=NUMBER_OF_HABITATS,
        habitat_in_focus=HABITAT_TARGETS,
        save_path=output_horizontal_target_hab
    )

    # ---- Fourth plot: horizontal stacked bar
    output_hab_discharge = os.path.join(OUTPUT_FOLDER_PLOTS,"habitat_availability_per_discharge_simulated_domain.png")
    plot_habitat_availability_per_discharge(
    csv_path=csv_path_static,
    habitat_labels_dict=habitat_labels_dict,
    habitat_colors_dict=habitat_colors_dict,
    number_of_habitats=NUMBER_OF_HABITATS,
    save_path=output_hab_discharge
    )
    
    output_hab_discharge_target_hab = os.path.join(OUTPUT_FOLDER_PLOTS,"habitat_availability_per_discharge_target_hab.png")
    plot_habitat_availability_per_discharge_target_hab(
        csv_path=csv_path_static,
        habitat_labels_dict=habitat_labels_dict,
        habitat_colors_dict=habitat_colors_dict,
        number_of_habitats=NUMBER_OF_HABITATS,
        target_habitat=HABITAT_TARGETS,
        save_path=output_hab_discharge_target_hab
    )

    output_hab_prob_distrib_target_hab = os.path.join(OUTPUT_FOLDER_PLOTS,"habitat_probabilities_target_hab.png")
    plot_probability_distribution_target_hab(
        csv_path=csv_path,
        habitat_labels_dict=habitat_labels_dict,
        habitat_colors_dict=habitat_colors_dict,
        number_of_habitats=NUMBER_OF_HABITATS,
        target_habitat=HABITAT_TARGETS,
        save_path=output_hab_prob_distrib_target_hab
    )    

    plot_histograms_target_habitat(csv_path=csv_path, target_habitat=HABITAT_TARGETS, METRICS_TO_COMPUTE=METRICS_TO_COMPUTE, save_dir=OUTPUT_FOLDER_PLOTS)


if __name__ == "__main__":
    main()