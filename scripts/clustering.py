import pandas as pd
import os
os.environ["OMP_NUM_THREADS"] = "2"  # fix MKL memory leak on Windows
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
import geopandas as gpd

from variables_from_config import METRICS_TO_COMPUTE, HABITAT_TARGETS, SHP_ID_COLNAME, FINAL_CSV_PATH, FINAL_SHP_PATH, OUTPUT_FOLDER_TIME
csv_path_to_metrics = FINAL_CSV_PATH
shp_path_to_metrics = FINAL_SHP_PATH

clustering_dir = os.path.join(OUTPUT_FOLDER_TIME, "Clustering")
os.makedirs(clustering_dir, exist_ok=True)

output_basepath = os.path.join(clustering_dir, os.path.basename(csv_path_to_metrics).replace(".csv", "_with_clusters.csv"))
df = pd.read_csv(csv_path_to_metrics)

target_habitat = HABITAT_TARGETS[0]
short_basepath = os.path.join(clustering_dir, f"hab{target_habitat}")
# Initialize cluster column
df[f"cluster_{target_habitat}"] = -1


def perform_clustering_target_habitat(
        df,
        target_habitat,
        METRICS_TO_COMPUTE,
        output_basepath):

    # -----------------------------
    # Filter rows where habitat occurs
    # -----------------------------
    prob_col = f"prob_hab_{target_habitat}"
    filtered = df[df[prob_col] > 0].copy()

    # -----------------------------
    # Select metrics dynamically
    # -----------------------------
    metrics = [prob_col]  # probability always included
    if METRICS_TO_COMPUTE.get("drift_percentile", False):
        metrics.append(f"hab{target_habitat}_DriftPerc")
    if METRICS_TO_COMPUTE.get("shift_targ_daily", False):
        metrics.append(f"hab{target_habitat}_shift_targ_daily")
    if METRICS_TO_COMPUTE.get("desiccation_risk", False):
        metrics.append(f"hab{target_habitat}_DesicRisk")

    print(f"Metrics used for clustering: {metrics}")

    # Prepare matrix
    X = filtered[metrics].dropna()
    indices = X.index

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # Determine max possible clusters
    # -----------------------------
    n_unique_points = np.unique(X_scaled, axis=0).shape[0]
    max_possible_k = min(9, n_unique_points)  # limit upper bound to unique points
    if max_possible_k < 2:
        print("⚠️ Not enough unique points for clustering.")
        df.loc[indices, f"cluster_{target_habitat}"] = 0
        return df

    # -----------------------------
    # Find optimal number of clusters
    # -----------------------------
    silhouette_scores = []

    K_range = range(2, max_possible_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
        kmeans.fit(X_scaled)
        if k == 1:
            silhouette_scores.append(0)
        else:
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

    # -----------------------------
    # Plot silhouette
    # -----------------------------
    plt.figure(figsize=(6, 5))
    plt.plot(K_range, silhouette_scores, marker='o', color='orange')
    plt.title("Silhouette Score")
    plt.xlabel("Number of clusters")
    plt.ylabel("Score")
    plt.tight_layout()

    silhouette_path = os.path.join(os.path.dirname(short_basepath), f"hab{target_habitat}_silhouette.png")
    plt.savefig(silhouette_path, dpi=600)
    plt.close()
    print(f"📈 Silhouette plot saved: {silhouette_path}")

    # -----------------------------
    # Select best cluster number
    # -----------------------------
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters = {optimal_k}")

    # -----------------------------
    # Final clustering
    # -----------------------------
    kmeans = KMeans(n_clusters=optimal_k, random_state=0, n_init='auto')
    labels = kmeans.fit_predict(X_scaled)
    cluster_col = f"cluster_{target_habitat}"
    df.loc[indices, cluster_col] = labels

    # -----------------------------
    # Cluster statistics
    # -----------------------------
    """ 
    cluster_stats = pd.DataFrame(X, columns=metrics)
    cluster_stats["cluster"] = labels
    cluster_agg = cluster_stats.groupby("cluster").agg(['mean', 'median']).round(3)
    cluster_agg.columns = [f"{metric}_{stat}" for metric, stat in cluster_agg.columns]

    stats_path = os.path.join(os.path.dirname(short_basepath), f"hab{target_habitat}_cluster_stats.csv")
    cluster_agg.to_csv(stats_path)
    print(f"📊 Cluster stats saved: {stats_path}")
    """

    # -----------------------------
    # PCA projection
    # -----------------------------
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=metrics)
    loadings_path = os.path.join(os.path.dirname(short_basepath), f"hab{target_habitat}_PCA_loadings.csv")
    loadings.to_csv(loadings_path)
    print(f"📂 PCA loadings saved: {loadings_path}")

    df.loc[indices, f"PC1_{target_habitat}"] = components[:, 0]
    df.loc[indices, f"PC2_{target_habitat}"] = components[:, 1]

    # -----------------------------
    # Plot PCA clusters
    # -----------------------------
    plt.figure(figsize=(8, 6))
    for cluster in range(optimal_k):
        plt.scatter(
            components[labels == cluster, 0],
            components[labels == cluster, 1],
            label=f"Cluster {cluster}",
            s=20
        )
    plt.title("KMeans clustering (PCA projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    clusterplot_path = os.path.join(os.path.dirname(short_basepath), f"hab{target_habitat}_PCA_clusters.png")
    plt.savefig(clusterplot_path, dpi=600)
    plt.close()
    print(f"🖼️ PCA cluster plot saved: {clusterplot_path}")

    return df

def join_mesh_with_CSV_data(mesh_file, csv_file, output_shp_file, id_col):
    """
    Join a GeoDataFrame (mesh geometry) with a results DataFrame and save to a new shapefile.

    Parameters:
    - mesh_file: str, path to the mesh shapefile (polygons or points)
    - csv_file: str, path to the CSV file with results
    - output_shp_file: str, path to save the merged shapefile
    - id_col: str, column name to join on (default "id")
    """
    mesh_gdf = gpd.read_file(mesh_file)  # Load base mesh geometry
    results_df = pd.read_csv(csv_file)   # Load results

    # ==========================================================
    # Keep only columns that are NOT already in shapefile
    # ==========================================================
    cols_to_add = [
        col for col in results_df.columns
        if col not in mesh_gdf.columns or col == id_col
    ]

    results_df = results_df[cols_to_add]

    print(f"Columns added to shapefile: {[c for c in cols_to_add if c != id_col]}")

    # -----------------------------
    # Merge on the specified ID column
    # -----------------------------
    mesh_with_results = mesh_gdf.merge(results_df, on=id_col, how="left")

    # -----------------------------
    # Rename columns to respect shapefile 10-char limit
    # -----------------------------
    COLUMN_RENAME_MAP = {}
    for h in range(10):  # habitat types 0 to 9
        COLUMN_RENAME_MAP.update({
            f"hab{h}_shift_all_daily":        f"h{h}_sh_all",
            f"hab{h}_shift_targ_daily":       f"h{h}_sh_suit",
            f"hab{h}_shift_dry_daily":        f"h{h}_sh_dry",
            f"hab{h}_DryMax":                 f"h{h}_dryMax",
            f"hab{h}_DesicRisk":              f"h{h}_desicR",
            f"hab{h}_DriftPerc":              f"h{h}_driftP",
            f"hab{h}_DriftMax":               f"h{h}_driftM",
            f"hab{h}_dur_drift_1":            f"h{h}_dd_1",
            f"hab{h}_dur_drift_2":            f"h{h}_dd_2",
            f"hab{h}_dur_drift_3":            f"h{h}_dd_3",
            f"hab{h}_dur_drift_4":            f"h{h}_dd_4",
            "prob_hab_-1":                    f"h{h}_prob_-1",
            f"hab{h}_first_occurrence_time":  f"h{h}_first",
            f"hab{h}_nb_seq":                 f"h{h}_nb_seq",
            f"hab{h}_max_cumul_dur":          f"h{h}_dur_max",
            f"hab{h}_median_dur":             f"h{h}_dur_med",
            f"hab{h}_dur_q1":                 f"h{h}_dur_q1",
            f"hab{h}_dur_q3":                 f"h{h}_dur_q3",
            f"hab{h}_dry_nb_seq":             f"h{h}_dry_seq",
            f"hab{h}_dry_max_cumul_dur":      f"h{h}_dry_max",
            f"hab{h}_dry_median_dur":         f"h{h}_dry_med",
            f"hab{h}_dry_q1_dur":             f"h{h}_dry_q1",
            f"hab{h}_dry_q3_dur":             f"h{h}_dry_q3",
            f"hab{h}_desicRisk_median":       f"h{h}_dryMedR",
            f"hab{h}_desicRisk_q1":           f"h{h}_dry_q1R",
            f"hab{h}_desicRisk_q3":           f"h{h}_dry_q3R",
        })
    # Fields not habitat-specific
    COLUMN_RENAME_MAP.update({
        "desicRisk_median": "desicMedR",
        "desicRisk_q1":     "desic_q1R",
        "desicRisk_q3":     "desic_q3R",
    })

    # Apply only existing columns
    rename_existing = {
        old: new
        for old, new in COLUMN_RENAME_MAP.items()
        if old in mesh_with_results.columns
    }
    mesh_with_results = mesh_with_results.rename(
        columns={k: v for k, v in COLUMN_RENAME_MAP.items() if k in mesh_with_results.columns}
    )

    # -----------------------------
    # FINAL SAFETY: remove duplicate column names
    # -----------------------------
    mesh_with_results = mesh_with_results.loc[:, ~mesh_with_results.columns.duplicated()]

    # -----------------------------
    # Save shapefile
    # -----------------------------
    mesh_with_results.to_file(output_shp_file)

    print(f"✅ Merged shapefile saved to {output_shp_file}")

df = perform_clustering_target_habitat(
    df,
    target_habitat,
    METRICS_TO_COMPUTE,
    output_basepath
)

df.to_csv(output_basepath, index=False)
print("✅ Clustering finished")

join_mesh_with_CSV_data(
    mesh_file=shp_path_to_metrics,
    csv_file=output_basepath,
    output_shp_file=os.path.join(clustering_dir, os.path.basename(shp_path_to_metrics).replace(".shp", "_with_clusters.shp")),
    id_col=SHP_ID_COLNAME
)