import pandas as pd
import os
os.environ["OMP_NUM_THREADS"] = "2"  # fix MKL memory leak on Windows
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np

from variables_from_config import METRICS_TO_COMPUTE, HABITAT_TARGETS

# Enter here the path to the csv file containing the metrics to cluster on (output of the main script)
csv_path_to_metrics = r"C:\Users\lecrivau\Documents\00_Research_Assistant\Toolbox\QGIS_to_share\Metrics_small_zone\Day\metrics.csv"
output_basepath = csv_path_to_metrics.replace(".csv", "_with_clusters.csv")
df = pd.read_csv(csv_path_to_metrics)

target_habitat = HABITAT_TARGETS[0]
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
    max_possible_k = min(10, n_unique_points)  # limit upper bound to unique points
    if max_possible_k < 2:
        print("⚠️ Not enough unique points for clustering.")
        df.loc[indices, f"cluster_{target_habitat}"] = 0
        return df

    # -----------------------------
    # Find optimal number of clusters
    # -----------------------------
    inertia = []
    silhouette_scores = []

    K_range = range(2, max_possible_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
        if k == 1:
            silhouette_scores.append(0)
        else:
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

    # -----------------------------
    # Plot elbow + silhouette
    # -----------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(K_range, inertia, marker='o')
    ax[0].set_title("Elbow Method")
    ax[0].set_xlabel("Number of clusters")
    ax[0].set_ylabel("Inertia")
    ax[1].plot(K_range, silhouette_scores, marker='o', color='orange')
    ax[1].set_title("Silhouette Score")
    ax[1].set_xlabel("Number of clusters")
    ax[1].set_ylabel("Score")
    plt.tight_layout()

    elbow_path = output_basepath.replace(
        ".csv",
        f"_hab{target_habitat}_elbow_silhouette.png"
    )
    plt.savefig(elbow_path, dpi=600)
    plt.close()
    print(f"📈 Elbow/Silhouette plot saved: {elbow_path}")

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
    cluster_stats = pd.DataFrame(X, columns=metrics)
    cluster_stats["cluster"] = labels
    cluster_agg = cluster_stats.groupby("cluster").agg(['mean', 'median']).round(3)
    cluster_agg.columns = [f"{metric}_{stat}" for metric, stat in cluster_agg.columns]

    stats_path = output_basepath.replace(
        ".csv",
        f"_hab{target_habitat}_cluster_stats.csv"
    )
    cluster_agg.to_csv(stats_path)
    print(f"📊 Cluster stats saved: {stats_path}")

    # -----------------------------
    # PCA projection
    # -----------------------------
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=metrics)
    loadings_path = output_basepath.replace(
        ".csv",
        f"_hab{target_habitat}_PCA_loadings.csv"
    )
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
    clusterplot_path = output_basepath.replace(
        ".csv",
        f"_hab{target_habitat}_PCA_clusters.png"
    )
    plt.savefig(clusterplot_path, dpi=600)
    plt.close()
    print(f"🖼️ PCA cluster plot saved: {clusterplot_path}")

    return df


df = perform_clustering_target_habitat(
    df,
    target_habitat,
    METRICS_TO_COMPUTE,
    output_basepath
)

df.to_csv(output_basepath, index=False)
print("✅ Clustering finished")
