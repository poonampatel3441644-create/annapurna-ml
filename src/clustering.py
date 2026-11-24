"""K-Means and hierarchical clustering helpers for meal patterns."""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

CLUSTER_COLS = ["students_registered", "plates_cooked", "plates_eaten",
                "leftover_kg", "temperature_c"]


def scale_for_clustering(df, cols=CLUSTER_COLS):
    """Standardize the clustering columns (distance-based methods need this)."""
    return StandardScaler().fit_transform(df[cols])


def elbow_and_silhouette(Xc, k_values=range(2, 8)):
    """Inertia (elbow) and silhouette score for each k."""
    inertia, silhouette = [], []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xc)
        inertia.append(km.inertia_)
        silhouette.append(silhouette_score(Xc, km.labels_))
    return list(k_values), inertia, silhouette


def fit_kmeans(Xc, k=3):
    """Fit K-Means and return the cluster label of every row."""
    return KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Xc)


def pca_2d(Xc):
    """Project to 2 dimensions so clusters can be plotted."""
    return PCA(n_components=2).fit_transform(Xc)


def cluster_profiles(df, labels, cols=CLUSTER_COLS):
    """Average of each column per cluster, plus the high-waste rate."""
    out = df.assign(cluster=labels)
    return out.groupby("cluster")[cols + ["high_waste"]].mean().round(2)
