"""Country segmentation with K-Means clustering."""
from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CLUSTER_FEATURES = [
    "population_latest",
    "gdp_latest",
    "gdp_per_capita_latest",
    "gdp_growth_pct",
    "population_growth_pct",
]

def _label_clusters(cluster_ids, x_scaled) -> dict[int, str]:
    """Assign stable business labels from cluster profiles, not arbitrary K-Means IDs."""
    profiles = pd.DataFrame(x_scaled, columns=CLUSTER_FEATURES)
    profiles["cluster_id"] = cluster_ids
    means = profiles.groupby("cluster_id").mean()
    available = set(int(cluster_id) for cluster_id in means.index)
    labels: dict[int, str] = {}

    def assign(label: str, signal: pd.Series) -> None:
        if not available:
            return
        candidates = signal.loc[list(available)]
        cluster_id = int(candidates.idxmax())
        labels[cluster_id] = label
        available.remove(cluster_id)

    assign("Scale Markets", means["population_latest"] + means["gdp_latest"])
    assign("High-Income Niches", means["gdp_per_capita_latest"])
    assign("Demographic Boom Markets", means["population_growth_pct"])
    assign("Growth Challengers", means["gdp_growth_pct"])
    for cluster_id in available:
        labels[cluster_id] = "Developing Capacity Markets"
    return labels


def segment_countries(features: pd.DataFrame, n_clusters: int = 5, random_state: int = 42) -> pd.DataFrame:
    """Cluster countries and add PCA coordinates for visualization."""
    data = features.dropna(subset=CLUSTER_FEATURES).copy()
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
    x_scaled = pipeline.fit_transform(data[CLUSTER_FEATURES])
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    data["cluster_id"] = model.fit_predict(x_scaled)

    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(x_scaled)
    data["pca_1"] = coords[:, 0]
    data["pca_2"] = coords[:, 1]

    data["segment"] = data["cluster_id"].map(_label_clusters(data["cluster_id"], x_scaled)).fillna("Strategic Segment")
    return data
