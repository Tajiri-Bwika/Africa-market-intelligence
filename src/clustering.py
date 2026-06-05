"""Country segmentation with K-Means clustering and validation."""
from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CLUSTER_FEATURES = [
    "population_latest",
    "gdp_latest",
    "gdp_per_capita_latest",
    "gdp_growth_pct",
    "population_growth_pct",
]


def _label_clusters_from_profiles(profiles: pd.DataFrame) -> dict[int, str]:
    """Generate data-driven cluster labels based on actual profile characteristics."""
    cluster_labels = {}
    for cluster_id in profiles["cluster_id"].unique():
        cluster_data = profiles[profiles["cluster_id"] == cluster_id]
        means = cluster_data[CLUSTER_FEATURES].mean()
        
        pop = means["population_latest"]
        gdp = means["gdp_latest"]
        pc = means["gdp_per_capita_latest"]
        gdp_g = means["gdp_growth_pct"]
        pop_g = means["population_growth_pct"]
        
        # Build label from relative characteristics
        size = "Large" if pop > profiles["population_latest"].quantile(0.75) else "Smaller"
        wealth = "High-income" if pc > profiles["gdp_per_capita_latest"].quantile(0.75) else "Lower-income"
        growth = "Fast-growing" if gdp_g > profiles["gdp_growth_pct"].quantile(0.75) else "Slower-growing"
        
        label = f"{size} markets, {wealth}, {growth}"
        cluster_labels[cluster_id] = label
    
    return cluster_labels


def segment_countries(
    features: pd.DataFrame, n_clusters: int = 5, random_state: int = 42
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Cluster countries and return PCA coordinates with validation metrics.
    
    Returns:
        Tuple of (segmented_df, validation_metrics dict)
        - validation_metrics includes silhouette_score, cluster_sizes, quality_assessment
    """
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

    # Calculate silhouette scores
    sil_score = silhouette_score(x_scaled, data["cluster_id"])
    sil_samples = silhouette_samples(x_scaled, data["cluster_id"])
    data["silhouette"] = sil_samples
    
    # Cluster sizes
    cluster_sizes = data["cluster_id"].value_counts().to_dict()
    
    # Data-driven labels
    profile_df = data[["cluster_id"] + CLUSTER_FEATURES].copy()
    labels = _label_clusters_from_profiles(profile_df)
    data["segment"] = data["cluster_id"].map(labels).fillna("Unclassified Market")
    
    # Quality assessment
    quality = "Good" if sil_score > 0.5 else ("Fair" if sil_score > 0.3 else "Weak")
    
    validation_metrics = {
        "silhouette_score": float(sil_score),
        "quality_assessment": quality,
        "cluster_sizes": cluster_sizes,
        "pca_explained_variance": float(sum(pca.explained_variance_ratio_)),
    }
    
    return data, validation_metrics


def elbow_analysis(
    features: pd.DataFrame, k_range: range = range(2, 8), random_state: int = 42
) -> pd.DataFrame:
    """Compute elbow and silhouette metrics for k-selection guidance."""
    data = features.dropna(subset=CLUSTER_FEATURES).copy()
    pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    x_scaled = pipeline.fit_transform(data[CLUSTER_FEATURES])
    
    results = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = model.fit_predict(x_scaled)
        inertia = model.inertia_
        sil = silhouette_score(x_scaled, labels)
        results.append({
            "k": k,
            "inertia": inertia,
            "silhouette_score": sil,
        })
    
    return pd.DataFrame(results)
