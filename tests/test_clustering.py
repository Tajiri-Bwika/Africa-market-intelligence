import pandas as pd

from src.clustering import segment_countries


def test_segment_countries_adds_profile_labels_and_coordinates():
    features = pd.DataFrame(
        {
            "Country": [f"Market {index}" for index in range(10)],
            "Region": ["Region"] * 10,
            "population_latest": [10, 12, 30, 32, 50, 52, 70, 72, 90, 92],
            "gdp_latest": [100, 110, 300, 320, 500, 520, 700, 720, 900, 920],
            "gdp_per_capita_latest": [1, 1.1, 3, 3.1, 5, 5.1, 7, 7.1, 9, 9.1],
            "gdp_growth_pct": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "population_growth_pct": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )
    segmented = segment_countries(features)
    assert {"cluster_id", "segment", "pca_1", "pca_2"}.issubset(segmented.columns)
    assert segmented["segment"].notna().all()
    assert segmented["segment"].nunique() == 5
