import pandas as pd
from src.features import build_country_features, build_region_summary


def test_build_country_features_calculates_growth():
    df = pd.DataFrame(
        {
            "Year": [2000, 2022],
            "Country": ["Kenya", "Kenya"],
            "Region": ["East Africa", "East Africa"],
            "Population": [100, 200],
            "GDP_USD": [1000, 3000],
        }
    )
    features = build_country_features(df)
    assert features.loc[0, "population_growth_pct"] == 100
    assert features.loc[0, "gdp_growth_pct"] == 200
    assert features.loc[0, "gdp_per_capita_latest"] == 15


def test_build_region_summary_aggregates_market_scale():
    features = pd.DataFrame(
        {
            "Country": ["A", "B"],
            "Region": ["East Africa", "East Africa"],
            "population_latest": [100, 300],
            "gdp_latest": [1000, 3000],
            "gdp_growth_pct": [10, 20],
            "population_growth_pct": [5, 15],
            "gdp_per_capita_latest": [10, 10],
        }
    )
    summary = build_region_summary(features)
    assert summary.loc[0, "countries"] == 2
    assert summary.loc[0, "population_latest"] == 400
    assert summary.loc[0, "gdp_latest"] == 4000
    assert summary.loc[0, "regional_gdp_per_capita"] == 10


def test_build_country_features_excludes_incomplete_gdp_profile():
    df = pd.DataFrame(
        {
            "Year": [2000, 2022, 2000, 2022],
            "Country": ["Complete", "Complete", "Incomplete", "Incomplete"],
            "Region": ["East Africa"] * 4,
            "Population": [100, 200, 100, 200],
            "GDP_USD": [1000, 3000, None, 3000],
        }
    )
    features = build_country_features(df)
    assert features["Country"].tolist() == ["Complete"]
