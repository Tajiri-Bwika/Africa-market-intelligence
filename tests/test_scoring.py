import pandas as pd
from src.scoring import calculate_market_index


def test_market_index_ranking_order():
    features = pd.DataFrame(
        {
            "Country": ["A", "B"],
            "Region": ["X", "X"],
            "population_latest": [100, 200],
            "gdp_latest": [1000, 2000],
            "gdp_per_capita_latest": [10, 10],
            "gdp_growth_pct": [10, 20],
            "population_growth_pct": [5, 10],
        }
    )
    weights = {
        "population_score": 0.25,
        "gdp_score": 0.25,
        "gdp_growth_score": 0.25,
        "gdp_per_capita_score": 0.15,
        "population_growth_score": 0.10,
    }
    scored = calculate_market_index(features, weights)
    assert scored.iloc[0]["Country"] == "B"
    assert scored.iloc[0]["rank"] == 1


def test_invalid_weights_raise_error():
    features = pd.DataFrame(
        {
            "Country": ["A"],
            "Region": ["X"],
            "population_latest": [100],
            "gdp_latest": [1000],
            "gdp_per_capita_latest": [10],
            "gdp_growth_pct": [10],
            "population_growth_pct": [5],
        }
    )
    weights = {
        "population_score": 1,
        "gdp_score": 1,
        "gdp_growth_score": 1,
        "gdp_per_capita_score": 1,
        "population_growth_score": 1,
    }
    try:
        calculate_market_index(features, weights)
        assert False
    except ValueError:
        assert True


def test_market_index_excludes_incomplete_profile():
    features = pd.DataFrame(
        {
            "Country": ["Complete", "Incomplete"],
            "Region": ["X", "X"],
            "population_latest": [100, 200],
            "gdp_latest": [1000, None],
            "gdp_per_capita_latest": [10, None],
            "gdp_growth_pct": [10, None],
            "population_growth_pct": [5, 10],
        }
    )
    weights = {
        "population_score": 0.25,
        "gdp_score": 0.25,
        "gdp_growth_score": 0.25,
        "gdp_per_capita_score": 0.15,
        "population_growth_score": 0.10,
    }
    scored = calculate_market_index(features, weights)
    assert scored["Country"].tolist() == ["Complete"]
