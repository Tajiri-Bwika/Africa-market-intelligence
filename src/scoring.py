"""Market Opportunity Index scoring logic."""
from __future__ import annotations

import pandas as pd

SCORE_COLUMNS = [
    "population_score",
    "gdp_score",
    "gdp_growth_score",
    "gdp_per_capita_score",
    "population_growth_score",
]

MARKET_PROFILE_COLUMNS = [
    "population_latest",
    "gdp_latest",
    "gdp_per_capita_latest",
    "gdp_growth_pct",
    "population_growth_pct",
]


def percentile_score(series: pd.Series) -> pd.Series:
    """Convert a numeric series into a 0-100 percentile score."""
    return series.rank(pct=True, na_option="bottom") * 100


def add_scores(features: pd.DataFrame) -> pd.DataFrame:
    """Add percentile scores for each opportunity factor."""
    scored = features.copy()
    scored["population_score"] = percentile_score(scored["population_latest"])
    scored["gdp_score"] = percentile_score(scored["gdp_latest"])
    scored["gdp_growth_score"] = percentile_score(scored["gdp_growth_pct"])
    scored["gdp_per_capita_score"] = percentile_score(scored["gdp_per_capita_latest"])
    scored["population_growth_score"] = percentile_score(scored["population_growth_pct"])
    return scored


def validate_weights(weights: dict[str, float]) -> None:
    """Validate Market Opportunity Index weights."""
    missing = [col for col in SCORE_COLUMNS if col not in weights]
    if missing:
        raise ValueError(f"Missing weights: {missing}")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1. Current total: {total:.4f}")


def calculate_market_index(features: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Calculate the index after excluding incomplete market profiles."""
    validate_weights(weights)
    eligible = features.dropna(subset=MARKET_PROFILE_COLUMNS).copy()
    scored = add_scores(eligible)
    scored["market_opportunity_index"] = sum(scored[col] * weight for col, weight in weights.items())
    scored = scored.sort_values("market_opportunity_index", ascending=False).reset_index(drop=True)
    scored.insert(0, "rank", scored.index + 1)
    return scored


def format_rankings(scored: pd.DataFrame) -> pd.DataFrame:
    """Return presentation-ready ranking columns."""
    cols = [
        "rank",
        "Country",
        "Region",
        "market_opportunity_index",
        "population_latest",
        "gdp_latest",
        "gdp_per_capita_latest",
        "gdp_growth_pct",
        "population_growth_pct",
    ]
    return scored[cols].copy()
