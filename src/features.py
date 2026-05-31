"""Feature engineering for market intelligence metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def safe_growth(start: float, end: float) -> float:
    """Calculate percentage growth while handling missing and zero values."""
    if pd.isna(start) or pd.isna(end) or start == 0:
        return np.nan
    return ((end - start) / start) * 100


def build_country_features(df: pd.DataFrame, start_year: int | None = None, end_year: int | None = None) -> pd.DataFrame:
    """Build complete, comparable one-row-per-country market profiles."""
    if start_year is None:
        start_year = int(df["Year"].min())
    if end_year is None:
        end_year = int(df["Year"].max())

    start = df[df["Year"] == start_year][["Country", "Region", "Population", "GDP_USD"]].rename(
        columns={"Population": "population_start", "GDP_USD": "gdp_start"}
    )
    end = df[df["Year"] == end_year][["Country", "Region", "Population", "GDP_USD"]].rename(
        columns={"Population": "population_latest", "GDP_USD": "gdp_latest"}
    )

    features = end.merge(start[["Country", "population_start", "gdp_start"]], on="Country", how="left")
    features["gdp_per_capita_latest"] = features["gdp_latest"] / features["population_latest"]
    features["gdp_growth_pct"] = features.apply(lambda row: safe_growth(row["gdp_start"], row["gdp_latest"]), axis=1)
    features["population_growth_pct"] = features.apply(
        lambda row: safe_growth(row["population_start"], row["population_latest"]), axis=1
    )
    features["analysis_start_year"] = start_year
    features["analysis_end_year"] = end_year
    required_metrics = [
        "population_start",
        "population_latest",
        "gdp_start",
        "gdp_latest",
        "gdp_per_capita_latest",
        "gdp_growth_pct",
        "population_growth_pct",
    ]
    features = features.replace([np.inf, -np.inf], np.nan).dropna(subset=required_metrics)
    features = features[
        (features["population_start"] > 0)
        & (features["population_latest"] > 0)
        & (features["gdp_start"] > 0)
        & (features["gdp_latest"] > 0)
    ]
    return features.reset_index(drop=True)


def build_region_summary(features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate country features to region-level summary."""
    summary = (
        features.groupby("Region", as_index=False)
        .agg(
            countries=("Country", "nunique"),
            population_latest=("population_latest", "sum"),
            gdp_latest=("gdp_latest", "sum"),
            median_gdp_growth_pct=("gdp_growth_pct", "median"),
            median_population_growth_pct=("population_growth_pct", "median"),
            avg_gdp_per_capita_latest=("gdp_per_capita_latest", "mean"),
        )
        .sort_values("gdp_latest", ascending=False)
    )
    summary["regional_gdp_per_capita"] = summary["gdp_latest"] / summary["population_latest"]
    return summary


def get_country_timeseries(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """Return a single country's annual time series with GDP per capita."""
    out = df[df["Country"] == country].dropna(subset=["GDP_USD", "Population"]).copy()
    out["GDP_per_capita"] = out["GDP_USD"] / out["Population"]
    return out.sort_values("Year")
