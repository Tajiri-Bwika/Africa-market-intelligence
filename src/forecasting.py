"""Simple trend forecasting helpers for the market intelligence dashboard."""
from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression


def _forecast_metric(ts: pd.DataFrame, metric: str, end_year: int) -> pd.DataFrame:
    """Fit a linear trend and append forecast values after the latest actual year."""
    training = ts[["Year", metric]].dropna().sort_values("Year")
    if len(training) < 2:
        raise ValueError(f"At least two {metric} observations are required for forecasting.")

    latest_year = int(training["Year"].max())
    if end_year <= latest_year:
        raise ValueError("Forecast end year must be later than the latest actual year.")

    model = LinearRegression()
    model.fit(training[["Year"]], training[metric])

    future = pd.DataFrame({"Year": range(latest_year, end_year + 1)})
    future[metric] = model.predict(future[["Year"]]).clip(min=0)
    future["Series"] = "Forecast"

    actual = training.copy()
    actual["Series"] = "Actual"
    return pd.concat([actual, future], ignore_index=True)


def forecast_country_timeseries(df: pd.DataFrame, country: str, end_year: int = 2030) -> dict[str, pd.DataFrame]:
    """Forecast GDP and population for one country using linear regression."""
    ts = df[df["Country"] == country].copy()
    if ts.empty:
        raise ValueError(f"Country not found: {country}")
    return {
        "GDP_USD": _forecast_metric(ts, "GDP_USD", end_year),
        "Population": _forecast_metric(ts, "Population", end_year),
    }
