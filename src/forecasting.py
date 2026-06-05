"""Simple trend forecasting helpers for the market intelligence dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def _forecast_metric(
    ts: pd.DataFrame, metric: str, end_year: int, confidence: float = 0.95
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fit a linear trend and append forecast values with confidence intervals.
    
    Returns:
        Tuple of (forecast_df, diagnostics dict) containing:
        - forecast_df: historical + forecast with lower/upper bounds
        - diagnostics: r_squared, n_obs, years_span, volatility_pct, quality_flag
    """
    training = ts[["Year", metric]].dropna().sort_values("Year")
    if len(training) < 2:
        raise ValueError(f"At least two {metric} observations are required for forecasting.")

    latest_year = int(training["Year"].max())
    if end_year <= latest_year:
        raise ValueError("Forecast end year must be later than the latest actual year.")

    # Fit model
    X = training[["Year"]].values
    y = training[metric].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Calculate diagnostics
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    std_error = np.sqrt(ss_res / max(len(y) - 2, 1))
    years_span = int(training["Year"].max() - training["Year"].min())
    volatility_pct = (np.std(y) / np.mean(y) * 100) if np.mean(y) > 0 else 0

    # Data quality flag
    quality_flag = "High confidence" if r_squared > 0.7 and years_span >= 5 else "Low confidence"

    # Build output with confidence intervals
    actual = training.copy()
    actual["Series"] = "Actual"
    actual[f"{metric}_lower"] = np.nan
    actual[f"{metric}_upper"] = np.nan

    future = pd.DataFrame({"Year": range(latest_year + 1, end_year + 1)})
    X_future = future[["Year"]].values
    future[metric] = model.predict(X_future).clip(min=0)

    # Prediction intervals: wider for longer horizons
    t_crit = 1.96  # 95% confidence
    years_ahead = X_future[:, 0] - latest_year
    margin = t_crit * std_error * np.sqrt(1 + 1 / len(training) + years_ahead**2 / np.sum((X[:, 0] - X[:, 0].mean()) ** 2))
    future[f"{metric}_lower"] = np.maximum(0, future[metric] - margin)
    future[f"{metric}_upper"] = future[metric] + margin
    future["Series"] = "Forecast"

    result = pd.concat([actual, future], ignore_index=True)
    diagnostics = {
        "r_squared": float(r_squared),
        "n_observations": int(len(training)),
        "years_span": years_span,
        "volatility_pct": float(volatility_pct),
        "quality_flag": quality_flag,
        "std_error": float(std_error),
    }
    return result, diagnostics


def forecast_country_timeseries(
    df: pd.DataFrame, country: str, end_year: int = 2030
) -> dict[str, tuple[pd.DataFrame, dict[str, float]]]:
    """Forecast GDP and population for one country using linear regression with diagnostics.
    
    Returns dict of {metric: (forecast_df, diagnostics)}.
    """
    ts = df[df["Country"] == country].copy()
    if ts.empty:
        raise ValueError(f"Country not found: {country}")
    return {
        "GDP_USD": _forecast_metric(ts, "GDP_USD", end_year),
        "Population": _forecast_metric(ts, "Population", end_year),
    }
