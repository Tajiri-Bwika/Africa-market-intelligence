import pandas as pd
import pytest

from src.forecasting import forecast_country_timeseries


def test_forecast_country_timeseries_extends_linear_trend():
    df = pd.DataFrame(
        {
            "Year": [2020, 2021, 2022],
            "Country": ["Kenya", "Kenya", "Kenya"],
            "Population": [100, 110, 120],
            "GDP_USD": [1000, 1200, 1400],
        }
    )
    forecasts = forecast_country_timeseries(df, "Kenya", end_year=2024)
    gdp_df, gdp_diag = forecasts["GDP_USD"]
    pop_df, pop_diag = forecasts["Population"]
    
    assert gdp_df[(gdp_df["Year"] == 2024) & (gdp_df["Series"] == "Forecast")]["GDP_USD"].iloc[0] == pytest.approx(1800)
    assert pop_df[(pop_df["Year"] == 2024) & (pop_df["Series"] == "Forecast")]["Population"].iloc[0] == pytest.approx(140)
    
    # Verify diagnostics
    assert "r_squared" in gdp_diag
    assert "quality_flag" in gdp_diag
    assert gdp_diag["n_observations"] == 3


def test_forecast_country_timeseries_rejects_unknown_country():
    df = pd.DataFrame({"Year": [2022], "Country": ["Kenya"], "Population": [100], "GDP_USD": [1000]})
    with pytest.raises(ValueError, match="Country not found"):
        forecast_country_timeseries(df, "Unknown")
