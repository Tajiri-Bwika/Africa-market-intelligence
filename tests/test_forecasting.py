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
    gdp = forecasts["GDP_USD"]
    population = forecasts["Population"]
    assert gdp[(gdp["Year"] == 2024) & (gdp["Series"] == "Forecast")]["GDP_USD"].iloc[0] == pytest.approx(1800)
    assert population[(population["Year"] == 2024) & (population["Series"] == "Forecast")][
        "Population"
    ].iloc[0] == pytest.approx(140)


def test_forecast_country_timeseries_rejects_unknown_country():
    df = pd.DataFrame({"Year": [2022], "Country": ["Kenya"], "Population": [100], "GDP_USD": [1000]})
    with pytest.raises(ValueError, match="Country not found"):
        forecast_country_timeseries(df, "Unknown")
