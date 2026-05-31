import pandas as pd
from src.cleaning import clean_data, validate_data
from src.config import REQUIRED_COLUMNS


def test_clean_data_standardizes_columns():
    raw = pd.DataFrame(
        {
            "ID": [1],
            "Year": [2022],
            "Country": ["Seyshelles"],
            "Continent": ["East Africa"],
            "Population ": [1000],
            "GDP (USD)": [5000],
        }
    )
    cleaned = clean_data(raw)
    assert "Region" in cleaned.columns
    assert "Population" in cleaned.columns
    assert "GDP_USD" in cleaned.columns
    assert cleaned.loc[0, "Country"] == "Seychelles"


def test_validate_data_has_no_required_missing_after_cleaning():
    raw = pd.DataFrame(
        {
            "Year": [2022],
            "Country": ["Kenya"],
            "Continent": ["East Africa"],
            "Population ": [56000000],
            "GDP (USD)": [113000000000],
        }
    )
    cleaned = clean_data(raw)
    report = validate_data(cleaned, REQUIRED_COLUMNS)
    assert report["missing_columns"] == []


def test_clean_data_marks_unresolved_gdp_for_exclusion():
    raw = pd.DataFrame(
        {
            "Year": [2022],
            "Country": ["Kenya"],
            "Continent": ["East Africa"],
            "Population ": [56000000],
            "GDP (USD)": [None],
        }
    )
    cleaned = clean_data(raw)
    report = validate_data(cleaned, REQUIRED_COLUMNS)
    assert len(cleaned) == 1
    assert report["excluded_gdp_rows"] == 1


def test_clean_data_calculates_internal_gdp_gap():
    raw = pd.DataFrame(
        {
            "Year": [2020, 2021, 2022],
            "Country": ["Kenya", "Kenya", "Kenya"],
            "Continent": ["East Africa", "East Africa", "East Africa"],
            "Population ": [50, 51, 52],
            "GDP (USD)": [1000, None, 1400],
        }
    )
    cleaned = clean_data(raw)
    report = validate_data(cleaned, REQUIRED_COLUMNS)
    assert cleaned.loc[1, "GDP_USD"] == 1200
    assert bool(cleaned.loc[1, "GDP_USD_derived"])
    assert report["calculated_gdp_rows"] == 1
    assert report["excluded_gdp_rows"] == 0
