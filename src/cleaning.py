"""Data loading and cleaning utilities."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the raw Africa CSV dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize data and calculate only defensible internal GDP gaps."""
    cleaned = df.copy()
    cleaned.columns = [col.strip().replace(" ", "_").replace("(", "").replace(")", "") for col in cleaned.columns]

    rename_map = {
        "Continent": "Region",
        "Population": "Population",
        "GDP_USD": "GDP_USD",
    }
    cleaned = cleaned.rename(columns=rename_map)

    country_fixes = {
        "Seyshelles": "Seychelles",
        "Union of the Comors": "Union of the Comoros",
    }
    cleaned["Country"] = cleaned["Country"].replace(country_fixes)

    cleaned["Year"] = pd.to_numeric(cleaned["Year"], errors="coerce").astype("Int64")
    cleaned["Population"] = pd.to_numeric(cleaned["Population"], errors="coerce")
    cleaned["GDP_USD"] = pd.to_numeric(cleaned["GDP_USD"], errors="coerce")

    cleaned = cleaned.dropna(subset=["Year", "Country", "Region", "Population"])
    cleaned["Year"] = cleaned["Year"].astype(int)
    cleaned["Population"] = cleaned["Population"].astype(float)
    cleaned = cleaned.sort_values(["Country", "Year"]).reset_index(drop=True)
    cleaned["GDP_USD_reported"] = cleaned["GDP_USD"].notna()
    cleaned["GDP_USD"] = cleaned.groupby("Country")["GDP_USD"].transform(
        lambda values: values.interpolate(method="linear", limit_area="inside")
    )
    cleaned["GDP_USD_derived"] = cleaned["GDP_USD"].notna() & ~cleaned["GDP_USD_reported"]
    return cleaned


def validate_data(df: pd.DataFrame, required_columns: list[str]) -> dict[str, object]:
    """Return a compact validation report."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    duplicate_rows = int(df.duplicated(subset=["Country", "Year"]).sum()) if not missing_columns else None
    return {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "country_count": int(df["Country"].nunique()) if "Country" in df.columns else None,
        "missing_columns": missing_columns,
        "duplicate_country_year_rows": duplicate_rows,
        "calculated_gdp_rows": int(df["GDP_USD_derived"].sum()) if "GDP_USD_derived" in df.columns else 0,
        "excluded_gdp_rows": int(df["GDP_USD"].isna().sum()) if "GDP_USD" in df.columns else None,
        "year_min": int(df["Year"].min()) if "Year" in df.columns and len(df) else None,
        "year_max": int(df["Year"].max()) if "Year" in df.columns and len(df) else None,
    }
