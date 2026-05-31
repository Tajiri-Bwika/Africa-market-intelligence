"""Configuration for the Africa Market Intelligence Dashboard."""

DATA_PATH = "data/Data_Africa.csv"

REQUIRED_COLUMNS = [
    "Year",
    "Country",
    "Region",
    "Population",
    "GDP_USD",
]

STRATEGY_PRESETS = {
    "Balanced Investor": {
        "population_score": 0.25,
        "gdp_score": 0.25,
        "gdp_growth_score": 0.25,
        "gdp_per_capita_score": 0.15,
        "population_growth_score": 0.10,
    },
    "Education Expansion": {
        "population_score": 0.30,
        "gdp_score": 0.15,
        "gdp_growth_score": 0.20,
        "gdp_per_capita_score": 0.10,
        "population_growth_score": 0.25,
    },
    "Retail Expansion": {
        "population_score": 0.35,
        "gdp_score": 0.25,
        "gdp_growth_score": 0.15,
        "gdp_per_capita_score": 0.20,
        "population_growth_score": 0.05,
    },
    "Luxury Market": {
        "population_score": 0.10,
        "gdp_score": 0.25,
        "gdp_growth_score": 0.10,
        "gdp_per_capita_score": 0.50,
        "population_growth_score": 0.05,
    },
    "High-Growth Startup": {
        "population_score": 0.20,
        "gdp_score": 0.15,
        "gdp_growth_score": 0.40,
        "gdp_per_capita_score": 0.10,
        "population_growth_score": 0.15,
    },
}

STRATEGY_DESCRIPTIONS = {
    "Balanced Investor": "Balances market scale, spending power, and growth momentum for broad market screening.",
    "Education Expansion": "Prioritizes large, fast-growing populations where long-term education demand can compound.",
    "Retail Expansion": "Favors consumer scale and economic capacity with a secondary focus on spending power.",
    "Luxury Market": "Emphasizes GDP per capita and economic scale to surface higher-spending niche markets.",
    "High-Growth Startup": "Weights GDP momentum most heavily for teams looking for faster-moving expansion bets.",
}
