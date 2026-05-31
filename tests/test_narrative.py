import pandas as pd

from src.narrative import build_markdown_brief


def test_build_markdown_brief_includes_country_position_and_flags():
    row = pd.Series(
        {
            "Country": "Kenya",
            "Region": "East Africa",
            "segment": "Growth Challengers",
            "rank": 3,
            "market_opportunity_index": 82.5,
            "gdp_latest": 1000,
            "population_latest": 200,
            "gdp_per_capita_latest": 5,
            "gdp_growth_pct": 10,
            "population_growth_pct": 20,
            "population_score": 80,
            "gdp_score": 70,
            "gdp_growth_score": 90,
            "gdp_per_capita_score": 50,
            "population_growth_score": 75,
        }
    )
    brief = build_markdown_brief(row, "Balanced Investor", total_markets=58, flags=["Example flag."])
    assert "# Kenya Market Expansion Brief" in brief
    assert "Rank: 3 of 58" in brief
    assert "Example flag." in brief
