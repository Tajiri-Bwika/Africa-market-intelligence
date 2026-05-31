"""Business interpretation helpers."""
from __future__ import annotations

import pandas as pd


def country_brief(row: pd.Series) -> str:
    """Create a concise market brief from a scored country row."""
    strengths = []
    if row["population_score"] >= 75:
        strengths.append("large addressable population")
    if row["gdp_score"] >= 75:
        strengths.append("strong economic base")
    if row["gdp_growth_score"] >= 75:
        strengths.append("high GDP growth momentum")
    if row["gdp_per_capita_score"] >= 75:
        strengths.append("strong spending-power proxy")
    if row["population_growth_score"] >= 75:
        strengths.append("fast demographic expansion")

    if not strengths:
        strengths.append("balanced mid-market profile")

    return f"{row['Country']} ranks #{int(row['rank'])} with a score of {row['market_opportunity_index']:.1f}. Key signals: {', '.join(strengths)}."


def risk_flags(row: pd.Series) -> list[str]:
    """Generate simple data-driven risk flags."""
    flags = []
    if row.get("gdp_per_capita_latest", 0) < 1000:
        flags.append("Low GDP per capita suggests lower average spending power.")
    if row.get("population_growth_pct", 0) > 90:
        flags.append("Rapid population growth creates demand, but also infrastructure pressure.")
    return flags


def market_recommendation(row: pd.Series, total_markets: int) -> str:
    """Return a plain-language recommendation based on relative rank."""
    rank = int(row["rank"])
    if rank <= max(5, round(total_markets * 0.15)):
        return "Priority market: move into deeper market research, partner mapping, and go-to-market validation."
    if rank <= max(15, round(total_markets * 0.35)):
        return "Watchlist market: compare entry economics against stronger peers before committing resources."
    return "Selective market: use a focused entry thesis and validate demand before committing resources."


def build_markdown_brief(
    row: pd.Series,
    strategy_name: str,
    total_markets: int,
    flags: list[str] | None = None,
) -> str:
    """Build a portable country strategy brief in Markdown."""
    flags = flags or []
    risk_section = "\n".join(f"- {flag}" for flag in flags) or "- No major data-driven flags detected."
    segment = row.get("segment", "Not segmented")
    return f"""# {row["Country"]} Market Expansion Brief

## Executive Recommendation

{market_recommendation(row, total_markets)}

## Strategic Position

- Strategy: {strategy_name}
- Region: {row["Region"]}
- Segment: {segment}
- Rank: {int(row["rank"])} of {total_markets}
- Market Opportunity Index: {row["market_opportunity_index"]:.1f}/100

## Market Scorecard

| Metric | Value |
|---|---:|
| Latest GDP | ${row["gdp_latest"]:,.0f} |
| Latest population | {row["population_latest"]:,.0f} |
| GDP per capita | ${row["gdp_per_capita_latest"]:,.0f} |
| GDP growth | {row["gdp_growth_pct"]:.1f}% |
| Population growth | {row["population_growth_pct"]:.1f}% |

## Opportunity Signals

{country_brief(row)}

## Risk Flags

{risk_section}

## Interpretation Note

This is a first-pass screening brief based on the available local dataset. It does not include political risk,
inflation, regulation, exchange rates, trade access, or category-specific demand.
"""
