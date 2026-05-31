"""PDF report generation for the Africa Market Intelligence Dashboard."""
from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, PageBreak, SimpleDocTemplate, Spacer, Table, TableStyle


def _money(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    value = float(value)
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.1f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.1f}M"
    return f"${value:,.0f}"


def _number(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):,.0f}"


def _pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):,.1f}%"


def _score(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.1f}"


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "CustomTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            alignment=TA_CENTER,
            spaceAfter=12,
            textColor=colors.HexColor("#1f2937"),
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4b5563"),
            spaceAfter=18,
        ),
        "h2": ParagraphStyle(
            "SectionHeader",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#111827"),
            spaceBefore=10,
            spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#111827"),
        ),
    }


def _table(data: list[list[Any]], col_widths: list[float] | None = None) -> Table:
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def build_market_report_pdf(
    rankings: pd.DataFrame,
    region_summary: pd.DataFrame,
    weights: dict[str, float],
    strategy_name: str,
    validation: dict[str, Any],
    top_n: int = 10,
) -> bytes:
    """Create an executive PDF report for market rankings."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.45 * inch,
        leftMargin=0.45 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
        title="Africa Market Intelligence Report",
    )
    styles = _styles()
    story: list[Any] = []

    top = rankings.head(top_n).copy()
    best_country = str(top.iloc[0]["Country"]) if not top.empty else ""

    story.append(Paragraph("Africa Market Intelligence Report", styles["title"]))
    story.append(
        Paragraph(
            f"Strategy preset: {strategy_name} | CSV-powered Market Opportunity Index | Generated from local dataset",
            styles["subtitle"],
        )
    )

    story.append(Paragraph("Executive Summary", styles["h2"]))
    story.append(
        Paragraph(
            f"This report ranks African markets using a weighted index across population size, GDP size, "
            f"GDP growth, GDP per capita, and population growth. Under the {strategy_name} strategy, "
            f"the leading market is {best_country}. The ranking supports early-stage market screening, "
            f"country comparison, and expansion prioritization.",
            styles["body"],
        )
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph(f"Top {min(top_n, len(top))} Market Opportunities", styles["h2"]))
    ranking_rows = [["Rank", "Country", "Region", "Score", "GDP", "Population", "GDP Growth", "Pop Growth"]]
    for _, row in top.iterrows():
        ranking_rows.append(
            [
                int(row["rank"]),
                row["Country"],
                row["Region"],
                _score(row["market_opportunity_index"]),
                _money(row["gdp_latest"]),
                _number(row["population_latest"]),
                _pct(row["gdp_growth_pct"]),
                _pct(row["population_growth_pct"]),
            ]
        )
    story.append(
        _table(
            ranking_rows,
            [0.35 * inch, 1.0 * inch, 0.85 * inch, 0.5 * inch, 0.75 * inch, 0.85 * inch, 0.65 * inch, 0.65 * inch],
        )
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("Strategy Weights", styles["h2"]))
    weight_rows = [["Factor", "Weight"]]
    for key, value in weights.items():
        factor = key.replace("_score", "").replace("_", " ").title()
        weight_rows.append([factor, f"{value:.0%}"])
    story.append(_table(weight_rows, [2.7 * inch, 1.0 * inch]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Screened-Market Regional Snapshot", styles["h2"]))
    story.append(
        Paragraph(
            "Regional figures include only markets with complete, comparable profiles in the ranking dataset.",
            styles["body"],
        )
    )
    story.append(Spacer(1, 6))
    region_rows = [["Region", "GDP", "Population", "GDP per Capita"]]
    for _, row in region_summary.iterrows():
        region_rows.append(
            [
                row["Region"],
                _money(row["gdp_latest"]),
                _number(row["population_latest"]),
                _money(row.get("gdp_per_capita_latest", row.get("regional_gdp_per_capita"))),
            ]
        )
    story.append(_table(region_rows, [1.2 * inch, 1.1 * inch, 1.2 * inch, 1.0 * inch]))

    story.append(PageBreak())
    story.append(Paragraph("Methodology", styles["h2"]))
    story.append(
        Paragraph(
            "Each indicator is converted into a percentile score from 0 to 100. The final Market Opportunity Index "
            "is the weighted sum of the selected strategy scores. Higher scores indicate stronger relative opportunity "
            "within the available dataset.",
            styles["body"],
        )
    )
    story.append(Spacer(1, 8))
    validation_rows = [["Validation Metric", "Value"]]
    for key, value in validation.items():
        validation_rows.append([str(key).replace("_", " ").title(), str(value)])
    story.append(_table(validation_rows, [2.5 * inch, 2.0 * inch]))

    doc.build(story)
    return buffer.getvalue()


def build_country_brief_pdf(
    country_row: pd.Series,
    rankings: pd.DataFrame,
    timeseries: pd.DataFrame,
    weights: dict[str, float],
    strategy_name: str,
    flags: list[str] | None = None,
) -> bytes:
    """Create a single-country market expansion brief PDF."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.45 * inch,
        leftMargin=0.45 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
        title=f"{country_row['Country']} Market Expansion Brief",
    )
    styles = _styles()
    story: list[Any] = []
    flags = flags or []

    country = country_row["Country"]
    region = country_row["Region"]
    score = _score(country_row["market_opportunity_index"])
    rank = int(country_row["rank"])
    total = len(rankings)

    story.append(Paragraph(f"{country} Market Expansion Brief", styles["title"]))
    story.append(
        Paragraph(
            f"Strategy preset: {strategy_name} | Region: {region} | Rank: {rank} of {total} | Score: {score}/100",
            styles["subtitle"],
        )
    )

    story.append(Paragraph("Executive Recommendation", styles["h2"]))
    if rank <= max(5, round(total * 0.15)):
        recommendation = "Priority market. Strong candidate for deeper market research, partner mapping, and go-to-market planning."
    elif rank <= max(15, round(total * 0.35)):
        recommendation = "Watchlist market. Solid opportunity profile, but expansion planning should compare this country against stronger peers."
    else:
        recommendation = "Selective market. Use targeted entry logic and validate demand before committing resources."
    story.append(Paragraph(recommendation, styles["body"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Market Scorecard", styles["h2"]))
    scorecard = [
        ["Metric", "Value"],
        ["Market Opportunity Index", f"{score}/100"],
        ["Rank", f"{rank} of {total}"],
        ["Region", region],
        ["Latest GDP", _money(country_row["gdp_latest"])],
        ["Latest Population", _number(country_row["population_latest"])],
        ["GDP per Capita", _money(country_row.get("gdp_per_capita_latest"))],
        ["GDP Growth", _pct(country_row["gdp_growth_pct"])],
        ["Population Growth", _pct(country_row["population_growth_pct"])],
    ]
    if "segment" in country_row.index:
        scorecard.insert(4, ["Segment", str(country_row["segment"])])
    story.append(_table(scorecard, [2.3 * inch, 2.6 * inch]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Strategic Interpretation", styles["h2"]))
    story.append(
        Paragraph(
            f"{country} ranks #{rank} under the {strategy_name} strategy. The profile combines economic scale, "
            f"demographic depth, income proxy, and growth momentum. This brief supports first-pass screening, "
            f"market sizing conversations, and shortlisting for deeper research.",
            styles["body"],
        )
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("Recent Time-Series Snapshot", styles["h2"]))
    ts = timeseries.dropna(subset=["GDP_USD", "Population"]).sort_values("Year").tail(6)
    ts_rows = [["Year", "GDP", "Population", "GDP per Capita"]]
    for _, row in ts.iterrows():
        gdp_pc = row.get("GDP_per_capita")
        if pd.isna(gdp_pc) and pd.notna(row.get("GDP_USD")) and pd.notna(row.get("Population")) and row.get("Population"):
            gdp_pc = row["GDP_USD"] / row["Population"]
        ts_rows.append([int(row["Year"]), _money(row.get("GDP_USD")), _number(row.get("Population")), _money(gdp_pc)])
    story.append(_table(ts_rows, [0.7 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Risk Flags", styles["h2"]))
    if flags:
        for flag in flags:
            story.append(Paragraph(f"- {flag}", styles["body"]))
    else:
        story.append(Paragraph("No major data-driven flags detected from the current scoring dataset.", styles["body"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Strategy Weights Used", styles["h2"]))
    weight_rows = [["Factor", "Weight"]]
    for key, value in weights.items():
        factor = key.replace("_score", "").replace("_", " ").title()
        weight_rows.append([factor, f"{value:.0%}"])
    story.append(_table(weight_rows, [2.7 * inch, 1.0 * inch]))

    doc.build(story)
    return buffer.getvalue()
