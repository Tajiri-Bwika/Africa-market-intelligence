from __future__ import annotations

from html import escape
import io
from typing import Any

import pandas as pd
import streamlit as st

from src.charts import (
    cluster_scatter,
    forecast_line,
    ranking_bar,
    region_bar,
    scatter_opportunity,
    trend_line,
    weight_bar,
)
from src.cleaning import clean_data, load_raw_data, validate_data
from src.clustering import segment_countries
from src.config import DATA_PATH, REQUIRED_COLUMNS, STRATEGY_DESCRIPTIONS, STRATEGY_PRESETS
from src.features import build_country_features, build_region_summary, get_country_timeseries
from src.forecasting import forecast_country_timeseries
from src.narrative import build_markdown_brief, country_brief, market_recommendation, risk_flags
from src.reporting import build_country_brief_pdf, build_market_report_pdf
from src.scoring import calculate_market_index, format_rankings

st.set_page_config(
    page_title="Africa Market Intelligence",
    page_icon=":material/public:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #0f172a;
        --muted: #64748b;
        --line: #e2e8f0;
        --canvas: #f8fafc;
        --teal: #0f766e;
        --teal-dark: #115e59;
        --amber: #d97706;
    }
    .stApp { background: var(--canvas); }
    .block-container { max-width: 1480px; padding-top: 1.4rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.025em; }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.2rem 2.35rem;
        margin: 0 0 1.15rem;
        border-radius: 22px;
        color: #ffffff;
        background: linear-gradient(120deg, #0f172a 0%, #115e59 72%, #0f766e 100%);
        box-shadow: 0 16px 34px rgba(15, 23, 42, 0.12);
    }
    .hero:after {
        content: "";
        position: absolute;
        width: 300px;
        height: 300px;
        right: -85px;
        top: -125px;
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 50%;
        box-shadow: 0 0 0 42px rgba(255,255,255,0.045), 0 0 0 84px rgba(255,255,255,0.025);
    }
    .eyebrow {
        display: inline-block;
        margin-bottom: .72rem;
        color: #99f6e4;
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .16em;
        text-transform: uppercase;
    }
    .hero h1 { color: #ffffff; margin: 0; max-width: 760px; font-size: 2.55rem; line-height: 1.05; }
    .hero p { color: #dbeafe; max-width: 790px; margin: .9rem 0 0; font-size: 1.02rem; line-height: 1.62; }
    .hero-meta { color: #ccfbf1; margin-top: 1.25rem; font-size: .83rem; font-weight: 700; letter-spacing: .025em; }
    .section-kicker {
        margin: 1.6rem 0 .2rem;
        color: var(--teal);
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .14em;
        text-transform: uppercase;
    }
    .metric-card, .shortlist-card {
        min-height: 128px;
        padding: 1.05rem 1.08rem;
        border: 1px solid var(--line);
        border-radius: 15px;
        background: #ffffff;
        box-shadow: 0 8px 18px rgba(15, 23, 42, .045);
    }
    .metric-label { color: var(--muted); font-size: .72rem; font-weight: 800; letter-spacing: .09em; text-transform: uppercase; }
    .metric-value { margin-top: .36rem; color: var(--ink); font-size: 1.65rem; font-weight: 800; line-height: 1.05; }
    .metric-note { margin-top: .42rem; color: var(--muted); font-size: .79rem; line-height: 1.35; }
    .shortlist-card { min-height: 174px; }
    .shortlist-rank { color: var(--teal); font-size: .73rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }
    .shortlist-country { margin-top: .34rem; color: var(--ink); font-size: 1.33rem; font-weight: 800; }
    .shortlist-score { color: var(--teal-dark); margin-top: .55rem; font-size: 1.55rem; font-weight: 800; }
    .shortlist-meta { color: var(--muted); margin-top: .25rem; font-size: .8rem; line-height: 1.5; }
    .brief-card {
        padding: 1rem 1.05rem;
        border-left: 4px solid var(--teal);
        border-radius: 4px 12px 12px 4px;
        background: #ffffff;
        color: #334155;
        line-height: 1.58;
    }
    .sidebar-brand { color: var(--ink); font-size: 1.2rem; font-weight: 850; letter-spacing: -.03em; }
    .sidebar-brand span { color: var(--teal); }
    .sidebar-note { color: var(--muted); font-size: .79rem; line-height: 1.5; }
    div[data-testid="stMetric"] {
        padding: .82rem 1rem;
        border: 1px solid var(--line);
        border-radius: 12px;
        background: #ffffff;
    }
    div[data-testid="stPlotlyChart"] {
        border: 1px solid var(--line);
        border-radius: 15px;
        background: #ffffff;
        overflow: hidden;
    }
    .stTabs [data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid var(--line); }
    .stTabs [data-baseweb="tab"] { padding: .78rem .82rem; font-weight: 750; }
    .stDownloadButton button, .stButton button { border-radius: 10px; font-weight: 750; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_pipeline(path: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    raw = load_raw_data(path)
    cleaned = clean_data(raw)
    validation = validate_data(cleaned, REQUIRED_COLUMNS)
    features = build_country_features(cleaned)
    return raw, cleaned, validation, features


def to_csv_download(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def slugify(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")


def money(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    value = float(value)
    if abs(value) >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:,.1f}T"
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.1f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.1f}M"
    return f"${value:,.0f}"


def number(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    value = float(value)
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:,.1f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.1f}M"
    return f"{value:,.0f}"


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):,.1f}%"


def render_metric(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{escape(label)}</div>
            <div class="metric-value">{escape(value)}</div>
            <div class="metric-note">{escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_shortlist_card(row: pd.Series) -> None:
    st.markdown(
        f"""
        <div class="shortlist-card">
            <div class="shortlist-rank">Continental rank #{int(row["rank"])}</div>
            <div class="shortlist-country">{escape(str(row["Country"]))}</div>
            <div class="shortlist-score">{float(row["market_opportunity_index"]):.1f}<span style="font-size:.82rem;color:#64748b"> / 100</span></div>
            <div class="shortlist-meta">{escape(str(row["Region"]))}<br>{escape(str(row["segment"]))}<br>{money(row["gdp_latest"])} GDP | {pct(row["gdp_growth_pct"])} growth</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_rankings_table(rankings: pd.DataFrame) -> pd.DataFrame:
    table = rankings.copy()
    table["market_opportunity_index"] = table["market_opportunity_index"].round(1)
    table["population_latest"] = table["population_latest"].map(number)
    table["gdp_latest"] = table["gdp_latest"].map(money)
    table["gdp_per_capita_latest"] = table["gdp_per_capita_latest"].map(money)
    table["gdp_growth_pct"] = table["gdp_growth_pct"].map(pct)
    table["population_growth_pct"] = table["population_growth_pct"].map(pct)
    return table.rename(
        columns={
            "rank": "Rank",
            "market_opportunity_index": "Opportunity score",
            "population_latest": "Population",
            "gdp_latest": "GDP",
            "gdp_per_capita_latest": "GDP per capita",
            "gdp_growth_pct": "GDP growth",
            "population_growth_pct": "Population growth",
        }
    )


def show_chart(fig) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


raw_df, df, validation, features = load_pipeline(DATA_PATH)

with st.sidebar:
    st.markdown('<div class="sidebar-brand">Africa <span>Market Intel</span></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-note">Screen markets, compare opportunity signals, and build an expansion shortlist.</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("### Strategy lens")
    strategy = st.selectbox("Preset", list(STRATEGY_PRESETS.keys()), label_visibility="collapsed")
    if strategy is None:
        st.stop()
    st.caption(STRATEGY_DESCRIPTIONS[strategy])

    preset_weights = STRATEGY_PRESETS[strategy]
    if st.session_state.get("_weight_strategy") != strategy:
        for factor, value in preset_weights.items():
            st.session_state[f"weight_{factor}"] = int(value * 100)
        st.session_state["_weight_strategy"] = strategy

    with st.expander("Advanced weighting"):
        use_custom_weights = st.toggle("Use a custom mix", value=False)
        if use_custom_weights:
            custom_weights = {}
            for factor in preset_weights:
                label = factor.replace("_score", "").replace("_", " ").title()
                custom_weights[factor] = st.slider(label, 0, 100, key=f"weight_{factor}", format="%d%%")
            total_weight = sum(custom_weights.values())
            st.caption(f"Current total: {total_weight}%")
            if total_weight != 100:
                st.error("Custom weights must total 100%.")
                st.stop()
            weights = {factor: value / 100 for factor, value in custom_weights.items()}
            strategy_label = f"{strategy} (custom mix)"
        else:
            weights = preset_weights
            strategy_label = strategy

    st.divider()
    st.markdown("### Market filters")
    regions = sorted(features["Region"].dropna().unique())
    selected_regions = st.multiselect("Regions", regions, default=regions)
    score_floor = st.slider("Minimum opportunity score", 0, 100, 0, format="%d")
    top_n = st.slider("Countries in ranking chart", 5, 30, 15)
    st.divider()
    st.markdown(
        f'<div class="sidebar-note"><b>Decision-ready coverage</b><br>{features["Country"].nunique()} ranked markets | {features["Region"].nunique()} regions<br>{df["Year"].min()}-{df["Year"].max()} comparable horizon</div>',
        unsafe_allow_html=True,
    )

if not selected_regions:
    st.warning("Select at least one region to build a market view.")
    st.stop()

scored = calculate_market_index(features, weights)
segmented = segment_countries(scored)
all_rankings = format_rankings(segmented)
filtered = segmented[
    segmented["Region"].isin(selected_regions) & (segmented["market_opportunity_index"] >= score_floor)
].copy()
rankings = format_rankings(filtered)
region_summary = build_region_summary(features[features["Region"].isin(selected_regions)])

if filtered.empty:
    st.warning("No markets match the active filters. Lower the score floor or add another region.")
    st.stop()

top_market = filtered.iloc[0]
strategy_slug = slugify(strategy_label)

st.markdown(
    f"""
    <div class="hero">
        <div class="eyebrow">Africa Market Intelligence</div>
        <h1>Find the markets worth a closer look.</h1>
        <p>Turn a continent-wide economic dataset into an expansion shortlist. Compare market scale, growth momentum, spending-power proxies, and demographic signals through a strategy lens that fits the decision.</p>
        <div class="hero-meta">{escape(strategy_label)} &nbsp; | &nbsp; {len(filtered)} visible markets &nbsp; | &nbsp; Data through {int(df["Year"].max())}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-kicker">Executive snapshot</div>', unsafe_allow_html=True)
metric_cols = st.columns(5)
with metric_cols[0]:
    render_metric("Leading market", str(top_market["Country"]), f"Continental rank #{int(top_market['rank'])}")
with metric_cols[1]:
    render_metric("Opportunity score", f"{top_market['market_opportunity_index']:.1f}", f"Selected lens: {strategy_label}")
with metric_cols[2]:
    render_metric("Visible markets", str(len(filtered)), f"Across {len(selected_regions)} selected regions")
with metric_cols[3]:
    render_metric("Data horizon", f"{df['Year'].min()}-{df['Year'].max()}", f"{len(df):,} country-year records")
with metric_cols[4]:
    render_metric("Leading market GDP", money(top_market["gdp_latest"]), f"Latest GDP for {top_market['Country']}")

st.markdown('<div class="section-kicker">Priority shortlist</div>', unsafe_allow_html=True)
st.markdown("### Three markets to put on the first page")
st.caption("The shortlist updates with the selected strategy, regions, and opportunity-score threshold.")
shortlist_cols = st.columns(3)
for column, (_, row) in zip(shortlist_cols, filtered.head(3).iterrows()):
    with column:
        render_shortlist_card(row)

st.markdown("<br>", unsafe_allow_html=True)
ranking_tab, comparison_tab, segmentation_tab, regional_tab, forecast_tab, methodology_tab = st.tabs(
    ["Opportunity map", "Compare markets", "Segments", "Regional lens", "Forecasting", "Methodology"]
)

with ranking_tab:
    st.markdown("### Opportunity landscape")
    st.caption("Use the ranking for prioritization, then read the scale chart to understand what is driving each market's position.")
    left, right = st.columns([1.03, 1])
    with left:
        show_chart(ranking_bar(filtered, top_n=top_n))
    with right:
        show_chart(scatter_opportunity(filtered))

    st.markdown("### Full filtered ranking")
    st.dataframe(
        display_rankings_table(rankings),
        width="stretch",
        hide_index=True,
        column_config={
            "Opportunity score": st.column_config.ProgressColumn(
                "Opportunity score", help="Weighted relative opportunity index", min_value=0, max_value=100, format="%.1f"
            )
        },
    )
    csv_col, pdf_col, _ = st.columns([1, 1, 2])
    with csv_col:
        st.download_button(
            "Download ranking CSV",
            data=to_csv_download(rankings),
            file_name=f"africa_market_rankings_{strategy_slug}.csv",
            mime="text/csv",
            width="stretch",
        )
    with pdf_col:
        executive_pdf = build_market_report_pdf(
            rankings=rankings,
            region_summary=region_summary,
            weights=weights,
            strategy_name=strategy_label,
            validation=validation,
            top_n=top_n,
        )
        st.download_button(
            "Download executive PDF",
            data=executive_pdf,
            file_name=f"africa_market_intelligence_{strategy_slug}.pdf",
            mime="application/pdf",
            width="stretch",
        )

with comparison_tab:
    st.markdown("### Head-to-head market comparison")
    st.caption("Compare the relative ranking with the underlying time series before moving a country into deeper diligence.")
    countries = filtered["Country"].tolist()
    c1, c2 = st.columns(2)
    country_a = c1.selectbox("Country A", countries, index=0)
    country_b = c2.selectbox("Country B", countries, index=min(1, len(countries) - 1))
    if country_a is None or country_b is None:
        st.stop()

    row_a = segmented[segmented["Country"] == country_a].iloc[0]
    row_b = segmented[segmented["Country"] == country_b].iloc[0]

    brief_col_a, brief_col_b = st.columns(2)
    with brief_col_a:
        st.markdown(
            f'<div class="brief-card"><b>{escape(country_a)}</b><br>{escape(country_brief(row_a))}</div>',
            unsafe_allow_html=True,
        )
    with brief_col_b:
        st.markdown(
            f'<div class="brief-card"><b>{escape(country_b)}</b><br>{escape(country_brief(row_b))}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("#### Market scorecard")
    compare_df = pd.DataFrame(
        [
            {
                "Country": row["Country"],
                "Rank": int(row["rank"]),
                "Opportunity score": round(row["market_opportunity_index"], 1),
                "GDP": money(row["gdp_latest"]),
                "Population": number(row["population_latest"]),
                "GDP per capita": money(row["gdp_per_capita_latest"]),
                "GDP growth": pct(row["gdp_growth_pct"]),
                "Population growth": pct(row["population_growth_pct"]),
            }
            for row in [row_a, row_b]
        ]
    )
    st.dataframe(compare_df, width="stretch", hide_index=True)

    ts_a = get_country_timeseries(df, country_a)
    ts_b = get_country_timeseries(df, country_b)
    both_ts = pd.concat([ts_a.assign(Selected_Country=country_a), ts_b.assign(Selected_Country=country_b)])
    t1, t2 = st.columns(2)
    with t1:
        show_chart(trend_line(both_ts, "GDP_USD", "GDP trajectory"))
    with t2:
        show_chart(trend_line(both_ts, "Population", "Population trajectory"))

    st.markdown("### Build a market expansion brief")
    st.caption("Generate a portable country brief for the next strategy conversation.")
    brief_country = st.selectbox("Brief market", countries, index=0)
    if brief_country is None:
        st.stop()
    brief_row = segmented[segmented["Country"] == brief_country].iloc[0]
    brief_flags = risk_flags(brief_row)
    brief_ts = get_country_timeseries(df, brief_country)
    brief_pdf = build_country_brief_pdf(
        country_row=brief_row,
        rankings=all_rankings,
        timeseries=brief_ts,
        weights=weights,
        strategy_name=strategy_label,
        flags=brief_flags,
    )
    brief_md = build_markdown_brief(
        row=brief_row,
        strategy_name=strategy_label,
        total_markets=len(all_rankings),
        flags=brief_flags,
    )
    pdf_col, markdown_col, _ = st.columns([1, 1, 2])
    with pdf_col:
        st.download_button(
            f"Download {brief_country} PDF",
            data=brief_pdf,
            file_name=f"{slugify(brief_country)}_market_expansion_brief.pdf",
            mime="application/pdf",
            width="stretch",
        )
    with markdown_col:
        st.download_button(
            f"Download {brief_country} Markdown",
            data=brief_md.encode("utf-8"),
            file_name=f"{slugify(brief_country)}_market_expansion_brief.md",
            mime="text/markdown",
            width="stretch",
        )

    with st.expander("Review risk flags"):
        for row in [row_a, row_b]:
            flags = risk_flags(row)
            st.markdown(f"**{row['Country']}**")
            if flags:
                for flag in flags:
                    st.write(f"- {flag}")
            else:
                st.write("No major data-driven flags detected.")

with segmentation_tab:
    st.markdown("### Strategic market segments")
    st.caption("K-Means groups markets with similar economic and demographic profiles. Treat segments as orientation, not as official classifications.")
    show_chart(cluster_scatter(filtered))
    segment_table = filtered[
        ["Country", "Region", "segment", "market_opportunity_index", "gdp_growth_pct", "population_growth_pct"]
    ].sort_values(["segment", "market_opportunity_index"], ascending=[True, False])
    segment_table["market_opportunity_index"] = segment_table["market_opportunity_index"].round(1)
    segment_table["gdp_growth_pct"] = segment_table["gdp_growth_pct"].map(pct)
    segment_table["population_growth_pct"] = segment_table["population_growth_pct"].map(pct)
    st.dataframe(
        segment_table.rename(
            columns={
                "segment": "Segment",
                "market_opportunity_index": "Opportunity score",
                "gdp_growth_pct": "GDP growth",
                "population_growth_pct": "Population growth",
            }
        ),
        width="stretch",
        hide_index=True,
    )

with regional_tab:
    st.markdown("### Regional lens")
    st.caption("Compare the economic capacity and population represented by complete, decision-ready market profiles.")
    c1, c2 = st.columns(2)
    with c1:
        show_chart(region_bar(region_summary, "gdp_latest", "GDP represented in screened markets"))
    with c2:
        show_chart(region_bar(region_summary, "population_latest", "Population represented in screened markets"))

    region_table = region_summary.copy()
    region_table["population_latest"] = region_table["population_latest"].map(number)
    region_table["gdp_latest"] = region_table["gdp_latest"].map(money)
    region_table["median_gdp_growth_pct"] = region_table["median_gdp_growth_pct"].map(pct)
    region_table["median_population_growth_pct"] = region_table["median_population_growth_pct"].map(pct)
    region_table["avg_gdp_per_capita_latest"] = region_table["avg_gdp_per_capita_latest"].map(money)
    region_table["regional_gdp_per_capita"] = region_table["regional_gdp_per_capita"].map(money)
    st.dataframe(region_table, width="stretch", hide_index=True)

with forecast_tab:
    st.markdown("### Directional outlook to 2030")
    st.caption("Extend the historical trend for one market with a simple linear-regression forecast.")
    forecast_countries = [
        country for country in filtered["Country"].tolist() if df[df["Country"] == country]["GDP_USD"].notna().sum() >= 2
    ]
    if not forecast_countries:
        st.info("No visible market has enough GDP history for a forecast. Adjust the active filters to continue.")
        st.stop()
    forecast_country = st.selectbox("Forecast market", forecast_countries, index=0)
    if forecast_country is None:
        st.stop()
    forecasts = forecast_country_timeseries(df, forecast_country, end_year=2030)
    gdp_forecast = forecasts["GDP_USD"]
    population_forecast = forecasts["Population"]

    latest_year = int(df["Year"].max())
    latest_gdp_year = int(gdp_forecast[gdp_forecast["Series"] == "Actual"]["Year"].max())
    latest_population_year = int(population_forecast[population_forecast["Series"] == "Actual"]["Year"].max())
    latest_gdp = gdp_forecast[
        (gdp_forecast["Series"] == "Actual") & (gdp_forecast["Year"] == latest_gdp_year)
    ][
        "GDP_USD"
    ].iloc[0]
    future_gdp = gdp_forecast[(gdp_forecast["Series"] == "Forecast") & (gdp_forecast["Year"] == 2030)]["GDP_USD"].iloc[0]
    latest_population = population_forecast[
        (population_forecast["Series"] == "Actual") & (population_forecast["Year"] == latest_population_year)
    ]["Population"].iloc[0]
    future_population = population_forecast[
        (population_forecast["Series"] == "Forecast") & (population_forecast["Year"] == 2030)
    ]["Population"].iloc[0]

    forecast_cols = st.columns(3)
    forecast_cols[0].metric(
        "2030 GDP trend", money(future_gdp), f"{100 * (future_gdp / latest_gdp - 1):.1f}% vs {latest_gdp_year}"
    )
    forecast_cols[1].metric(
        "2030 population trend",
        number(future_population),
        f"{100 * (future_population / latest_population - 1):.1f}% vs {latest_population_year}",
    )
    forecast_cols[2].metric("Forecast horizon", "2030", f"{2030 - latest_year} projected years")

    chart_col_a, chart_col_b = st.columns(2)
    with chart_col_a:
        show_chart(forecast_line(gdp_forecast, "GDP_USD", f"{forecast_country} GDP trend"))
    with chart_col_b:
        show_chart(forecast_line(population_forecast, "Population", f"{forecast_country} population trend"))

    future_table = gdp_forecast[gdp_forecast["Series"] == "Forecast"][["Year", "GDP_USD"]].merge(
        population_forecast[population_forecast["Series"] == "Forecast"][["Year", "Population"]],
        on="Year",
    )
    future_table = future_table[future_table["Year"] > latest_year].copy()
    future_table["GDP_USD"] = future_table["GDP_USD"].map(money)
    future_table["Population"] = future_table["Population"].map(number)
    with st.expander("View projected values"):
        st.dataframe(future_table.rename(columns={"GDP_USD": "GDP"}), width="stretch", hide_index=True)

    st.info(
        "Forecasts are directional trend extensions, not economic predictions. They do not account for inflation, "
        "exchange rates, policy shifts, shocks, or category-specific demand."
    )

with methodology_tab:
    st.markdown("### How the index works")
    st.write(
        "The Market Opportunity Index converts each indicator into a 0-100 percentile score, then applies the "
        "selected strategy weights. A higher score means a stronger relative opportunity signal within this dataset."
    )
    method_col, validation_col = st.columns([1.15, 1])
    with method_col:
        show_chart(weight_bar(weights))
    with validation_col:
        st.markdown("#### Active strategy")
        st.write(f"**{strategy_label}**")
        st.write(STRATEGY_DESCRIPTIONS[strategy])
        st.markdown("#### Data quality")
        st.write(f"**{features['Country'].nunique()}** complete market profiles ranked")
        st.write(f"**{validation['calculated_gdp_rows']}** internal GDP gaps calculated from bounded observations")
        st.write(f"**{validation['duplicate_country_year_rows']}** duplicate country-year rows")

    st.markdown("#### Formula")
    st.code(
        "Market Opportunity Index = weighted sum of population, GDP, GDP growth, GDP per capita, and population growth percentile scores",
        language="text",
    )
    st.markdown("#### Important limitations")
    st.write(
        "This is a first-pass screening model. It does not include inflation, political risk, regulation, exchange "
        "rates, trade access, category demand, or consumer behavior. Use it to prioritize deeper research, not to "
        "replace it."
    )
