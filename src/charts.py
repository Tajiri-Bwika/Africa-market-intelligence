"""Plotly chart builders for the Streamlit app."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REGION_COLORS = {
    "Central Africa": "#7C3AED",
    "East Africa": "#0F766E",
    "North Africa": "#D97706",
    "South Africa": "#2563EB",
    "West Africa": "#DC2626",
}

SEGMENT_COLORS = {
    "Scale Markets": "#0F766E",
    "High-Income Niches": "#7C3AED",
    "Growth Challengers": "#D97706",
    "Demographic Boom Markets": "#DC2626",
    "Developing Capacity Markets": "#2563EB",
    "Strategic Segment": "#64748B",
}


def _style_chart(
    fig: go.Figure,
    *,
    height: int = 420,
    legend_title: str | None = None,
    show_legend: bool = True,
) -> go.Figure:
    """Apply the shared executive-dashboard visual style."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=24, r=24, t=76, b=92 if show_legend else 44),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif", color="#334155", size=12),
        title=dict(font=dict(color="#0F172A", size=17), x=0.01, xanchor="left", y=0.97, yanchor="top"),
        showlegend=show_legend,
        legend=dict(
            title_text=legend_title,
            orientation="h",
            yanchor="top",
            y=-0.16,
            xanchor="left",
            x=0,
        ),
        hoverlabel=dict(bgcolor="#0F172A", font_color="#FFFFFF", bordercolor="#0F172A"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E2E8F0", zeroline=False, automargin=True, title_standoff=12)
    fig.update_yaxes(showgrid=False, zeroline=False, automargin=True, title_standoff=12)
    return fig


def ranking_bar(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    data = df.head(top_n).sort_values("market_opportunity_index")
    fig = px.bar(
        data,
        x="market_opportunity_index",
        y="Country",
        orientation="h",
        color="Region",
        color_discrete_map=REGION_COLORS,
        title=f"Top {min(top_n, len(data))} opportunity markets",
        labels={"market_opportunity_index": "Opportunity score", "Country": ""},
        hover_data={"market_opportunity_index": ":.1f", "Region": True},
    )
    fig.update_traces(marker_line_width=0, hovertemplate="<b>%{y}</b><br>Opportunity score: %{x:.1f}<extra></extra>")
    fig.update_xaxes(range=[0, 100])
    return _style_chart(fig, height=480, legend_title=None)


def trend_line(ts: pd.DataFrame, metric: str, title: str) -> go.Figure:
    color = "Selected_Country" if "Selected_Country" in ts.columns else None
    fig = px.line(
        ts,
        x="Year",
        y=metric,
        color=color,
        markers=True,
        title=title,
        color_discrete_sequence=["#0F766E", "#D97706", "#7C3AED"],
    )
    fig.update_traces(line_width=3, marker_size=6)
    return _style_chart(fig, height=360, legend_title=None)


def scatter_opportunity(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df,
        x="population_latest",
        y="gdp_latest",
        size="market_opportunity_index",
        color="Region",
        color_discrete_map=REGION_COLORS,
        hover_name="Country",
        hover_data={
            "market_opportunity_index": ":.1f",
            "population_latest": ":,.0f",
            "gdp_latest": ":$,.0f",
        },
        log_x=True,
        log_y=True,
        title="Economic scale and addressable population",
        labels={
            "population_latest": "Population (log scale)",
            "gdp_latest": "GDP in USD (log scale)",
            "market_opportunity_index": "Opportunity score",
        },
    )
    fig.update_traces(marker=dict(opacity=0.82, line=dict(width=1, color="#FFFFFF")))
    return _style_chart(fig, height=480, legend_title=None)


def cluster_scatter(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df,
        x="pca_1",
        y="pca_2",
        color="segment",
        color_discrete_map=SEGMENT_COLORS,
        hover_name="Country",
        size="market_opportunity_index" if "market_opportunity_index" in df.columns else None,
        hover_data={"market_opportunity_index": ":.1f"} if "market_opportunity_index" in df.columns else None,
        title="Country segments by economic and demographic profile",
        labels={"pca_1": "Market profile axis 1", "pca_2": "Market profile axis 2", "segment": "Segment"},
    )
    fig.update_traces(marker=dict(opacity=0.82, line=dict(width=1, color="#FFFFFF")))
    return _style_chart(fig, height=500, legend_title=None)


def region_bar(summary: pd.DataFrame, metric: str, title: str) -> go.Figure:
    data = summary.sort_values(metric)
    fig = px.bar(
        data,
        x=metric,
        y="Region",
        orientation="h",
        color="Region",
        color_discrete_map=REGION_COLORS,
        title=title,
        labels={metric: "", "Region": ""},
    )
    fig.update_traces(marker_line_width=0, showlegend=False)
    return _style_chart(fig, height=370, show_legend=False)


def forecast_line(ts: pd.DataFrame, metric: str, title: str, diagnostics: dict | None = None) -> go.Figure:
    """Plot actual values, forecast, and confidence intervals with quality badge.
    
    Args:
        ts: DataFrame with Year, metric, Series, and optional {metric}_lower/{metric}_upper
        metric: Column name for the metric to plot
        title: Chart title
        diagnostics: Optional dict with r_squared, quality_flag, n_observations
    """
    fig = go.Figure()
    
    # Separate actual and forecast
    actual = ts[ts["Series"] == "Actual"]
    forecast = ts[ts["Series"] == "Forecast"]
    
    # Actual line
    fig.add_trace(go.Scatter(
        x=actual["Year"], y=actual[metric],
        mode="lines+markers",
        name="Actual",
        line=dict(color="#0F766E", width=3),
        marker=dict(size=6),
    ))
    
    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast["Year"], y=forecast[metric],
        mode="lines+markers",
        name="Forecast",
        line=dict(color="#D97706", width=3, dash="dash"),
        marker=dict(size=6),
    ))
    
    # Confidence interval (if available)
    lower_col, upper_col = f"{metric}_lower", f"{metric}_upper"
    if lower_col in forecast.columns and upper_col in forecast.columns:
        fig.add_trace(go.Scatter(
            x=forecast["Year"], y=forecast[upper_col],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=forecast["Year"], y=forecast[lower_col],
            mode="lines",
            line=dict(width=0),
            fillcolor="rgba(217, 119, 6, 0.2)",
            fill="tonexty",
            name="95% Confidence",
            hoverinfo="skip",
        ))
    
    # Add quality badge annotation
    if diagnostics:
        quality = diagnostics.get("quality_flag", "")
        r2 = diagnostics.get("r_squared", 0)
        badge_text = f"{quality}<br>R² = {r2:.2f}"
        fig.add_annotation(
            text=badge_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor="rgba(15, 118, 110, 0.1)",
            bordercolor="#0F766E",
            borderwidth=1,
            font=dict(size=10, color="#0F766E"),
            align="left",
        )
    
    fig.update_layout(
        title_text=title,
        template="plotly_white",
        height=420,
        margin=dict(l=24, r=24, t=76, b=92),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif", color="#334155", size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E2E8F0", zeroline=False, automargin=True)
    fig.update_yaxes(showgrid=False, zeroline=False, automargin=True)
    
    return _style_chart(fig, height=420, legend_title=None)


def weight_bar(weights: dict[str, float]) -> go.Figure:
    """Show the selected strategy mix."""
    data = pd.DataFrame(
        {
            "Factor": [key.replace("_score", "").replace("_", " ").title() for key in weights],
            "Weight": [value * 100 for value in weights.values()],
        }
    ).sort_values("Weight")
    fig = px.bar(
        data,
        x="Weight",
        y="Factor",
        orientation="h",
        title="Selected strategy weighting",
        text="Weight",
        color_discrete_sequence=["#0F766E"],
        labels={"Weight": "Weight (%)", "Factor": ""},
    )
    fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside", cliponaxis=False)
    fig.update_xaxes(range=[0, max(55, float(data["Weight"].max()) + 8)])
    return _style_chart(fig, height=340, show_legend=False)


def elbow_plot(elbow_df: pd.DataFrame) -> go.Figure:
    """Plot inertia and silhouette scores for k-selection guidance."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=elbow_df["k"], y=elbow_df["inertia"],
            mode="lines+markers",
            name="Inertia",
            line=dict(color="#0F766E", width=3),
            marker=dict(size=8),
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=elbow_df["k"], y=elbow_df["silhouette_score"],
            mode="lines+markers",
            name="Silhouette",
            line=dict(color="#D97706", width=3),
            marker=dict(size=8),
        ),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="Number of clusters (k)")
    fig.update_yaxes(title_text="Inertia", secondary_y=False)
    fig.update_yaxes(title_text="Silhouette score", secondary_y=True)
    fig.update_layout(
        title_text="K-selection guidance: elbow and silhouette analysis",
        template="plotly_white",
        height=400,
        margin=dict(l=24, r=24, t=76, b=92),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif", color="#334155", size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0),
    )
    
    return fig


def cluster_profile_table(features: pd.DataFrame, segmented: pd.DataFrame) -> pd.DataFrame:
    """Generate a summary table of cluster profiles with counts and averages."""
    CLUSTER_FEATURES = [
        "population_latest",
        "gdp_latest",
        "gdp_per_capita_latest",
        "gdp_growth_pct",
        "population_growth_pct",
    ]
    
    profile = (
        segmented.groupby("segment", as_index=False)
        .agg({
            "Country": "count",
            "population_latest": "mean",
            "gdp_latest": "mean",
            "gdp_per_capita_latest": "mean",
            "gdp_growth_pct": "mean",
            "population_growth_pct": "mean",
            "silhouette": "mean",
        })
        .rename(columns={
            "Country": "Countries",
            "population_latest": "Avg Population",
            "gdp_latest": "Avg GDP",
            "gdp_per_capita_latest": "Avg GDP per Capita",
            "gdp_growth_pct": "Avg GDP Growth %",
            "population_growth_pct": "Avg Pop Growth %",
            "silhouette": "Cohesion Score",
        })
    )
    
    return profile
