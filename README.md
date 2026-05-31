# Africa Market Intelligence Dashboard

A Streamlit dashboard for first-pass African market screening. It turns the included country-year CSV into strategy-specific rankings, comparisons, segments, regional summaries, downloadable briefs, and directional 2030 forecasts.

## What It Includes

- Five strategy presets plus validated custom scoring weights
- Executive shortlist and continent-wide market ranking
- GDP and population comparison views
- Profile-driven K-Means market segments
- Regional screened-market GDP and population summaries
- Downloadable CSV, executive PDF, country PDF, and Markdown briefs
- Linear-regression GDP and population trend extensions to 2030
- Conservative missing-data handling: bounded GDP gaps are calculated and unresolved profiles are excluded

## Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

## Run Tests

```powershell
python -m pytest -q
```

## Model Scope

The Market Opportunity Index is a relative screening model based on population, GDP, GDP growth, GDP per capita, and population growth. It is designed to prioritize deeper research, not to replace diligence. GDP gaps are calculated only when bounded by reported observations; unresolved market profiles are excluded from intelligence outputs. The dataset does not include inflation, political risk, regulation, exchange rates, trade access, or category-specific demand.
