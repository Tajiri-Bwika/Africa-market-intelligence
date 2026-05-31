from src.cleaning import clean_data, load_raw_data
from src.clustering import segment_countries
from src.config import DATA_PATH, STRATEGY_PRESETS
from src.features import build_country_features, get_country_timeseries
from src.narrative import risk_flags
from src.reporting import build_country_brief_pdf
from src.scoring import calculate_market_index, format_rankings


def test_country_brief_pdf_generates_bytes():
    cleaned = clean_data(load_raw_data(DATA_PATH))
    features = build_country_features(cleaned)
    weights = STRATEGY_PRESETS["Balanced Investor"]
    scored = calculate_market_index(features, weights)
    segmented = segment_countries(scored)
    rankings = format_rankings(segmented)
    row = segmented[segmented["Country"] == "Kenya"].iloc[0]
    pdf = build_country_brief_pdf(
        country_row=row,
        rankings=rankings,
        timeseries=get_country_timeseries(cleaned, "Kenya"),
        weights=weights,
        strategy_name="Balanced Investor",
        flags=risk_flags(row),
    )
    assert isinstance(pdf, bytes)
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000
