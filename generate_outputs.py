"""Generate portfolio-ready CSV outputs from the project pipeline."""
from src.cleaning import clean_data, load_raw_data
from src.config import DATA_PATH, STRATEGY_PRESETS
from src.features import build_country_features, build_region_summary
from src.scoring import calculate_market_index, format_rankings
from src.clustering import segment_countries


def main() -> None:
    raw = load_raw_data(DATA_PATH)
    cleaned = clean_data(raw)
    features = build_country_features(cleaned)

    for strategy_name, weights in STRATEGY_PRESETS.items():
        scored = calculate_market_index(features, weights)
        rankings = format_rankings(scored)
        filename = strategy_name.lower().replace(" ", "_")
        rankings.to_csv(f"outputs/{filename}_rankings.csv", index=False)

    balanced = calculate_market_index(features, STRATEGY_PRESETS["Balanced Investor"])
    segments = segment_countries(balanced)
    segments.to_csv("outputs/country_segments.csv", index=False)

    region_summary = build_region_summary(features)
    region_summary.to_csv("outputs/regional_summary.csv", index=False)

    print("Outputs generated in the outputs/ folder.")


if __name__ == "__main__":
    main()
