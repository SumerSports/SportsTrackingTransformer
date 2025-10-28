"""
Generate Results Summary

Processes model predictions and generates result summary for publication.
Usage: uv run python src/generate_results_summary.py
"""

import json
from pathlib import Path

import polars as pl

from metrics import calculate_ade

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MODELS_DIR = Path("models/best_models")
ZOO_RESULTS = MODELS_DIR / "zoo" / "best_model_results.parquet"
TRANSFORMER_RESULTS = MODELS_DIR / "transformer" / "best_model_results.parquet"


def load_results() -> pl.DataFrame:
    """Load and combine results from both models."""
    print("Loading model results...")
    results_df = pl.concat(
        [pl.read_parquet(ZOO_RESULTS), pl.read_parquet(TRANSFORMER_RESULTS)],
        how="diagonal",
    )
    # Filter to mirrored=False to avoid double-counting predictions
    # (mirrored data is just augmentation - same play, flipped horizontally)
    results_df = results_df.filter(pl.col("mirrored") == False)
    print(f"  Loaded {len(results_df):,} predictions (mirrored=False only)")
    return results_df


def calculate_results(results_df: pl.DataFrame) -> list[dict]:
    """Calculate results for all splits and events."""
    print("\nCalculating results...")

    results = []

    # Main splits
    for split in ["train", "val", "test"]:
        split_df = results_df.filter(pl.col("dataset_split") == split)

        row = {"split": split, "metric": "ade_yards"}

        for model_type in ["zoo", "transformer"]:
            model_df = split_df.filter(pl.col("model_type") == model_type)

            ade = model_df.select(
                pl.map_groups(
                    exprs=["tackle_x", "tackle_y", "tackle_x_pred", "tackle_y_pred"],
                    function=lambda ls: calculate_ade(*ls),
                    returns_scalar=True,
                )
            ).item()

            row[model_type] = round(ade, 2)

        row["improvement_pct"] = round((row["zoo"] - row["transformer"]) / row["zoo"] * 100, 1)
        row["improvement_yards"] = round(row["zoo"] - row["transformer"], 2)
        row["n_plays"] = split_df.select(pl.struct(["gameId", "playId"]).n_unique()).item()
        row["n_frames"] = split_df.select(pl.len()).item()

        results.append(row)

    # Test event breakdowns
    print("Calculating test set event breakdowns...")
    test_df = results_df.filter(pl.col("dataset_split") == "test")
    events = sorted(test_df["tackle_event"].unique().drop_nulls().to_list())

    event_results = []
    for event in events:
        # Filter to only the specific frame where the event occurred
        event_df = test_df.filter(
            (pl.col("tackle_event") == event) & (pl.col("frameId") == pl.col("tackle_frameId"))
        )

        row = {"split": f"test-{event}", "metric": "ade_yards"}

        for model_type in ["zoo", "transformer"]:
            model_df = event_df.filter(pl.col("model_type") == model_type)

            ade = model_df.select(
                pl.map_groups(
                    exprs=["tackle_x", "tackle_y", "tackle_x_pred", "tackle_y_pred"],
                    function=lambda ls: calculate_ade(*ls),
                    returns_scalar=True,
                )
            ).item()

            row[model_type] = round(ade, 2)

        row["improvement_pct"] = round((row["zoo"] - row["transformer"]) / row["zoo"] * 100, 1)
        row["improvement_yards"] = round(row["zoo"] - row["transformer"], 2)
        row["n_plays"] = event_df.select(pl.struct(["gameId", "playId"]).n_unique()).item()
        row["n_frames"] = event_df.select(pl.len()).item()

        event_results.append(row)

    # Sort event results by n_plays descending
    event_results.sort(key=lambda x: x["n_plays"], reverse=True)
    results.extend(event_results)

    return results


def main():
    """Generate results summary."""
    print("=" * 60)
    print("GENERATING RESULTS")
    print("=" * 60)

    results_df = load_results()
    results = calculate_results(results_df)

    # Save JSON
    json_path = RESULTS_DIR / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved: {json_path}")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)

    # Print test set summary
    test_row = next(r for r in results if r["split"] == "test")
    print(f"\nTest Set Overall:")
    print(f"  Zoo:         {test_row['zoo']:.2f} yards")
    print(f"  Transformer: {test_row['transformer']:.2f} yards")
    print(f"  Improvement: {test_row['improvement_yards']:.2f} yards ({test_row['improvement_pct']:.1f}%)")

    print(f"\nTest Set Events:")
    for row in results:
        if row["split"].startswith("test-"):
            event_name = row["split"].replace("test-", "")
            print(f"  {event_name:15s}: {row['improvement_pct']:5.1f}% improvement")


if __name__ == "__main__":
    main()
