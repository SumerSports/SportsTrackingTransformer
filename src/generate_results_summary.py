"""
Generate Results Summary and Analysis

This module performs comprehensive analysis of trained model performance, generating
publication-ready figures, tables, and metrics for comparing different architectures.

Key Analyses:
1. Overall Performance Comparison: Calculates Average Displacement Error (ADE) across
   all data splits for each model architecture.

2. Event-Type Breakdown: Analyzes model performance at different game moments
   (snap, handoff, tackle, etc.) to understand where models excel or struggle.

3. Temporal Analysis: Examines how prediction accuracy changes as plays progress,
   measuring performance at different frames before the tackle event.

4. Model Scaling Analysis: Compares all trained models across different configurations
   to understand how each architecture responds to increased model capacity.

5. Computational Efficiency: Calculates FLOPs for inference to compare computational
   costs across different model sizes and architectures.

Outputs:
- results/results.csv: Comprehensive metrics table for all analyses
- results/model_comparison.json: Details of all trained model configurations
- results/frame_difference_plot.png: Temporal performance visualization
- results/model_scaling_plot.png: Model capacity vs. performance comparison

Usage:
    uv run python src/generate_results_summary.py
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from calflops import calculate_flops

from models import LitModel


def calculate_ade(
    x: pl.Series | np.ndarray,
    y: pl.Series | np.ndarray,
    x_pred: pl.Series | np.ndarray,
    y_pred: pl.Series | np.ndarray,
) -> float:
    """
    Calculate Average Displacement Error (ADE).

    ADE = mean Euclidean distance between predicted and true (x, y) locations.
    Standard metric for trajectory prediction, pose estimation, and spatial tasks.

    Formula: mean(sqrt((x_pred - x)² + (y_pred - y)²))

    Args:
        x: True x coordinates
        y: True y coordinates
        x_pred: Predicted x coordinates
        y_pred: Predicted y coordinates

    Returns:
        Average displacement error in the same units as input coordinates (yards)
    """
    if isinstance(x, pl.Series):
        x = x.to_numpy()
    if isinstance(y, pl.Series):
        y = y.to_numpy()
    if isinstance(x_pred, pl.Series):
        x_pred = x_pred.to_numpy()
    if isinstance(y_pred, pl.Series):
        y_pred = y_pred.to_numpy()

    distances = np.sqrt((x_pred - x) ** 2 + (y_pred - y) ** 2)
    return float(np.mean(distances))


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MODELS_DIR = Path("models/best_models")
ZOO_RESULTS = MODELS_DIR / "zoo" / "best_model_results.parquet"
TRANSFORMER_RESULTS = MODELS_DIR / "transformer" / "best_model_results.parquet"
ZOO_CHECKPOINT = MODELS_DIR / "zoo" / "best_model.ckpt"
TRANSFORMER_CHECKPOINT = MODELS_DIR / "transformer" / "best_model.ckpt"


def load_results() -> pl.DataFrame:
    """Load and combine results from both models, including per-frame events from tracking data."""
    print("Loading model results...")
    results_df = pl.concat(
        [pl.read_parquet(ZOO_RESULTS), pl.read_parquet(TRANSFORMER_RESULTS)],
        how="diagonal",
    )

    # Load tracking data to get per-frame events
    print("Loading tracking data for per-frame events...")
    tracking_df = pl.read_parquet("data/split_prepped_data/*_features.parquet")

    # Join with tracking data to get per-frame events
    results_df = results_df.join(
        tracking_df.filter(pl.col("is_ball_carrier") == 1)
        .select(["x", "y", "gameId", "playId", "frameId", "mirrored", "event"])
        .rename({"x": "ball_carrier_x", "y": "ball_carrier_y"}),
        on=["gameId", "playId", "frameId", "mirrored"],
        how="inner",
    )

    # Filter to mirrored=False to avoid double-counting predictions
    #
    # During training, we augment data by horizontally flipping each play (data augmentation).
    # This gives us 2× more training examples from the same data, helping the model generalize.
    #
    # Example: Original play has ball carrier running right → tackle at x=30, y=25
    #          Mirrored play has ball carrier running left → tackle at x=70, y=25 (x is flipped)
    #
    # When evaluating, we only count each unique play once to avoid inflating our metrics.
    # Both the original and mirrored versions produce predictions, but we only evaluate
    # the original (mirrored=False) to get true performance on unique plays.
    results_df = results_df.filter(pl.col("mirrored") == False)
    print(f"  Loaded {len(results_df):,} predictions (mirrored=False only)")
    return results_df


def _calculate_ade_for_df(df: pl.DataFrame) -> float:
    """
    Helper to calculate ADE from a DataFrame with prediction columns.

    Args:
        df: DataFrame with columns: tackle_x, tackle_y, tackle_x_pred, tackle_y_pred

    Returns:
        ADE in yards, rounded to 2 decimal places
    """
    ade = df.select(
        pl.map_groups(
            exprs=["tackle_x", "tackle_y", "tackle_x_pred", "tackle_y_pred"],
            function=lambda ls: calculate_ade(*ls),
            returns_scalar=True,
        )
    ).item()
    return round(ade, 2)


def _calculate_improvement_metrics(zoo_ade: float, transformer_ade: float) -> tuple[float, float]:
    """
    Calculate improvement metrics comparing Zoo baseline to Transformer model.

    Args:
        zoo_ade: ADE for Zoo model (baseline)
        transformer_ade: ADE for Transformer model

    Returns:
        Tuple of (improvement_pct, improvement_yards)
        - improvement_pct: Percentage improvement (positive = better)
        - improvement_yards: Absolute yards improvement
    """
    improvement_pct = round((zoo_ade - transformer_ade) / zoo_ade * 100, 1)
    improvement_yards = round(zoo_ade - transformer_ade, 2)
    return improvement_pct, improvement_yards


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
            row[model_type] = _calculate_ade_for_df(model_df)

        row["improvement_pct"], row["improvement_yards"] = _calculate_improvement_metrics(
            row["zoo"], row["transformer"]
        )
        row["n_plays"] = split_df.select(pl.struct(["gameId", "playId"]).n_unique()).item()
        row["n_frames"] = split_df.select(pl.len()).item()

        results.append(row)

    # Test event breakdowns (using per-frame events)
    print("Calculating test set event breakdowns (per-frame events)...")
    test_df = results_df.filter(pl.col("dataset_split") == "test")
    events = sorted(test_df.filter(pl.col("event").is_not_null())["event"].unique().to_list())

    event_results = []
    for event in events:
        # Use per-frame events (all frames with this event)
        event_df = test_df.filter(pl.col("event") == event)

        # Skip events with too few plays
        n_plays = event_df.select(pl.struct(["gameId", "playId"]).n_unique()).item()
        if n_plays < 100:
            continue

        row = {"split": f"test-event-{event}", "metric": "ade_yards"}

        for model_type in ["zoo", "transformer"]:
            model_df = event_df.filter(pl.col("model_type") == model_type)
            row[model_type] = _calculate_ade_for_df(model_df)

        row["improvement_pct"], row["improvement_yards"] = _calculate_improvement_metrics(
            row["zoo"], row["transformer"]
        )
        row["n_plays"] = n_plays
        row["n_frames"] = event_df.select(pl.len()).item()
        row["_avg_frameId"] = round(event_df["frameId"].mean(), 1)  # For sorting only

        event_results.append(row)

    # Sort event results by avg_frameId, then remove the sorting key
    event_results.sort(key=lambda x: x["_avg_frameId"])
    for row in event_results:
        del row["_avg_frameId"]
    results.extend(event_results)

    return results


def calculate_frame_difference_results(results_df: pl.DataFrame) -> tuple[list[dict], pl.DataFrame]:
    """
    Calculate results by frame difference from tackle (test set only).

    Returns:
        tuple: (list of result dicts, DataFrame for plotting)
    """
    print("\nCalculating frame-difference breakdown (test set only)...")

    frame_diff_df = (
        results_df.with_columns(
            frame_difference_from_tackle=(pl.col("tackle_frameId") - pl.col("frameId")),
        )
        .with_columns(
            frame_difference_from_tackle_cat=(
                pl.col("frame_difference_from_tackle").cut(
                    breaks=range(0, 31, 5),
                    labels=["after tackle", "0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30+"],
                    left_closed=True,
                )
            )
        )
        .filter(pl.col("dataset_split") == "test")
        .group_by(["model_type", "frame_difference_from_tackle_cat"])
        .agg(
            order=pl.col("frame_difference_from_tackle").mean() * -1,
            n_frames=pl.len(),
            n_plays=pl.struct(["gameId", "playId"]).n_unique(),
            ade_yards=pl.map_groups(
                exprs=["tackle_x", "tackle_y", "tackle_x_pred", "tackle_y_pred"],
                function=lambda ls: round(calculate_ade(*ls), 2),
                returns_scalar=True,
            ),
        )
        .sort("frame_difference_from_tackle_cat")
    )

    # Convert to results format for JSON
    results = []
    categories = sorted(frame_diff_df["frame_difference_from_tackle_cat"].unique().to_list())

    for category in categories:
        cat_df = frame_diff_df.filter(pl.col("frame_difference_from_tackle_cat") == category)

        row = {"split": f"test-frames-before-tackle-{category}", "metric": "ade_yards"}

        for model_type in ["zoo", "transformer"]:
            model_data = cat_df.filter(pl.col("model_type") == model_type)
            if len(model_data) > 0:
                row[model_type] = model_data["ade_yards"].item()

        if "zoo" in row and "transformer" in row:
            row["improvement_pct"], row["improvement_yards"] = _calculate_improvement_metrics(
                row["zoo"], row["transformer"]
            )

        # Get n_plays and n_frames (should be same for both models)
        first_row = cat_df.row(0, named=True)
        row["n_plays"] = int(first_row["n_plays"])
        row["n_frames"] = int(first_row["n_frames"])

        results.append(row)

    return results, frame_diff_df


def generate_frame_difference_plot(frame_diff_df: pl.DataFrame) -> None:
    """Generate and save frame-difference plot."""
    print("\nGenerating frame-difference plot...")

    # Convert to pandas for plotting
    frame_diff_df_pandas = frame_diff_df.to_pandas()

    # Create the line plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=frame_diff_df_pandas,
        x="frame_difference_from_tackle_cat",
        y="ade_yards",
        hue="model_type",
        marker="o",
    )

    # Flip the x-axis
    plt.gca().invert_xaxis()

    # Customize the plot
    plt.title("Model Performance by Frames Before Tackle", fontsize=16)
    plt.xlabel("Frames Before Tackle", fontsize=12)
    plt.ylabel("Average Displacement Error (yards)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Model Type", title_fontsize="12", fontsize="10")

    # Adjust layout and save
    plt.tight_layout()
    plot_path = RESULTS_DIR / "frame_difference_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {plot_path}")


def find_all_model_checkpoints() -> list[dict]:
    """
    Find all model checkpoints and group by configuration.

    Returns:
        list[dict]: List of config dicts with model_type, model_dim, num_layers, and best checkpoint path.
    """
    models_base = Path("models")
    configs = []

    for model_type in ["zoo", "transformer"]:
        model_dir = models_base / model_type
        if not model_dir.exists():
            continue

        # Find all config directories (e.g., M128_L2_LR1e-04)
        for config_dir in model_dir.iterdir():
            if not config_dir.is_dir() or not config_dir.name.startswith("M"):
                continue

            # Parse config from directory name
            match = re.match(r"M(\d+)_L(\d+)_LR", config_dir.name)
            if not match:
                continue

            model_dim = int(match.group(1))
            num_layers = int(match.group(2))

            # Find best checkpoint (lowest val_loss)
            checkpoints_dir = config_dir / "checkpoints"
            if not checkpoints_dir.exists():
                continue

            checkpoint_files = list(checkpoints_dir.glob("*.ckpt"))
            if not checkpoint_files:
                continue

            # Parse val_loss from filename and find best
            best_checkpoint = None
            best_val_loss = float("inf")

            for ckpt_file in checkpoint_files:
                # Parse: epoch=X-val_loss=Y.YYY.ckpt
                val_loss_match = re.search(r"val_loss=([\d.]+?)\.ckpt", ckpt_file.name)
                if val_loss_match:
                    val_loss = float(val_loss_match.group(1))
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_checkpoint = ckpt_file

            if best_checkpoint:
                # Find corresponding results file
                results_file = best_checkpoint.with_suffix(".results.parquet")
                if results_file.exists():
                    configs.append(
                        {
                            "model_type": model_type,
                            "model_dim": model_dim,
                            "num_layers": num_layers,
                            "checkpoint_path": str(best_checkpoint),
                            "results_path": str(results_file),
                            "val_loss": best_val_loss,
                        }
                    )

    return configs


def compute_model_metrics(checkpoint_path: str, model_type: str) -> dict:
    """
    Compute params and FLOPs for a single model checkpoint.

    Args:
        checkpoint_path (str): Path to checkpoint file.
        model_type (str): 'zoo' or 'transformer'.

    Returns:
        dict: Metrics including params and inference_flops.
    """
    # Load model
    lit_model = LitModel.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model = lit_model.model
    model.eval()

    # Create dummy input shape
    if model_type == "transformer":
        input_shape = (1, 22, 6)
    else:  # zoo
        input_shape = (1, 10, 11, 10)

    # Calculate params
    params = int(lit_model.hparams["params"])

    # Calculate FLOPs using calflops
    # Note: We use calflops instead of fvcore because it properly counts
    # transformer attention operations (scaled_dot_product_attention),
    # which are critical for accurate FLOP comparison between models.
    try:
        flops, macs, _ = calculate_flops(
            model=model,
            input_shape=input_shape,
            print_results=False,
            output_as_string=False,
        )
        inference_flops = int(flops)
    except Exception:
        inference_flops = None

    return {"params": params, "inference_flops": inference_flops}


def compute_test_ade(results_path: str) -> float:
    """
    Compute test set ADE from results parquet file.

    Args:
        results_path (str): Path to results parquet file.

    Returns:
        float: Test set ADE in yards.
    """
    df = pl.read_parquet(results_path)
    test_df = df.filter((pl.col("dataset_split") == "test") & (pl.col("mirrored") == False))

    ade = test_df.select(
        pl.map_groups(
            exprs=["tackle_x", "tackle_y", "tackle_x_pred", "tackle_y_pred"],
            function=lambda ls: calculate_ade(*ls),
            returns_scalar=True,
        )
    ).item()

    return float(ade)


def compute_model_comparison() -> list[dict]:
    """
    Compute comprehensive comparison of all trained models.

    Returns:
        list[dict]: List of model configs with params, FLOPs, and test ADE.
    """
    print("\nComputing comprehensive model comparison...")

    # Find all checkpoints
    configs = find_all_model_checkpoints()
    print(f"  Found {len(configs)} model configurations")

    results = []

    for i, config in enumerate(configs, 1):
        print(
            f"  [{i}/{len(configs)}] Processing {config['model_type']} "
            f"M{config['model_dim']}_L{config['num_layers']}..."
        )

        # Compute metrics
        metrics = compute_model_metrics(config["checkpoint_path"], config["model_type"])
        test_ade = compute_test_ade(config["results_path"])

        results.append(
            {
                "model_type": config["model_type"],
                "model_dim": config["model_dim"],
                "num_layers": config["num_layers"],
                "params": metrics["params"],
                "inference_flops": metrics["inference_flops"],
                "test_ade_yards": round(test_ade, 2),
                "val_loss": round(config["val_loss"], 3),
            }
        )

    # Sort by model_type, then params
    results.sort(key=lambda x: (x["model_type"], x["params"]))

    return results


def generate_model_scaling_plot(model_comparison: list[dict]) -> None:
    """
    Generate model scaling plot showing Test ADE vs FLOPs.

    This visualization supports the "Model Selection and Architectural Scaling"
    section of the paper, demonstrating how Zoo and Transformer architectures
    scale with computational budget.
    """
    print("\nGenerating model scaling plot...")

    # Convert to DataFrame for easier manipulation
    df = pl.DataFrame(model_comparison).to_pandas()

    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Define colors and markers
    colors = {"zoo": "#FF7F0E", "transformer": "#1F77B4"}
    markers = {"zoo": "s", "transformer": "o"}

    # Plot: Test ADE vs FLOPs
    for model_type in ["zoo", "transformer"]:
        data = df[df["model_type"] == model_type].sort_values("inference_flops")
        ax.plot(
            data["inference_flops"],
            data["test_ade_yards"],
            marker=markers[model_type],
            markersize=8,
            linewidth=2,
            label=model_type.capitalize(),
            color=colors[model_type],
            alpha=0.8,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Inference FLOPs (log scale)", fontsize=12)
    ax.set_ylabel("Test ADE (yards) - Lower is Better", fontsize=12)
    ax.set_title("Model Scaling: Test ADE vs FLOPs", fontsize=14, fontweight="bold")
    ax.legend(title="Architecture", fontsize=11, title_fontsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Adjust layout and save
    plt.tight_layout()
    plot_path = RESULTS_DIR / "model_scaling_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {plot_path}")


def main():
    """Generate results summary."""
    print("=" * 60)
    print("GENERATING RESULTS")
    print("=" * 60)

    results_df = load_results()
    results = calculate_results(results_df)

    # Add frame-difference results (test only)
    frame_diff_results, frame_diff_df = calculate_frame_difference_results(results_df)
    results.extend(frame_diff_results)

    # Generate frame-difference plot
    generate_frame_difference_plot(frame_diff_df)

    # Save results CSV
    results_csv_path = RESULTS_DIR / "results.csv"
    results_pl_df = pl.DataFrame(results)
    results_pl_df.write_csv(results_csv_path)
    print(f"\n✓ Saved: {results_csv_path}")

    # Compute and save comprehensive model comparison
    model_comparison = compute_model_comparison()
    comparison_path = RESULTS_DIR / "model_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(model_comparison, f, indent=2)
    print(f"\n✓ Saved: {comparison_path} ({len(model_comparison)} models)")

    # Generate model scaling plot
    generate_model_scaling_plot(model_comparison)

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
        if row["split"].startswith("test-event-"):
            event_name = row["split"].replace("test-event-", "")
            print(f"  {event_name:20s}: {row['improvement_pct']:5.1f}% improvement")

    print(f"\nTest Set Frame Differences:")
    for row in results:
        if row["split"].startswith("test-frames-before-tackle-"):
            frame_cat = row["split"].replace("test-frames-before-tackle-", "")
            print(f"  {frame_cat:15s}: {row['improvement_pct']:5.1f}% improvement")


if __name__ == "__main__":
    main()
