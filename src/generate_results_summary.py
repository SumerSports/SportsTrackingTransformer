"""
Generate Results Summary

Processes model predictions and generates result summary for publication.
Usage: uv run python src/generate_results_summary.py
"""

import json
import re
from pathlib import Path

import polars as pl
import torch
from fvcore.nn import FlopCountAnalysis

from metrics import calculate_ade
from models import LitModel

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MODELS_DIR = Path("models/best_models")
ZOO_RESULTS = MODELS_DIR / "zoo" / "best_model_results.parquet"
TRANSFORMER_RESULTS = MODELS_DIR / "transformer" / "best_model_results.parquet"
ZOO_CHECKPOINT = MODELS_DIR / "zoo" / "best_model.ckpt"
TRANSFORMER_CHECKPOINT = MODELS_DIR / "transformer" / "best_model.ckpt"


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

    # Create dummy input
    if model_type == "transformer":
        dummy_input = torch.randn(1, 22, 6)
    else:  # zoo
        dummy_input = torch.randn(1, 10, 11, 10)

    # Calculate params
    params = int(lit_model.hparams["params"])

    # Calculate FLOPs
    try:
        flops_analysis = FlopCountAnalysis(model, dummy_input)
        inference_flops = int(flops_analysis.total())
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


def main():
    """Generate results summary."""
    print("=" * 60)
    print("GENERATING RESULTS")
    print("=" * 60)

    results_df = load_results()
    results = calculate_results(results_df)

    # Save results JSON
    json_path = RESULTS_DIR / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved: {json_path}")

    # Compute and save comprehensive model comparison
    model_comparison = compute_model_comparison()
    comparison_path = RESULTS_DIR / "model_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(model_comparison, f, indent=2)
    print(f"\n✓ Saved: {comparison_path} ({len(model_comparison)} models)")

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
