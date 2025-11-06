"""
Best Model Selection Module for NFL Big Data Bowl 2024

This module automates the selection of optimal models from hyperparameter search results.
After training multiple model configurations (e.g., 12 Transformer and 12 Zoo models),
this script identifies the best-performing checkpoint for each architecture based on
validation loss and consolidates them for easy access.

The selection process:
1. Scans all model directories under models/{transformer,zoo}/
2. Examines checkpoint filenames to extract validation loss (e.g., epoch=10-val_loss=2.543.ckpt)
3. Selects the checkpoint with lowest validation loss for each architecture
4. Copies best checkpoints and their associated prediction results to models/best_models/

This enables fair comparison between architectures by using each one's optimal configuration,
acknowledging that different architectures may have different optimal hyperparameters.

Functions:
    find_best_checkpoint: Locate the best checkpoint for each model type
    main: Main execution function for selecting and copying best models

Output Structure:
    models/best_models/
    ├── transformer/
    │   ├── best_model.ckpt          # Best Transformer checkpoint
    │   └── best_model_results.parquet  # Predictions from this model
    └── zoo/
        ├── best_model.ckpt          # Best Zoo checkpoint
        └── best_model_results.parquet  # Predictions from this model
"""

from argparse import ArgumentParser
from pathlib import Path
from shutil import copy

from train import get_epoch_val_loss_from_ckpt


def find_best_checkpoint(root_dir: Path) -> dict[str, Path]:
    """
    Find the best checkpoint for each model type based on validation loss.

    Args:
        root_dir (Path): Root directory containing model checkpoints.

    Returns:
        dict[str, Path]: Dictionary mapping model names to their best checkpoint paths.
    """
    root_path = Path(root_dir)
    best_checkpoints = {}

    # Traverse through each model's directory
    for model_dir in root_path.iterdir():
        if model_dir.is_dir() and model_dir.name != "best_models":
            # Model name is the directory name
            model_name = model_dir.name
            checkpoints = list(model_dir.rglob("*.ckpt"))

            if checkpoints:
                # Find the checkpoint with the lowest val_loss
                best_checkpoint = min(
                    checkpoints,
                    key=lambda x: get_epoch_val_loss_from_ckpt(x)[1],
                )
                best_checkpoints[model_name] = best_checkpoint

    return best_checkpoints


def main(args):
    """
    Main execution function for selecting and copying best models.

    Args:
        args (Namespace): Command-line arguments.

    This function finds the best checkpoints for each model type, copies them
    to a designated directory, and renames them for consistency.
    """
    # Set up directories
    root_dir = Path("models")
    out_root_dir = Path("models/best_models")

    # Find best checkpoints
    best_checkpoints = find_best_checkpoint(root_dir)

    # Process each best checkpoint
    for model_name, checkpoint_path in best_checkpoints.items():
        print(f"Best checkpoint for {model_name}: {checkpoint_path}")

        # Create output directory
        out_dir = out_root_dir / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Copy and rename checkpoint model file
        best_model_out_path = out_dir / "best_model.ckpt"
        if best_model_out_path.exists():
            best_model_out_path.unlink()
        copy(checkpoint_path, best_model_out_path)

        # Copy associated results file
        results_df_path = checkpoint_path.with_suffix(".results.parquet")
        best_model_results_path = out_dir / "best_model_results.parquet"
        if best_model_results_path.exists():
            best_model_results_path.unlink()
        copy(results_df_path, best_model_results_path)


if __name__ == "__main__":
    parser = ArgumentParser(description="Select and copy the best performing models.")
    # Add any command-line arguments if needed
    args = parser.parse_args()
    main(args)
