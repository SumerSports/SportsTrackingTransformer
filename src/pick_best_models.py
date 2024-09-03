import re
from argparse import ArgumentParser
from pathlib import Path
from shutil import copy


def find_best_checkpoint(root_dir: Path) -> dict[str, Path]:
    root_path = Path(root_dir)
    best_checkpoints = {}

    # Regex to extract validation loss from filename
    loss_pattern = re.compile(r"val_loss=([\d\.]+)\D*")

    # Traverse through each model's directory
    for model_dir in root_path.iterdir():
        if model_dir.is_dir():
            # Model name is the directory name
            model_name = model_dir.name
            lowest_loss = float("inf")
            best_checkpoint_path = None

            # Loop through all subdirectories and files
            for checkpoint_file in model_dir.rglob("*.ckpt"):
                # skip if result file doesnt exist
                ckpt_results_file = checkpoint_file.with_suffix(".results.parquet")
                if not ckpt_results_file.exists():
                    continue

                # Find the validation loss from the filename
                match = loss_pattern.search(checkpoint_file.name)
                if match:
                    match_str = match.group(1)
                    current_loss = float(match_str.rstrip("."))

                    # Update the best checkpoint if the current one has a lower loss
                    if current_loss < lowest_loss:
                        lowest_loss = current_loss
                        best_checkpoint_path = checkpoint_file

            if best_checkpoint_path:
                best_checkpoints[model_name] = best_checkpoint_path

    return best_checkpoints


def main(args):
    # Example usage
    root_dir = Path("models")
    out_root_dir = Path("models/best_models")
    best_checkpoints = find_best_checkpoint(root_dir)

    for model_name, checkpoint_path in best_checkpoints.items():
        print(f"Best checkpoint for {model_name}: {checkpoint_path}")

        out_dir = out_root_dir / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        # copy and rename checkpoint model file used
        best_model_out_path = out_dir / "best_model.ckpt"
        if best_model_out_path.exists():
            best_model_out_path.unlink()
        copy(checkpoint_path, best_model_out_path)

        results_df_path = checkpoint_path.with_suffix(".results.parquet")
        best_model_results_path = out_dir / "best_model_results.parquet"
        if best_model_results_path.exists():
            best_model_results_path.unlink()
        copy(results_df_path, best_model_results_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)
