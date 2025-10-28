"""
Data Preparation Module for NFL Big Data Bowl 2026

This module processes raw NFL tracking data to prepare it for machine learning models.
It includes functions for loading, cleaning, and transforming the data, as well as
splitting it into train, validation, and test sets.

Functions:
    load_input_data: Load input tracking data from CSV files
    convert_tracking_to_cartesian: Convert polar coordinates to Cartesian
    standardize_tracking_directions: Standardize play directions
    prepare_tracking_data: Prepare tracking data with position and kinematic features
    split_train_test_val: Split data into train, validation, and test sets
    main: Main execution function

"""

from argparse import ArgumentParser
from pathlib import Path

import polars as pl

INPUT_DATA_DIR = Path("data/train/")


def load_input_data() -> pl.DataFrame:
    """
    Load input tracking data from CSV files.

    Returns:
        pl.DataFrame: Raw tracking data with all fields from input files.
    """
    # Read all input CSV files from the train directory
    df = pl.read_csv(INPUT_DATA_DIR / "input_*.csv", null_values=["NA", "nan", "N/A", "NaN", ""])
    print(f"Loaded {len(df)} rows from input files")
    print(f"Unique plays: {df.n_unique(['game_id', 'play_id'])}")
    return df


def convert_tracking_to_cartesian(tracking_df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert polar coordinates to Unit-circle Cartesian format.
    We keep the original position (x, y) and kinematic variables (s, a, dir, o),
    and also compute cartesian velocity components (vx, vy) and orientation (ox, oy).

    Args:
        tracking_df (pl.DataFrame): Tracking data

    Returns:
        pl.DataFrame: Tracking data with Cartesian coordinates.
    """
    return (
        tracking_df.with_columns(
            # Adjust dir and o to match unit circle convention
            dir_adjusted=((pl.col("dir") - 90) * -1) % 360,
            o_adjusted=((pl.col("o") - 90) * -1) % 360,
        )
        # convert polar vectors to cartesian ((s, dir) -> (vx, vy), (o) -> (ox, oy))
        .with_columns(
            vx=pl.col("s") * pl.col("dir_adjusted").radians().cos(),
            vy=pl.col("s") * pl.col("dir_adjusted").radians().sin(),
            ox=pl.col("o_adjusted").radians().cos(),
            oy=pl.col("o_adjusted").radians().sin(),
        )
        .drop(["dir_adjusted", "o_adjusted"])
    )


def standardize_tracking_directions(tracking_df: pl.DataFrame) -> pl.DataFrame:
    """
    Standardize play directions to always moving left to right.
    Also standardize ball_land_x and ball_land_y targets.

    Args:
        tracking_df (pl.DataFrame): Tracking data

    Returns:
        pl.DataFrame: Tracking data with standardized directions.
    """
    return tracking_df.with_columns(
        x=pl.when(pl.col("play_direction") == "right").then(pl.col("x")).otherwise(120 - pl.col("x")),
        y=pl.when(pl.col("play_direction") == "right").then(pl.col("y")).otherwise(53.3 - pl.col("y")),
        vx=pl.when(pl.col("play_direction") == "right").then(pl.col("vx")).otherwise(-1 * pl.col("vx")),
        vy=pl.when(pl.col("play_direction") == "right").then(pl.col("vy")).otherwise(-1 * pl.col("vy")),
        ox=pl.when(pl.col("play_direction") == "right").then(pl.col("ox")).otherwise(-1 * pl.col("ox")),
        oy=pl.when(pl.col("play_direction") == "right").then(pl.col("oy")).otherwise(-1 * pl.col("oy")),
        # Also standardize the target variables
        ball_land_x=pl.when(pl.col("play_direction") == "right").then(pl.col("ball_land_x")).otherwise(120 - pl.col("ball_land_x")),
        ball_land_y=pl.when(pl.col("play_direction") == "right").then(pl.col("ball_land_y")).otherwise(53.3 - pl.col("ball_land_y")),
    ).drop("play_direction")


def prepare_tracking_data(tracking_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Prepare tracking data with position and kinematic features, and extract targets.

    Args:
        tracking_df (pl.DataFrame): Tracking data

    Returns:
        tuple: (features_df, targets_df) where features_df contains position/kinematic features
               and targets_df contains ball_land_x and ball_land_y per frame
    """
    # Filter out rows where ball_land_x or ball_land_y are null
    tracking_df = tracking_df.filter(
        pl.col("ball_land_x").is_not_null() & pl.col("ball_land_y").is_not_null()
    )

    print(f"After filtering nulls: {len(tracking_df)} rows")
    print(f"Unique frames: {tracking_df.n_unique(['game_id', 'play_id', 'frame_id'])}")

    # Create target dataframe (one row per frame with ball landing location)
    targets_df = (
        tracking_df.select([
            "game_id",
            "play_id",
            "frame_id",
            "ball_land_x",
            "ball_land_y",
        ])
        .unique()
    )

    # Select only position and kinematic variables for features
    # Position: x, y
    # Kinematic: s (speed), a (acceleration), vx, vy (velocity components), ox, oy (orientation components)
    features_df = tracking_df.select([
        "game_id",
        "play_id",
        "nfl_id",
        "frame_id",
        "x",
        "y",
        "s",
        "a",
        "vx",
        "vy",
        "ox",
        "oy",
        "player_side",  # Keep side information to distinguish offense/defense
        "ball_land_x",  # Keep for joining later
        "ball_land_y",
    ])

    return features_df, targets_df


def split_train_test_val(features_df: pl.DataFrame, targets_df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """
    Split data into train, validation, and test sets.
    Split is 70-15-15 for train-test-val respectively. Notably, we split at the play level and not frame level.
    This ensures no target contamination between splits.

    Args:
        features_df (pl.DataFrame): Features data
        targets_df (pl.DataFrame): Target data

    Returns:
        dict: Dictionary containing train, validation, and test dataframes.
    """
    features_df = features_df.sort(["game_id", "play_id", "frame_id"])
    targets_df = targets_df.sort(["game_id", "play_id", "frame_id"])

    print(
        f"Total set: {features_df.n_unique(['game_id', 'play_id'])} plays,",
        f"{features_df.n_unique(['game_id', 'play_id', 'frame_id'])} frames",
    )

    # Split at play level
    test_val_ids = features_df.select(["game_id", "play_id"]).unique(maintain_order=True).sample(fraction=0.3, seed=42)
    train_features_df = features_df.join(test_val_ids, on=["game_id", "play_id"], how="anti")
    train_targets_df = targets_df.join(test_val_ids, on=["game_id", "play_id"], how="anti")
    print(
        f"Train set: {train_features_df.n_unique(['game_id', 'play_id'])} plays,",
        f"{train_features_df.n_unique(['game_id', 'play_id', 'frame_id'])} frames",
    )

    test_ids = test_val_ids.sample(fraction=0.5, seed=42)  # 70-15-15 split
    test_features_df = features_df.join(test_ids, on=["game_id", "play_id"], how="inner")
    test_targets_df = targets_df.join(test_ids, on=["game_id", "play_id"], how="inner")
    print(
        f"Test set: {test_features_df.n_unique(['game_id', 'play_id'])} plays,",
        f"{test_features_df.n_unique(['game_id', 'play_id', 'frame_id'])} frames",
    )

    val_ids = test_val_ids.join(test_ids, on=["game_id", "play_id"], how="anti")
    val_features_df = features_df.join(val_ids, on=["game_id", "play_id"], how="inner")
    val_targets_df = targets_df.join(val_ids, on=["game_id", "play_id"], how="inner")
    print(
        f"Validation set: {val_features_df.n_unique(['game_id', 'play_id'])} plays,",
        f"{val_features_df.n_unique(['game_id', 'play_id', 'frame_id'])} frames",
    )

    return {
        "train_features": train_features_df,
        "train_targets": train_targets_df,
        "test_features": test_features_df,
        "test_targets": test_targets_df,
        "val_features": val_features_df,
        "val_targets": val_targets_df,
    }


def main():
    """
    Main execution function for data preparation.

    This function orchestrates the entire data preparation process, including:
    1. Loading raw input data
    2. Converting coordinates to cartesian
    3. Standardizing play directions
    4. Preparing features (position and kinematic only) and targets (ball_land_x, ball_land_y)
    5. Splitting data into train, validation, and test sets
    6. Saving processed data to parquet files
    """
    # Load input data
    tracking_df = load_input_data()

    # Convert to cartesian coordinates
    tracking_df = convert_tracking_to_cartesian(tracking_df)

    # Standardize directions (all plays moving left to right)
    tracking_df = standardize_tracking_directions(tracking_df)

    # Prepare features and targets
    features_df, targets_df = prepare_tracking_data(tracking_df)

    # Split into train/val/test
    split_dfs = split_train_test_val(features_df, targets_df)

    # Save to parquet files
    out_dir = Path("data/split_prepped_data/")
    out_dir.mkdir(exist_ok=True, parents=True)

    for key, df in split_dfs.items():
        sort_keys = ["game_id", "play_id", "frame_id"]
        if "nfl_id" in df.columns:
            sort_keys.append("nfl_id")
        df.sort(sort_keys).write_parquet(out_dir / f"{key}.parquet")

    print("\nData preparation complete!")
    print(f"Output saved to: {out_dir}")


if __name__ == "__main__":
    main()
