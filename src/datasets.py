"""
Dataset Module for NFL Big Data Bowl 2024

This module handles data loading and preprocessing for tackle prediction models.
It implements two distinct feature engineering approaches:

1. Transformer Model: Minimal feature engineering, providing raw player features
   to leverage self-attention for learning spatial relationships end-to-end.

2. Zoo Model: Complex pairwise feature engineering creating a 10x11 grid of
   offensive-defensive player interactions, following the architecture that won
   the 2020 NFL Big Data Bowl.

The key insight is that Transformer models can learn these interaction patterns
automatically, while Zoo models require manual feature engineering.

Classes:
    BDB2024_Dataset: Custom PyTorch dataset class for NFL tracking data

Functions:
    load_datasets: Load preprocessed datasets from disk
    main: Precompute and cache datasets for all splits and model types

Usage:
    dataset = load_datasets('transformer', 'train')
    features, targets = dataset[0]  # Get first sample
"""

import multiprocessing as mp
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from torch.utils.data import Dataset
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

PREPPED_DATA_DIR = Path("data/split_prepped_data/")
DATASET_DIR = Path("data/datasets/")


class BDB2024_Dataset(Dataset):
    """
    Custom dataset class for NFL tracking data.

    This class preprocesses and stores NFL tracking data for use in machine learning models.
    It supports both 'transformer' and 'zoo' model types.

    Attributes:
        model_type (str): Type of model ('transformer' or 'zoo')
        keys (list): List of unique identifiers for each data point
        feature_df_partition (pd.DataFrame): Preprocessed feature data
        tgt_df_partition (pd.DataFrame): Preprocessed target data
        tgt_arrays (dict): Precomputed target arrays
        feature_arrays (dict): Precomputed feature arrays
    """

    def __init__(
        self,
        model_type: str,
        feature_df: pl.DataFrame,
        tgt_df: pl.DataFrame,
    ):
        """
        Initialize the dataset.

        Args:
            model_type (str): Type of model ('transformer' or 'zoo')
            feature_df (pl.DataFrame): DataFrame containing feature data
            tgt_df (pl.DataFrame): DataFrame containing target data

        Raises:
            ValueError: If an invalid model_type is provided
        """
        if model_type not in ["transformer", "zoo"]:
            raise ValueError("model_type must be either 'transformer' or 'zoo'")

        self.model_type = model_type
        # Sort keys to ensure deterministic ordering across runs
        self.keys = sorted(feature_df.select(["gameId", "playId", "mirrored", "frameId"]).unique().rows())

        # Convert to pandas form with index for quick row retrieval
        self.feature_df_partition = (
            feature_df.to_pandas(use_pyarrow_extension_array=True)
            .set_index(["gameId", "playId", "mirrored", "frameId", "nflId"])
            .sort_index()
        )
        self.tgt_df_partition = (
            tgt_df.to_pandas(use_pyarrow_extension_array=True)
            .set_index(["gameId", "playId", "mirrored", "frameId"])
            .sort_index()
        )

        # Precompute features and store in dicts
        # Note: Using pool.map() preserves input order, ensuring deterministic dictionary construction
        self.tgt_arrays: dict[tuple, np.ndarray] = {}
        self.feature_arrays: dict[tuple, np.ndarray] = {}
        with mp.Pool(processes=min(8, mp.cpu_count())) as pool:
            results = pool.map(
                self.process_key,
                tqdm(self.keys, desc="Pre-computing feature transforms", total=len(self.keys)),
            )
            # Unpack results in the same order as self.keys (pool.map guarantees order)
            for key, tgt_array, feature_array in results:
                self.tgt_arrays[key] = tgt_array
                self.feature_arrays[key] = feature_array

    def process_key(self, key: tuple) -> tuple[tuple, np.ndarray, np.ndarray]:
        """
        Process a single key to generate target and feature arrays.

        Args:
            key (tuple): Key (gameId, playId, mirrored, frameId) identifying a specific data point

        Returns:
            tuple[tuple, np.ndarray, np.ndarray]: Processed key, target array, and feature array
        """
        tgt_array = self.transform_target_df(self.tgt_df_partition.loc[key])
        feature_array = self.transform_input_frame_df(self.feature_df_partition.loc[key])
        return key, tgt_array, feature_array

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Number of samples in the dataset
        """
        return len(self.keys)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get a single item from the dataset.

        Args:
            idx (int): Index of the item to retrieve

        Returns:
            tuple[np.ndarray, np.ndarray]: Feature array and target array for the specified index

        Raises:
            IndexError: If the index is out of range
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        key = self.keys[idx]
        return self.feature_arrays[key], self.tgt_arrays[key]

    def transform_input_frame_df(self, frame_df: pd.DataFrame) -> np.ndarray:
        """
        Transform input frame DataFrame to numpy array based on model type.

        Args:
            frame_df (pd.DataFrame): Input frame DataFrame

        Returns:
            np.ndarray: Transformed input features

        Raises:
            ValueError: If an unknown model type is specified
        """
        if self.model_type == "transformer":
            return self.transformer_transform_input_frame_df(frame_df)
        elif self.model_type == "zoo":
            return self.zoo_transform_input_frame_df(frame_df)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def transform_target_df(self, tgt_df: pd.DataFrame) -> np.ndarray:
        """
        Transform target DataFrame to numpy array.

        Args:
            tgt_df (pd.DataFrame): Target DataFrame

        Returns:
            np.ndarray: Transformed target values

        Raises:
            AssertionError: If the output shape is not as expected
        """
        y = tgt_df[["tackle_x_rel", "tackle_y_rel"]].to_numpy(dtype=np.float32).squeeze()
        assert y.shape == (2,), f"Expected shape (2,), got {y.shape}"
        return y

    def transformer_transform_input_frame_df(self, frame_df: pd.DataFrame) -> np.ndarray:
        """
        Transform input frame DataFrame for transformer model.

        Args:
            frame_df (pd.DataFrame): Input frame DataFrame

        Returns:
            np.ndarray: Transformed input features for transformer model

        Raises:
            AssertionError: If the output shape is not as expected
        """
        # Features fed to the Transformer model (per player):
        # - x_rel, y_rel: Player position relative to ball carrier (in yards)
        #                 Using relative positions makes the model learn spatial relationships
        #                 (e.g., "defender 5 yards ahead") rather than absolute field positions
        # - vx, vy: Player velocity in x and y directions (yards/second)
        #           Velocity helps predict where players will be, not just where they are now
        # - side: Offensive (+1) or Defensive (-1) team indicator
        #         Helps model learn different roles (e.g., blockers vs. tacklers)
        # - is_ball_carrier: Binary flag (1 = has ball, 0 = doesn't)
        #                    Critical for identifying the target player being tackled
        #
        # Shape: (22 players, 6 features)
        # The Transformer's self-attention mechanism will learn to focus on relevant players
        # (e.g., nearby defenders, blocking assignments) automatically during training.
        features = ["x_rel", "y_rel", "vx", "vy", "side", "is_ball_carrier"]
        x = frame_df[features].to_numpy(dtype=np.float32)
        assert x.shape == (22, len(features)), f"Expected shape (22, {len(features)}), got {x.shape}"
        return x

    def zoo_transform_input_frame_df(self, frame_df: pd.DataFrame) -> np.ndarray:
        """
        Transform input frame DataFrame for zoo model.

        Args:
            frame_df (pd.DataFrame): Input frame DataFrame

        Returns:
            np.ndarray: Transformed input features for zoo model

        Raises:
            AssertionError: If the output shape is not as expected
        """
        # Isolate offensive and defensive players
        ball_carrier = frame_df[frame_df["is_ball_carrier"] == 1]
        off_plyrs = frame_df[(frame_df["side"] == 1) & (frame_df["is_ball_carrier"] == 0)]
        def_plyrs = frame_df[frame_df["side"] == -1]

        ball_carr_mvmt_feats = ball_carrier[["x_rel", "y_rel", "vx", "vy"]].to_numpy(dtype=np.float32).squeeze()
        off_mvmt_feats = off_plyrs[["x_rel", "y_rel", "vx", "vy"]].to_numpy(dtype=np.float32)
        def_mvmt_feats = def_plyrs[["x_rel", "y_rel", "vx", "vy"]].to_numpy(dtype=np.float32)

        # Zoo interaction features
        x = [
            # def_vx, def_vy
            np.tile(def_mvmt_feats[:, 2:], (10, 1, 1)),
            # def_x - ball_x, def_y - ball_y
            np.tile(
                def_mvmt_feats[None, :, :2] - ball_carr_mvmt_feats[None, None, :2],
                (10, 1, 1),
            ),
            # def_vx - ball_vx, def_vy - ball_vy
            np.tile(
                def_mvmt_feats[None, :, 2:] - ball_carr_mvmt_feats[None, None, 2:],
                (10, 1, 1),
            ),
            # off_x - def_x, off_y - def_y
            off_mvmt_feats[:, None, :2] - def_mvmt_feats[None, :, :2],
            # off_vx - def_vx, off_vy - def_vy
            off_mvmt_feats[:, None, 2:] - def_mvmt_feats[None, :, 2:],
        ]

        x = np.concatenate(
            x,
            dtype=np.float32,
            axis=-1,
        )

        # Zoo Architecture expects shape: (10 offensive players, 11 defensive players, 10 interaction features)
        #
        # This creates a grid where each cell [i, j] represents the interaction between
        # offensive player i and defensive player j:
        #
        #           Defender 1    Defender 2    ...    Defender 11
        # Offense 1  [10 feats]    [10 feats]    ...    [10 feats]
        # Offense 2  [10 feats]    [10 feats]    ...    [10 feats]
        #   ...         ...           ...        ...       ...
        # Offense 10 [10 feats]    [10 feats]    ...    [10 feats]
        #
        # The 10 features per interaction include:
        # - Defensive player velocity (2 features: vx, vy)
        # - Relative position: defender - ball carrier (2 features: dx, dy)
        # - Relative velocity: defender - ball carrier (2 features: dvx, dvy)
        # - Relative position: offensive blocker - defender (2 features: dx, dy)
        # - Relative velocity: offensive blocker - defender (2 features: dvx, dvy)
        #
        # This grid structure allows the Zoo model to learn pairwise offensive-defensive interactions,
        # but limits its ability to see complex multi-player patterns (e.g., 3 defenders converging).
        assert x.shape == (10, 11, 10), f"Expected shape (10, 11, 10), got {x.shape}"
        return x


def load_datasets(model_type: str, split: str) -> BDB2024_Dataset:
    """
    Load datasets for a specific model type and data split.

    Args:
        model_type (str): Type of model ('transformer' or 'zoo')
        split (str): Data split ('train', 'val', or 'test')

    Returns:
        BDB2024_Dataset: Loaded dataset for the specified model type and split

    Raises:
        ValueError: If an unknown split is specified
        FileNotFoundError: If the dataset file is not found
    """
    ds_dir = DATASET_DIR / model_type
    file_path = ds_dir / f"{split}_dataset.pkl"

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with open(file_path, "rb") as f:
        return pickle.load(f)


def main():
    """
    Main function to create and save datasets for different model types and splits.
    """
    for split in ["test", "val", "train"]:
        feature_df = pl.read_parquet(PREPPED_DATA_DIR / f"{split}_features.parquet")
        tgt_df = pl.read_parquet(PREPPED_DATA_DIR / f"{split}_targets.parquet")
        for model_type in ["zoo", "transformer"]:
            print(f"Creating dataset for {model_type=}, {split=}...")
            tic = time.time()
            dataset = BDB2024_Dataset(model_type, feature_df, tgt_df)
            out_dir = DATASET_DIR / model_type
            out_dir.mkdir(exist_ok=True, parents=True)
            with open(out_dir / f"{split}_dataset.pkl", "wb") as f:
                pickle.dump(dataset, f)
            print(f"Took {(time.time() - tic)/60:.1f} mins")


if __name__ == "__main__":
    main()
