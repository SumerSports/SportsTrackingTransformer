"""
Dataset Module for NFL Big Data Bowl 2026

This module defines the BDB2026_Dataset class, which is used to load and preprocess
data for training machine learning models to predict ball landing locations.
It includes functionality for 'transformer' model type using position and kinematic features only.

Classes:
    BDB2026_Dataset: Custom dataset class for NFL tracking data

Functions:
    load_datasets: Load preprocessed datasets for a specific model type and data split
    main: Main execution function for creating and saving datasets

"""

import multiprocessing as mp
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from torch.utils.data import Dataset
from tqdm import tqdm

PREPPED_DATA_DIR = Path("data/split_prepped_data/")
DATASET_DIR = Path("data/datasets/")


class BDB2026_Dataset(Dataset):
    """
    Custom dataset class for NFL tracking data to predict ball landing locations.

    This class preprocesses and stores NFL tracking data for use in machine learning models.
    Uses only position and kinematic features.

    Attributes:
        model_type (str): Type of model ('transformer')
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
            model_type (str): Type of model (currently only 'transformer' supported)
            feature_df (pl.DataFrame): DataFrame containing feature data
            tgt_df (pl.DataFrame): DataFrame containing target data

        Raises:
            ValueError: If an invalid model_type is provided
        """
        if model_type not in ["transformer"]:
            raise ValueError("model_type must be 'transformer'")

        self.model_type = model_type
        self.keys = list(feature_df.select(["game_id", "play_id", "frame_id"]).unique().rows())

        # Convert to pandas form with index for quick row retrieval
        self.feature_df_partition = (
            feature_df.to_pandas(use_pyarrow_extension_array=True)
            .set_index(["game_id", "play_id", "frame_id", "nfl_id"])
            .sort_index()
        )
        self.tgt_df_partition = (
            tgt_df.to_pandas(use_pyarrow_extension_array=True)
            .set_index(["game_id", "play_id", "frame_id"])
            .sort_index()
        )

        # Precompute features and store in dicts
        self.tgt_arrays: dict[tuple, np.ndarray] = {}
        self.feature_arrays: dict[tuple, np.ndarray] = {}
        with mp.Pool(processes=min(8, mp.cpu_count())) as pool:
            results = pool.map(
                self.process_key,
                tqdm(self.keys, desc="Pre-computing feature transforms", total=len(self.keys)),
            )
            # Unpack results
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
        Transform input frame DataFrame to numpy array using position and kinematic features only.

        Args:
            frame_df (pd.DataFrame): Input frame DataFrame

        Returns:
            np.ndarray: Transformed input features

        Raises:
            ValueError: If an unknown model type is specified
        """
        if self.model_type == "transformer":
            return self.transformer_transform_input_frame_df(frame_df)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def transform_target_df(self, tgt_df: pd.DataFrame) -> np.ndarray:
        """
        Transform target DataFrame to numpy array (ball_land_x, ball_land_y).

        Args:
            tgt_df (pd.DataFrame): Target DataFrame

        Returns:
            np.ndarray: Transformed target values

        Raises:
            AssertionError: If the output shape is not as expected
        """
        y = tgt_df[["ball_land_x", "ball_land_y"]].to_numpy(dtype=np.float32).squeeze()
        assert y.shape == (2,), f"Expected shape (2,), got {y.shape}"
        return y

    def transformer_transform_input_frame_df(self, frame_df: pd.DataFrame) -> np.ndarray:
        """
        Transform input frame DataFrame for transformer model using position and kinematic features.
        Pads or truncates to a fixed number of players (22) for batching.

        Features used:
        - x, y: position
        - s: speed
        - a: acceleration
        - vx, vy: velocity components
        - ox, oy: orientation components

        Args:
            frame_df (pd.DataFrame): Input frame DataFrame

        Returns:
            np.ndarray: Transformed input features for transformer model (shape: [22, 8])

        Raises:
            AssertionError: If the output shape is not as expected
        """
        features = ["x", "y", "s", "a", "vx", "vy", "ox", "oy"]
        x = frame_df[features].to_numpy(dtype=np.float32)
        num_players = x.shape[0]

        # Pad or truncate to fixed size (22 players)
        max_players = 22
        if num_players < max_players:
            # Pad with zeros
            padding = np.zeros((max_players - num_players, len(features)), dtype=np.float32)
            x = np.vstack([x, padding])
        elif num_players > max_players:
            # Truncate (shouldn't happen with this data, but handle it)
            x = x[:max_players]

        assert x.shape == (max_players, len(features)), f"Expected shape ({max_players}, {len(features)}), got {x.shape}"
        return x


def load_datasets(model_type: str, split: str) -> "BDB2026_Dataset":
    """
    Load datasets for a specific model type and data split.

    Args:
        model_type (str): Type of model ('transformer')
        split (str): Data split ('train', 'val', or 'test')

    Returns:
        BDB2026_Dataset: Loaded dataset for the specified model type and split

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
    Main function to create and save datasets for transformer model.
    """
    for split in ["test", "val", "train"]:
        feature_df = pl.read_parquet(PREPPED_DATA_DIR / f"{split}_features.parquet")
        tgt_df = pl.read_parquet(PREPPED_DATA_DIR / f"{split}_targets.parquet")
        model_type = "transformer"
        print(f"Creating dataset for {model_type=}, {split=}...")
        tic = time.time()
        dataset = BDB2026_Dataset(model_type, feature_df, tgt_df)
        out_dir = DATASET_DIR / model_type
        out_dir.mkdir(exist_ok=True, parents=True)
        with open(out_dir / f"{split}_dataset.pkl", "wb") as f:
            pickle.dump(dataset, f)
        print(f"Took {(time.time() - tic)/60:.1f} mins")


if __name__ == "__main__":
    main()
