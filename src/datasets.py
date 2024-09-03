import multiprocessing as mp
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from torch.utils.data import Dataset
from tqdm import tqdm


class BDB2024_Dataset(Dataset):
    def __init__(
        self,
        model_type,
        feature_df: pl.DataFrame,
        tgt_df: pl.DataFrame,
    ):
        self.model_type = model_type
        self.keys = list(feature_df.select(["gameId", "playId", "mirrored", "frameId"]).unique().rows())

        # convert to pandas form with index so we can quickly grab a set of rows by key
        self.feature_df_partition = (
            feature_df.to_pandas(use_pyarrow_extension_array=True)
            .set_index(["gameId", "playId", "mirrored", "frameId", "nflId"])
            .sort_index()
        )
        self.tgt_df_partition = (
            tgt_df.to_pandas(use_pyarrow_extension_array=True)
            # .partition_by(['gameId', 'playId', 'mirrored'], as_dict=True, maintain_order=False)
            .set_index(["gameId", "playId", "mirrored"])
            .sort_index()
        )

        # precompute features and store in dicts
        self.tgt_arrays = {}
        self.feature_arrays = {}
        with mp.Pool(processes=8) as pool:
            results = pool.map(
                self.process_key,
                tqdm(self.keys, desc="Pre-computing feature transforms", total=len(self.keys)),
            )
            # Unpack results
            for key, tgt_array, feature_array in results:
                self.tgt_arrays[key[:-1]] = tgt_array
                self.feature_arrays[key] = feature_array

    def process_key(self, key):
        tgt_array = self.transform_target_df(self.tgt_df_partition.loc[key[:-1]])
        feature_array = self.transform_input_frame_df(self.feature_df_partition.loc[key])
        return (key, tgt_array, feature_array)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]:
        # convert idx to key and grab the precomputed arrays
        key = self.keys[idx]
        return self.feature_arrays[key], self.tgt_arrays[key[:-1]]

    def transform_input_frame_df(self, frame_df: pd.DataFrame) -> np.ndarray:
        # needs to be implemented differently for each model architecture
        # convert a frame_df to a numpy array
        if self.model_type == "transformer":
            return self.transformer_transform_input_frame_df(frame_df)
        elif self.model_type == "zoo":
            return self.zoo_transform_input_frame_df(frame_df)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def transform_target_df(self, tgt_df: pd.DataFrame) -> np.ndarray:
        # should be same for all model architectures
        y = tgt_df[["tackle_x_rel", "tackle_y_rel"]].to_numpy(dtype=np.float32).squeeze()
        assert y.shape == (2,)
        return y

    def transformer_transform_input_frame_df(self, frame_df: pd.DataFrame) -> np.ndarray:
        features = ["x_rel", "y_rel", "vx", "vy", "side", "is_ball_carrier"]

        x = frame_df[features].to_numpy(dtype=np.float32)
        assert x.shape == (22, len(features))
        return x

    def zoo_transform_input_frame_df(self, frame_df: pd.DataFrame) -> np.ndarray:
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

        assert x.shape == (10, 11, 10)
        return x


def load_datasets(model_type: str, split: str) -> BDB2024_Dataset:
    ds_dir = Path("data/datasets") / model_type
    if "train" in split:
        with open(ds_dir / "train_dataset.pkl", "rb") as f:
            return pickle.load(f)
    elif "val" in split:
        with open(ds_dir / "val_dataset.pkl", "rb") as f:
            return pickle.load(f)
    elif "test" in split:
        with open(ds_dir / "test_dataset.pkl", "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unknown split: {split}")


def main():
    PREPPED_DATA_DIR = Path("data/split_prepped_data/")
    DATASET_DIR = Path("data/datasets/")
    for split in ["test", "val", "train"]:
        feature_df = pl.read_parquet(PREPPED_DATA_DIR / f"{split}_features.parquet")
        tgt_df = pl.read_parquet(PREPPED_DATA_DIR / f"{split}_targets.parquet")
        for model_type in ["zoo", "transformer"]:
            print(f"Creating {model_type} {split} dataset...")
            tic = time.time()
            dataset = BDB2024_Dataset(model_type, feature_df, tgt_df)
            out_dir = DATASET_DIR / f"{model_type}"
            out_dir.mkdir(exist_ok=True, parents=True)
            with open(out_dir / f"{split}_dataset.pkl", "wb") as f:
                pickle.dump(dataset, f)
            print(f"Took {(time.time() - tic)/60:.1f} mins")


if __name__ == "__main__":
    main()
