"""
Training Script for NFL Big Data Bowl 2024 Tackle Prediction Models

This module handles the training process for tackle prediction models. It includes
functions for loading datasets, predicting using trained models, and conducting
hyperparameter searches.

Functions:
    predict_model_as_df: Generate predictions using a trained model and return results as a DataFrame
    train_model: Train a single model with specified hyperparameters
    main: Main execution function for hyperparameter search and model training

Classes:
    None (uses classes from other modules)
"""

import random
import re
from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import lightning.pytorch.callbacks as callbacks
import numpy as np
import polars as pl
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import BDB2024_Dataset, load_datasets
from models import LitModel

MODELS_PATH = Path("models")
MODELS_PATH.mkdir(exist_ok=True)


def predict_model_as_df(model: LitModel = None, ckpt_path: Path = None, devices=1) -> pl.DataFrame:
    """
    Generate predictions using a trained model and return results as a DataFrame.

    Args:
        model (LitModel, optional): Trained model instance. Defaults to None.
        ckpt_path (Path, optional): Path to model checkpoint. Defaults to None.
        devices (int or list): Devices to use for prediction. Defaults to 1.

    Returns:
        pl.DataFrame: DataFrame containing model predictions and metadata.

    Raises:
        AssertionError: If neither model nor ckpt_path is provided, or if multiple devices are specified.
    """
    assert model is not None or ckpt_path is not None, "Must provide either model or ckpt_path"
    if isinstance(devices, list):
        assert len(devices) == 1, "Only one device should be used for prediction"

    if model is None:
        model = LitModel.load_from_checkpoint(ckpt_path)

    # Load datasets
    train_ds: BDB2024_Dataset = load_datasets(model.model_type, split="train")
    val_ds: BDB2024_Dataset = load_datasets(model.model_type, split="val")
    test_ds: BDB2024_Dataset = load_datasets(model.model_type, split="test")

    # Create unshuffled dataloaders for prediction
    dataloaders = {
        "train": DataLoader(train_ds, batch_size=1024, shuffle=False, num_workers=10),
        "val": DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=10),
        "test": DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=10),
    }

    pred_dfs = []
    for split, dataloader in dataloaders.items():
        # Generate predictions
        pred_trainer = Trainer(devices=devices, logger=False, enable_model_summary=False)
        preds = pred_trainer.predict(model, dataloaders=dataloader, ckpt_path=ckpt_path)
        preds: np.ndarray = torch.concat(preds, dim=0).cpu().numpy()

        # Prepare metadata
        dataset: BDB2024_Dataset = dataloader.dataset
        tgt_df = pl.from_pandas(dataset.tgt_df_partition, include_index=True)
        ds_keys = np.array(dataset.keys)

        assert preds.shape[0] == ds_keys.shape[0], f"Pred Shape: {preds.shape}, Keys Shape: {ds_keys.shape}"

        # Create prediction DataFrame
        pred_df = (
            tgt_df.join(
                pl.DataFrame(
                    {
                        "gameId": ds_keys[:, 0],
                        "playId": ds_keys[:, 1],
                        "mirrored": ds_keys[:, 2],
                        "frameId": ds_keys[:, 3],
                        "dataset_split": split,
                        "tackle_x_rel_pred": preds[:, 0].round(2),
                        "tackle_y_rel_pred": preds[:, 1].round(2),
                    },
                    schema_overrides={"mirrored": bool},
                ),
                on=["gameId", "playId", "mirrored", "frameId"],
                how="inner",
            )
            .with_columns(
                tackle_x_rel_pred=pl.col("tackle_x_rel_pred").round(2),
                tackle_y_rel_pred=pl.col("tackle_y_rel_pred").round(2),
                tackle_x_pred=(pl.col("tackle_x_rel_pred") + pl.col("anchor_x")).round(2),
                tackle_y_pred=(pl.col("tackle_y_rel_pred") + pl.col("anchor_y")).round(2),
            )
            # add model hparams to pred df
            .with_columns(**{k: pl.lit(v) for k, v in model.hparams.items()})
        )

        assert pred_df.shape[0] == len(dataset)
        pred_dfs.append(pred_df)

    return pl.concat(pred_dfs, how="vertical")


def get_val_loss_from_ckpt(ckpt_path):
    val_loss_pattern = re.compile(r"val_loss=([\d\.]+)")
    match = val_loss_pattern.search(ckpt_path.name)
    if match:
        try:
            return float(match.group(1).rstrip("."))
        except ValueError:
            print(f"Warning: Invalid val_loss in checkpoint name: {ckpt_path.name}")
            return float("inf")
    return float("inf")


def train_model(
    model_type,
    batch_size,
    model_dim,
    num_layers,
    learning_rate,
    dropout,
    device=0,
    dbg_run=False,
):
    """
    Train a single model with specified hyperparameters.

    Args:
        model_type (str): Type of model to train ('transformer' or 'zoo').
        batch_size (int): Batch size for training.
        model_dim (int): Dimension of the model's internal representations.
        num_layers (int): Number of layers in the model.
        learning_rate (float): Learning rate for optimization.
        dropout (float): Dropout rate for regularization.
        device (int, optional): GPU device to use. Defaults to 0.
        dbg_run (bool, optional): Whether to run in debug mode. Defaults to False.

    Returns:
        LitModel: Trained model instance.
    """
    # Load datasets
    train_ds: BDB2024_Dataset = load_datasets(model_type, split="train")
    val_ds: BDB2024_Dataset = load_datasets(model_type, split="val")

    # Create dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=30)
    val_dataloader = DataLoader(val_ds, batch_size=1024, shuffle=False, pin_memory=True, num_workers=30)

    # Initialize model
    lit_model = LitModel(
        model_type,
        batch_size=batch_size,
        model_dim=model_dim,
        num_layers=num_layers,
        learning_rate=learning_rate,
        dropout=dropout,
    )

    # Set up devices
    devices = [device] if device >= 0 else [0, 1]  # if device is specified, use it, otherwise pick 1 gpu to use

    if dbg_run:
        # Debug run with limited epochs and profiling
        dbg_trainer = Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=1,
            profiler="simple",
            fast_dev_run=True,
            enable_model_summary=False,
            num_sanity_val_steps=3,
        )
        dbg_trainer.fit(lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Set up logger and trainer for full run
    logger = TensorBoardLogger(
        save_dir=MODELS_PATH,
        name=model_type,
        log_graph=False,
        default_hp_metric=False,
        version=f"M{model_dim}_L{num_layers}_LR{learning_rate:.0e}",
    )

    # Check for existing checkpoint with best val_loss
    ckpt_dir = Path(logger.log_dir) / "checkpoints"
    existing_ckpt = None
    if ckpt_dir.exists():
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        if ckpts:
            # Find the checkpoint with the lowest val_loss
            best_ckpt = min(ckpts, key=get_val_loss_from_ckpt)
            existing_ckpt = str(best_ckpt)
            print(f"Resuming training from best checkpoint: {existing_ckpt}")

    trainer = Trainer(
        max_epochs=100,
        accelerator="gpu",
        profiler=None,
        logger=logger,
        devices=devices,
        sync_batchnorm=True,
        enable_model_summary=True,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=5),
            callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, filename="{epoch}-{val_loss:.3f}"),
            callbacks.ModelSummary(max_depth=2),
        ],
    )

    # Train the model
    print(lit_model.get_hyperparams())
    logger.log_hyperparams(lit_model.get_hyperparams())
    trainer.fit(lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=existing_ckpt)

    # Generate and save predictions for the best model
    best_ckpt_path = Path(trainer.checkpoint_callback.best_model_path)
    preds_df = predict_model_as_df(lit_model, best_ckpt_path, devices[:1])
    preds_df.write_parquet(best_ckpt_path.with_suffix(".results.parquet"))

    return lit_model


def main(args):
    """
    Main execution function for hyperparameter search and model training.

    Args:
        args (Namespace): Command-line arguments.

    This function sets up the hyperparameter search space, shuffles the combinations,
    and trains models for each combination. It supports both exhaustive grid search
    and random search based on the provided arguments.
    """
    # Define hyperparameter search space
    lrs = [1e-4]
    model_dims = [32, 128, 512]
    num_layers = [2, 4, 8]

    # Create gridsearch iterable
    gridsearch = list(product(lrs, model_dims, num_layers))
    if args.shuffle:
        random.shuffle(gridsearch)

    if args.hparam_search_iters > 0:
        # Perform random search if hparam_search_iters is specified
        gridsearch = gridsearch[: args.hparam_search_iters]

    # Train models for each hyperparameter combination
    for LR, M, L in tqdm(gridsearch, desc="Hyperparam Gridsearch"):
        train_model(
            model_type=args.model_type,
            batch_size=256,
            model_dim=M,
            num_layers=L,
            learning_rate=LR,
            dropout=0.3,
            device=args.device,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--device", type=int, default=-1, help="GPU device to use (-1 for CPU)")
    parser.add_argument(
        "--hparam_search_iters", type=int, default=-1, help="Number of random hyperparameter combinations to try"
    )
    parser.add_argument("--shuffle", "-s", action="store_true", help="Shuffle the hyperparameter gridsearch")
    parser.add_argument(
        "--model_type", type=str, default="transformer", help="Type of model to train ('transformer' or 'zoo')"
    )
    args = parser.parse_args()
    main(args)
