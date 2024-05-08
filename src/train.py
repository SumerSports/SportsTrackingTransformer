import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

import lightning.pytorch.callbacks as callbacks
import numpy as np
import polars as pl
import torch
from datasets import BDB2024_Dataset
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from models import LitModel
from torch.utils.data import DataLoader

DATASET_DIR = Path("data/datasets")
MODELS_PATH = Path("data/models")
MODELS_PATH.mkdir(exist_ok=True)


def predict_model_as_df(
    model: LitModel,
    dataloaders: Dict[str, DataLoader],
    ckpt_path=None,
) -> pl.DataFrame:
    pred_dfs = []
    for split, dataloader in dataloaders.items():
        preds = Trainer(devices=1, logger=False).predict(model, dataloaders=dataloader, ckpt_path=ckpt_path)
        preds: np.ndarray = torch.concat(preds, dim=0).cpu().numpy()

        dataset: BDB2024_Dataset = dataloader.dataset
        tgt_df = pl.from_pandas(dataset.tgt_df_partition, include_index=True)
        ds_keys = np.array(dataset.keys)

        assert preds.shape[0] == ds_keys.shape[0], f"Pred Shape: {preds.shape}, Keys Shape: {ds_keys.shape}"

        pred_df = (
            pl.DataFrame(
                {
                    "gameId": ds_keys[:, 0],
                    "playId": ds_keys[:, 1],
                    "mirrored": ds_keys[:, 2],
                    "frameId": ds_keys[:, 3],
                    "tackle_x_rel_pred": preds[:, 0].round(2),
                    "tackle_y_rel_pred": preds[:, 1].round(2),
                }
            )
            .with_columns(
                pl.col("mirrored").cast(bool),
                pl.lit(split).alias("dataset_split"),
            )
            .join(tgt_df, on=["gameId", "playId", "mirrored"], how="inner")
        )

        assert pred_df.shape[0] == len(dataset)

        pred_dfs.append(pred_df)

    pred_dfs = pl.concat(pred_dfs, how="vertical")
    return pred_dfs


def train_model(
    model_type,
    batch_size,
    hidden_dim,
    num_layers,
    learning_rate,
    dropout,
    use_play_features=False,
    device=0,
    dbg_run=False,
):
    # load datasets
    ds_dir = DATASET_DIR / f"{model_type}{'_play' if use_play_features else ''}"
    with open(ds_dir / "train_dataset.pkl", "rb") as f:
        train_ds: BDB2024_Dataset = pickle.load(f)
    with open(ds_dir / "val_dataset.pkl", "rb") as f:
        val_ds: BDB2024_Dataset = pickle.load(f)
    with open(ds_dir / "test_dataset.pkl", "rb") as f:
        test_ds: BDB2024_Dataset = pickle.load(f)
    # convert to dataloaders for model fit
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=30)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=30)

    feature_len = train_ds[0][0].shape[-1]
    lit_model = LitModel(
        model_type,
        feature_len=feature_len,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        learning_rate=learning_rate,
        dropout=dropout,
    )

    devices = [device] if device >= 0 else "auto"  # if device is specified, use it, otherwise find 1 available GPU
    if dbg_run:
        # debug run
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

    logger = TensorBoardLogger(
        save_dir=MODELS_PATH,
        name=f"{model_type}{'_play' if use_play_features else ''}",
        log_graph=False,
        default_hp_metric=False,
        version=f"H{hidden_dim}_L{num_layers}",
    )
    trainer = Trainer(
        max_epochs=1000,
        accelerator="gpu",
        profiler=None,
        logger=logger,
        devices=devices,
        # strategy="ddp_find_unused_parameters_true",
        enable_model_summary=False,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=5),
            callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, filename="{epoch}-{val_loss:.1f}"),
            callbacks.ModelSummary(max_depth=3),
        ],
    )

    print(lit_model.get_hyperparams())
    logger.log_hyperparams(lit_model.get_hyperparams())
    trainer.fit(lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Model Predict
    # Recreate unshuffled dataloaders for prediction
    pred_dataloaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=30),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=10),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=10),
    }
    best_model_path = trainer.checkpoint_callback.best_model_path
    model_pred_df = predict_model_as_df(lit_model, dataloaders=pred_dataloaders, ckpt_path=best_model_path)
    model_pred_df.write_parquet(Path(best_model_path).parent.parent / "model_preds.parquet")

    return lit_model


def main(args):
    lr = 1e-04
    for hidden_dim in [16, 32, 64, 128]:
        for num_layers in [1, 2, 4, 8]:
            train_model(
                args.model_type,
                args.batch_size,
                hidden_dim,
                num_layers,
                lr,
                dropout=0.3,
                use_play_features=args.play_features,
                device=args.device,
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--model_type", type=str, default="transformer")
    parser.add_argument("--play_features", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()
    main(args)
