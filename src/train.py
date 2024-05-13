from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import lightning.pytorch.callbacks as callbacks
import numpy as np
import polars as pl
import torch
from datasets import BDB2024_Dataset, load_datasets
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from models import LitModel
from torch.utils.data import DataLoader
from tqdm import tqdm

MODELS_PATH = Path("data/models")
MODELS_PATH.mkdir(exist_ok=True)


def predict_model_as_df(model: LitModel = None, ckpt_path: Path = None) -> pl.DataFrame:
    """
    Have to provide one of model or ckpt_path
    """
    if model is None:
        model = LitModel.load_from_checkpoint(ckpt_path)

    train_ds: BDB2024_Dataset = load_datasets(model.model_type, model.advanced, split="train")
    val_ds: BDB2024_Dataset = load_datasets(model.model_type, model.advanced, split="val")
    test_ds: BDB2024_Dataset = load_datasets(model.model_type, model.advanced, split="test")
    # Recreate unshuffled dataloaders for prediction
    dataloaders = {
        "train": DataLoader(train_ds, batch_size=1024, shuffle=False, num_workers=10),
        "val": DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=10),
        "test": DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=10),
    }
    pred_dfs = []
    for split, dataloader in dataloaders.items():
        pred_trainer = Trainer(devices=1, logger=False, enable_model_summary=False)
        preds = pred_trainer.predict(model, dataloaders=dataloader, ckpt_path=ckpt_path)
        preds: np.ndarray = torch.concat(preds, dim=0).cpu().numpy()

        dataset: BDB2024_Dataset = dataloader.dataset
        tgt_df = pl.from_pandas(dataset.tgt_df_partition, include_index=True)
        ds_keys = np.array(dataset.keys)

        assert preds.shape[0] == ds_keys.shape[0], f"Pred Shape: {preds.shape}, Keys Shape: {ds_keys.shape}"

        pred_df = tgt_df.join(
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
            on=["gameId", "playId", "mirrored"],
            how="inner",
        ).with_columns(
            tackle_x_rel_pred=pl.col("tackle_x_rel_pred").round(1),
            tackle_y_rel_pred=pl.col("tackle_y_rel_pred").round(1),
            tackle_x_pred=(pl.col("tackle_x_rel_pred") + pl.col("play_origin_x")).round(1),
            tackle_y_pred=(pl.col("tackle_y_rel_pred") + pl.col("play_origin_y")).round(1),
            model_type=pl.lit(model.model_type),
            advanced=model.hparams["advanced"],
            batch_size=model.hparams["batch_size"],
            hidden_dim=model.hparams["hidden_dim"],
            num_layers=model.hparams["num_layers"],
            dropout=model.hparams["dropout"],
            learning_rate=model.hparams["learning_rate"],
        )

        assert pred_df.shape[0] == len(dataset)
        pred_dfs.append(pred_df)

    return pl.concat(pred_dfs, how="vertical")


def train_model(
    model_type,
    batch_size,
    hidden_dim,
    num_layers,
    learning_rate,
    dropout,
    advanced=False,
    device=0,
    dbg_run=False,
):
    train_ds: BDB2024_Dataset = load_datasets(model_type, advanced, split="test")
    val_ds: BDB2024_Dataset = load_datasets(model_type, advanced, split="val")

    # convert to dataloaders for model fit
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=100)
    val_dataloader = DataLoader(val_ds, batch_size=1024, shuffle=False, pin_memory=True, num_workers=30)

    # feature_len = train_ds[0][0].shape[-1]
    lit_model = LitModel(
        model_type,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        learning_rate=learning_rate,
        dropout=dropout,
        advanced=advanced,
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
        name=f"{model_type}{'_advanced' if advanced else ''}",
        log_graph=False,
        default_hp_metric=False,
        version=f"B{batch_size}_H{hidden_dim}_L{num_layers}_LR{learning_rate:.0e}_D{dropout}",
    )
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

    print(lit_model.get_hyperparams())
    logger.log_hyperparams(lit_model.get_hyperparams())
    trainer.fit(lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # record best model preds
    best_ckpt_path = Path(trainer.checkpoint_callback.best_model_path)
    preds_df = predict_model_as_df(lit_model, best_ckpt_path)
    preds_df.write_parquet(best_ckpt_path.with_suffix(".results.parquet"))

    return lit_model


def main(args):
    batch_sizes = [32]
    lrs = [1e-4]
    hidden_dims = [64, 128, 256, 512, 1024]
    num_layers = [2, 4, 8]

    # create gridsearch iterable
    gridsearch = product(batch_sizes, lrs, hidden_dims, num_layers)
    gridsearch_len = len(batch_sizes) * len(lrs) * len(hidden_dims) * len(num_layers)

    for B, LR, H, L in tqdm(gridsearch, desc="Hyperparam Gridsearch", total=gridsearch_len):
        train_model(
            model_type=args.model_type,
            batch_size=B,
            hidden_dim=H,
            num_layers=L,
            learning_rate=LR,
            dropout=0.3,
            advanced=args.advanced,
            device=args.device,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--model_type", type=str, default="transformer")
    parser.add_argument("--advanced", action="store_true")
    args = parser.parse_args()
    main(args)
