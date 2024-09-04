from typing import Any

import torch
from lightning import LightningModule
from torch import Tensor, nn, squeeze
from torch.optim import AdamW

torch.set_float32_matmul_precision("medium")


class SportsTransformerEncoder(nn.Module):
    """
    Transformer encoder for sports tracking data.
    """

    def __init__(
        self,
        feature_len: int,
        model_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        """
        Initialize the SportsTransformerEncoder.

        Args:
            feature_len (int): Number of input features.
            model_dim (int): Dimension of the model.
            num_layers (int): Number of transformer layers.
            dropout (float): Dropout rate.
        """
        super().__init__()
        dim_feedforward = model_dim * 4
        num_heads = max(1, model_dim // 32)
        self.hyperparams = {
            "model_dim": model_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "dim_feedforward": dim_feedforward,
        }

        self.feature_norm_layer = nn.BatchNorm1d(feature_len)

        self.feature_embedding_layer = nn.Sequential(
            nn.Linear(feature_len, model_dim),
            nn.ReLU(),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        # encoded dim should be # of players x model_dim
        # pool across player dim before decoding
        self.player_pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.decoder = nn.Sequential(
            nn.Linear(model_dim, model_dim // 4),
            nn.ReLU(),
            nn.Linear(model_dim // 4, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the SportsTransformerEncoder.

        Args:
            x (Tensor): Input tensor of shape [B, P, F].

        Returns:
            Tensor: Output tensor of shape [B, 2].
        """
        # x: [B: batch_size, P: # of players, F: feature_len]
        B, P, F = x.size()

        x = self.feature_norm_layer(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B,P,F] -> [B,P,F]
        x = self.feature_embedding_layer(x)  # [B,P,F] -> [B,P,H: model_dim]
        x = self.transformer_encoder(x)  # [B,P,H] -> [B,P,H]
        # pool over player dimension
        x = squeeze(self.player_pooling_layer(x.permute(0, 2, 1)), -1)  # [B,H,P] -> [B,H]
        x = self.decoder(x)  # [B,H] -> [B,2]
        # assert x.shape == (B, 2)
        return x

    def get_hyperparams(self) -> dict[str, Any]:
        """
        Get the hyperparameters of the model.

        Returns:
            Dict[str, Any]: Dictionary of hyperparameters.
        """
        return self.hyperparams


# Zoo Model code based on:
# https://github.com/juancamilocampos/nfl-big-data-bowl-2020/blob/master/1st_place_zoo_solution_v2.ipynb
class ZooSpacialEncoder(nn.Module):
    """
    Spatial encoder for the Zoo model architecture.
    """

    def __init__(
        self,
        feature_len: int,
        model_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        """
        Initialize the ZooSpacialEncoder.

        Args:
            feature_len (int): Number of input features.
            model_dim (int): Dimension of the model.
            num_layers (int): Number of convolutional layers.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.hyperparams = {
            "model_dim": model_dim,
            "num_layers": num_layers,
        }

        self.feature_norm_layer = nn.BatchNorm2d(feature_len)

        # expected input shape is [B, O=10, D=11, F]
        self.feature_embedding_layer = nn.Sequential(
            nn.Linear(feature_len, model_dim),
            nn.ReLU(),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )
        # after embedding raw features, we will move features to a channel dimension [B, H, O=10, D=11]
        # self.feature_norm_layer = nn.Sequential(
        #     nn.BatchNorm2d(model_dim),
        #     nn.Dropout(dropout),
        # )
        self.off_def_player_block = nn.Sequential(
            *[
                nn.Conv2d(in_channels=model_dim, out_channels=model_dim, kernel_size=(1, 1), stride=(1, 1)),
                nn.ReLU(),
            ]
            * num_layers,
        )
        self.def_player_block = nn.Sequential(
            *[
                nn.Conv1d(in_channels=model_dim, out_channels=model_dim, kernel_size=1, stride=1),
                nn.ReLU(),
                nn.BatchNorm1d(model_dim),
            ]
            * num_layers,
        )
        self.decoder = nn.Sequential(
            # *[
            #     nn.Linear(model_dim, model_dim),
            #     nn.ReLU(),
            #     nn.BatchNorm1d(model_dim),
            # ]
            # * max(0, num_layers // 3 - 1),
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.BatchNorm1d(model_dim),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim // 4),
            nn.ReLU(),
            nn.LayerNorm(model_dim // 4),
            nn.Linear(model_dim // 4, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the ZooSpacialEncoder.

        Args:
            x (Tensor): Input tensor of shape [B, O, D, F].

        Returns:
            Tensor: Output tensor of shape [B, 2].
        """
        # x: [B: batch_size, O: offense, D: defense, F: feature_len]
        B, O, D, F = x.size()  # B=Batch, O=Offense, D=Defense, F=Feature

        x = self.feature_norm_layer(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # [B,O,D,F] -> [B,O,D,F]
        x = self.feature_embedding_layer(x)  # [B,O,D,F] -> [B,O,D,H: model_dim]

        # apply first block, pool and collapse offensive dimension
        x = self.off_def_player_block(x.permute(0, 3, 2, 1))  # [B,O,D,H] -> [B,H,D,O]
        x = nn.MaxPool2d((1, O))(x) * 0.3 + nn.AvgPool2d((1, O))(x) * 0.7  # [B,H,D,O] -> [B,H,D,1]
        x = x.squeeze(-1)  # [B,H,D,1] -> [B,H,D]

        # apply second block, pool and collapse defensive dimension
        x = self.def_player_block(x)  # [B,H,D] -> [B,H,D]
        x = nn.MaxPool1d(D)(x) * 0.3 + nn.AvgPool1d(D)(x) * 0.7  # [B,H,D] -> [B,H,1]
        x = x.squeeze(-1)  # [B,H,1] -> [B,H]

        # apply decoder
        x = self.decoder(x)  # [B,H] -> [B,2]
        assert x.shape == (B, 2)
        return x

    def get_hyperparams(self) -> dict[str, Any]:
        """
        Get the hyperparameters of the model.

        Returns:
            Dict[str, Any]: Dictionary of hyperparameters.
        """
        return self.hyperparams


class LitModel(LightningModule):
    """
    Lightning module for training and evaluating models.
    """

    def __init__(
        self,
        model_type: str,
        batch_size: int,
        model_dim: int,
        num_layers: int,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
    ):
        """
        Initialize the LitModel.

        Args:
            model_type (str): Type of model ('transformer' or 'zoo').
            batch_size (int): Batch size for training.
            model_dim (int): Dimension of the model.
            num_layers (int): Number of layers in the model.
            dropout (float): Dropout rate.
            learning_rate (float): Learning rate for optimization.
        """
        super().__init__()
        self.model_type = model_type.lower()
        self.model_class = SportsTransformerEncoder if self.model_type == "transformer" else ZooSpacialEncoder
        self.feature_len = 6 if self.model_type == "transformer" else 10

        self.model = self.model_class(
            feature_len=self.feature_len, model_dim=model_dim, num_layers=num_layers, dropout=dropout
        )
        self.example_input_array = (
            torch.randn((batch_size, 22, self.feature_len))
            if self.model_type == "transformer"
            else torch.randn((batch_size, 10, 11, self.feature_len))
        )

        self.learning_rate = learning_rate
        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.hparams["params"] = self.num_params
        print(self.hparams)
        self.save_hyperparameters()
        self.loss_fn = torch.nn.SmoothL1Loss()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self.model(x)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Training step for the model.

        Args:
            batch (Tuple[Tensor, Tensor]): Batch of input and target tensors.
            batch_idx (int): Index of the current batch.

        Returns:
            Tensor: Computed loss.
        """
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Validation step for the model.

        Args:
            batch (Tuple[Tensor, Tensor]): Batch of input and target tensors.
            batch_idx (int): Index of the current batch.

        Returns:
            Tensor: Computed loss.
        """
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Test step for the model.

        Args:
            batch (Tuple[Tensor, Tensor]): Batch of input and target tensors.
            batch_idx (int): Index of the current batch.

        Returns:
            Tensor: Computed loss.
        """
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def predict_step(self, batch: tuple[Tensor, Tensor], batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """
        Prediction step for the model.

        Args:
            batch (Tuple[Tensor, Tensor]): Batch of input and target tensors.
            batch_idx (int): Index of the current batch.
            dataloader_idx (int): Index of the dataloader.

        Returns:
            Tensor: Predicted output tensor.
        """
        x, y = batch
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self) -> AdamW:
        """
        Configure the optimizer for training.

        Returns:
            AdamW: Configured optimizer.
        """
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def get_hyperparams(self) -> dict[str, Any]:
        """
        Get the hyperparameters of the model.

        Returns:
            Dict[str, Any]: Dictionary of hyperparameters.
        """
        return self.hparams
