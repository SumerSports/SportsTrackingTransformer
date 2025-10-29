"""
Model Architectures for NFL Big Data Bowl 2024 tackle prediction task

This module defines neural network architectures for predicting tackle
locations in NFL plays. It includes two main model types: SportsTransformer and
TheZooArchitecture, along with a shared LightningModule wrapper for training.

Classes:
    SportsTransformer: Generalized Transformer-based model for sports tracking data
    TheZooArchitecture: Baseline approach based on the winning solution of the NFL Big Data Bowl 2020
    LitModel: LightningModule wrapper for shared training functionality
"""

from typing import Any

import torch
from lightning import LightningModule
from torch import Tensor, nn, squeeze
from torch.optim import AdamW

torch.set_float32_matmul_precision("medium")


class SportsTransformer(nn.Module):
    """
    Transformer model that treats all 22 players as a sequence for tackle prediction.
    """

    def __init__(
        self,
        feature_len: int,
        model_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        """
        Initialize the SportsTransformer.

        Args:
            feature_len (int): Number of input features per player.
            model_dim (int): Dimension of the model's internal representations.
            num_layers (int): Number of transformer encoder layers.
            dropout (float): Dropout rate for regularization.
        """
        super().__init__()
        dim_feedforward = model_dim * 4
        num_heads = min(16, max(2, 2 * round(model_dim / 64)))  # attention is better optimized for even number of heads

        self.hyperparams = {
            "model_dim": model_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "dim_feedforward": dim_feedforward,
        }

        # Normalize input features
        self.feature_norm_layer = nn.BatchNorm1d(feature_len)

        # Embed input features to model dimension
        self.feature_embedding_layer = nn.Sequential(
            nn.Linear(feature_len, model_dim),
            nn.ReLU(),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )

        # Transformer Encoder
        # This component applies multiple layers of self-attention and feed-forward networks
        # to process player data in a permutation-equivariant manner.

        # Key properties:
        # 1. Player-order equivariance: The output maintains the same shape and player order as the input.
        # 2. Contextual feature extraction: Transforms initial player features into rich, context-aware representations.
        # 3. Inter-player relationships: Captures complex interactions between players across the field.
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

        # Pool across player dimension
        # We pool because this task is a single value across all players, you don't need to pool for all tasks.
        self.player_pooling_layer = nn.AdaptiveAvgPool1d(1)

        # Task-specific Decoder to predict tackle location.
        # self.decoder = nn.Sequential(
        #     nn.Linear(model_dim, model_dim // 4),
        #     nn.ReLU(),
        #     nn.Linear(model_dim // 4, 2),
        # )

        self.decoder = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim // 4),
            nn.ReLU(),
            nn.LayerNorm(model_dim // 4),
            nn.Linear(model_dim // 4, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the SportsTransformer.

        Args:
            x (Tensor): Input tensor of shape [batch_size, num_players, feature_len].

        Returns:
            Tensor: Predicted tackle location of shape [batch_size, 2].
        """
        # x: [B: batch_size, P: # of players, F: feature_len]
        B, P, F = x.size()

        # Normalize features
        x = self.feature_norm_layer(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B,P,F] -> [B,P,F]

        # Embed features
        x = self.feature_embedding_layer(x)  # [B,P,F] -> [B,P,M: model_dim]

        # Apply transformer encoder
        x = self.transformer_encoder(x)  # [B,P,M] -> [B,P,M]

        # Pool over player dimension
        x = squeeze(self.player_pooling_layer(x.permute(0, 2, 1)), -1)  # [B,M,P] -> [B,M]

        # Decode to predict tackle location
        x = self.decoder(x)  # [B,M] -> [B,2]

        return x


class TheZooArchitecture(nn.Module):
    """
    TheZooArchitecture represents a baseline model to compare against the SportsTransformer.
    It was the winning solution for the 2020 Big Data Bowl designed to predict run game yardage gained with an
    innovative (at the time) approach to solving player-equivariance problem. At a high level, the approach requires
    generating a set of pairwise interaction vectors between offense (10) and defense (11) players, applying feedforward layers to each
    interaction embedding independently, and then pooling across interaction dimensions to get to a final output.

    Based on: https://github.com/juancamilocampos/nfl-big-data-bowl-2020/blob/master/1st_place_zoo_solution_v2.ipynb
    """

    # 10 offensive players and 11 defensive players
    NUM_OFFENSE = 10
    NUM_DEFENSE = 11

    def __init__(
        self,
        feature_len: int,
        model_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        """
        Initialize the TheZooArchitecture.

        Args:
            feature_len (int): Number of input features in each interaction vector.
            model_dim (int): Dimension of the model.
            num_layers (int): Number of convolutional layers.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.hyperparams = {
            "model_dim": model_dim,
            "num_layers": num_layers,
        }

        # Normalize input features
        self.feature_norm_layer = nn.BatchNorm2d(feature_len)

        # Embed input features to model dimension
        self.feature_embedding_layer = nn.Sequential(
            nn.Linear(feature_len, model_dim),
            nn.ReLU(),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )

        # Feedforward Layer block 1 across all interaction vectors
        # Notably, this "CNN" is just a hacky way to apply a Linear or Feedforward layer across all interaction vectors.
        # It is not doing any convolution between neighboring players (as that would violate permutation equivariance).
        self.ff_block1 = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(in_channels=model_dim, out_channels=model_dim, kernel_size=(1, 1), stride=(1, 1)),
                    nn.ReLU(),
                )
                for _ in range(num_layers)
            ]
        )

        # Feedforward Layer block 2 after pooling across offensive players
        self.ff_block2 = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv1d(in_channels=model_dim, out_channels=model_dim, kernel_size=1, stride=1),
                    nn.ReLU(),
                    nn.BatchNorm1d(model_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.output_layer = nn.Sequential(
            *(
                [
                    nn.Sequential(
                        nn.Linear(model_dim, model_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(model_dim),
                    )
                    for _ in range(max(0, num_layers - 2))
                ]
                + [
                    nn.Dropout(dropout),
                    nn.Linear(model_dim, model_dim // 4),
                    nn.ReLU(),
                    nn.LayerNorm(model_dim // 4),
                    nn.Linear(model_dim // 4, 2),
                ]
            )
        )

        # Pooling layers for collapsing offensive and defensive dimensions
        # Created in __init__ (not forward()) so fvcore can trace FLOPs
        self.pool_offense_max = nn.MaxPool2d((1, self.NUM_OFFENSE))
        self.pool_offense_avg = nn.AvgPool2d((1, self.NUM_OFFENSE))
        self.pool_defense_max = nn.MaxPool1d(self.NUM_DEFENSE)
        self.pool_defense_avg = nn.AvgPool1d(self.NUM_DEFENSE)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the TheZooArchitecture.

        Args:
            x (Tensor): Input tensor of shape [B, O, D, F].

        Returns:
            Tensor: Output tensor of shape [B, 2].
        """
        # x: [B: batch_size, O: offense, D: defense, F: feature_len]
        B, O, D, F = x.size()  # B=Batch, O=Offense, D=Defense, F=Feature

        # Normalize features
        x = self.feature_norm_layer(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # [B,O,D,F] -> [B,O,D,F]
        # Embed features
        x = self.feature_embedding_layer(x)  # [B,O,D,F] -> [B,O,D,M: model_dim]

        # apply first block, pool and collapse offensive dimension
        x = self.ff_block1(x.permute(0, 3, 2, 1))  # [B,O,D,M] -> [B,M,D,O]
        # Zoo Authors mentioned using a weighted sum of max and avg pooling helped most (experimentally verified hparam)
        x = self.pool_offense_max(x) * 0.3 + self.pool_offense_avg(x) * 0.7  # [B,M,D,O] -> [B,M,D,1]
        x = x.squeeze(-1)  # [B,M,D,1] -> [B,M,D]

        # apply second block, pool and collapse defensive dimension
        x = self.ff_block2(x)  # [B,M,D] -> [B,M,D]
        x = self.pool_defense_max(x) * 0.3 + self.pool_defense_avg(x) * 0.7  # [B,M,D] -> [B,M,1]
        x = x.squeeze(-1)  # [B,M,1] -> [B,M]

        # apply decoder
        x = self.output_layer(x)  # [B,M] -> [B,2]
        assert x.shape == (B, 2)
        return x


class LitModel(LightningModule):
    """
    Lightning module for training and evaluating tackle prediction models.
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
            batch_size (int): Batch size for training and evaluation.
            model_dim (int): Dimension of the model's internal representations.
            num_layers (int): Number of layers in the model.
            dropout (float): Dropout rate for regularization.
            learning_rate (float): Learning rate for the optimizer.
        """
        super().__init__()
        self.model_type = model_type.lower()
        self.model_class = SportsTransformer if self.model_type == "transformer" else TheZooArchitecture
        self.feature_len = 6 if self.model_type == "transformer" else 10

        # Initialize model with architecture-specific parameters
        self.model = self.model_class(
            feature_len=self.feature_len,
            model_dim=model_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.example_input_array = (
            torch.randn((batch_size, 22, self.feature_len))
            if self.model_type == "transformer"
            else torch.randn((batch_size, 10, 11, self.feature_len))
        )

        self.learning_rate = learning_rate
        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.hparams["params"] = self.num_params
        for k, v in self.model.hyperparams.items():
            self.hparams[k] = v

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
        Perform a single training step.

        Args:
            batch (tuple[Tensor, Tensor]): Batch of input features and target locations.
            batch_idx (int): Index of the current batch.

        Returns:
            Tensor: Computed loss for the batch.
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
            batch (tuple[Tensor, Tensor]): Batch of input and target tensors.
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
            batch (tuple[Tensor, Tensor]): Batch of input and target tensors.
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
            batch (tuple[Tensor, Tensor]): Batch of input and target tensors.
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
        return AdamW(self.parameters(), lr=self.learning_rate)

    def get_hyperparams(self) -> dict[str, Any]:
        """
        Get the hyperparameters of the model.

        Returns:
            Dict[str, Any]: Dictionary of hyperparameters.
        """
        return self.hparams
