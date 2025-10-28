"""
Evaluation Metrics for Sports Tracking Transformer

Provides standard evaluation metrics for spatial prediction tasks.
Primary metric: Average Displacement Error (ADE) - mean Euclidean distance.
"""

import numpy as np
import polars as pl
from numpy.typing import NDArray


def calculate_ade(
    x: pl.Series | NDArray,
    y: pl.Series | NDArray,
    x_pred: pl.Series | NDArray,
    y_pred: pl.Series | NDArray,
) -> float:
    """
    Calculate Average Displacement Error (ADE).

    ADE = mean Euclidean distance between predicted and true (x, y) locations.
    Standard metric for trajectory prediction, pose estimation, and spatial tasks.

    Formula: mean(sqrt((x_pred - x)² + (y_pred - y)²))
    """
    if isinstance(x, pl.Series):
        x = x.to_numpy()
    if isinstance(y, pl.Series):
        y = y.to_numpy()
    if isinstance(x_pred, pl.Series):
        x_pred = x_pred.to_numpy()
    if isinstance(y_pred, pl.Series):
        y_pred = y_pred.to_numpy()

    distances = np.sqrt((x_pred - x) ** 2 + (y_pred - y) ** 2)
    return float(np.mean(distances))


def calculate_mse(
    x: pl.Series | NDArray,
    y: pl.Series | NDArray,
    x_pred: pl.Series | NDArray,
    y_pred: pl.Series | NDArray,
) -> float:
    """
    Calculate Mean Squared Error (MSE).

    Formula: mean((x_pred - x)² + (y_pred - y)²)
    """
    if isinstance(x, pl.Series):
        x = x.to_numpy()
    if isinstance(y, pl.Series):
        y = y.to_numpy()
    if isinstance(x_pred, pl.Series):
        x_pred = x_pred.to_numpy()
    if isinstance(y_pred, pl.Series):
        y_pred = y_pred.to_numpy()

    squared_distances = (x_pred - x) ** 2 + (y_pred - y) ** 2
    return float(np.mean(squared_distances))
