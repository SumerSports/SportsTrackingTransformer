
# Attention is All You Need, for Sports Tracking Data

## Introduction

The rapid advancement of spatial tracking technologies in sports has led to an unprecedented surge in high-quality, high-volume tracking data across all levels of play. While this data has catalyzed innovations in sports analytics, current methodologies often struggle with a fundamental challenge: the player-ordering problem. This issue arises from the dynamic nature of team sports, where player roles and formations are fluid and can vary between games, making it difficult to create consistent input structures for machine learning models.

This paper introduces a transformative approach to sports analytics by applying Transformer architectures to address these challenges. Our framework operates end-to-end on raw player tracking data, naturally handles unordered collections of player vectors, and is inherently designed to learn pairwise spatial interactions between players. Using the NFL's 2024 Big Data Bowl dataset, we demonstrate our approach's effectiveness in predicting tackle locations, showing significant improvements over commonly used architectures, particularly in generalizing to diverse game situations.

The repository contains our implementation and experimental results, providing a foundation for future research in sports analytics using Transformer-based architectures. We believe this approach can be extended beyond American football to other team sports, offering a more robust and generalizable framework for analyzing player tracking data.

![Simple Architecture Diagram](./Sumer%20Sports%20Transformer%20Simple%20Arch.jpg)

Our key contributions include:

* A minimal-feature-engineering approach to handling the player-ordering problem
* An end-to-end Transformer architecture adapted for sports tracking data
* Empirical evidence showing superior generalization compared to existing methods
* Open-source implementation for reproducibility and further research

We hope this work catalyzes a shift in sports analytics research methodologies, advancing our ability to derive meaningful insights from tracking data across various sports domains.

## Getting Started

This repo uses `uv` to manage python version, environment, and dependencies. Start off by [installing it](https://docs.astral.sh/uv/getting-started/installation/):

```bash
# On macOS and Linux
$ curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
$ powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After cloning the repo and opening a new terminal inside the repo workspace, perform the following steps:

1. Run `uv sync` to create a `.venv` and install all dependencies
2. Run `source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows) to activate the virtual environment

## Reproducing the Pipeline

The full pipeline is managed by DVC. To reproduce from scratch:

```bash
# Download data (requires Kaggle credentials)
dvc pull data/nfl-big-data-bowl-2024.zip

# Run full pipeline
dvc repro
```

Individual stages:
```bash
# Data preparation
uv run python src/prep_data.py

# Dataset preprocessing (takes 1-2 hours)
uv run python src/datasets.py

# Train models
uv run python src/train.py --model_type zoo --device 0
uv run python src/train.py --model_type transformer --device 0

# Select best models
uv run python src/pick_best_models.py
```

## Results

Pre-computed results are available in the `results/` directory. See [`results/README.md`](results/README.md) for detailed performance metrics.

**Quick Summary (Test Set):**
- Zoo Architecture: 5.73 yards ADE
- Transformer: 4.58 yards ADE
- **Improvement: 1.15 yards (20.1%)**

## Analysis

The notebook `results/results_analysis.ipynb` contains visualizations and detailed analysis.

## Citation

If you use this work, please cite:

```bibtex
@article{ranasaria2024attention,
  title={Attention is All You Need, for Sports Tracking Data},
  author={Ranasaria, Udit and Vabishchevich, Pavel},
  journal={arXiv preprint},
  year={2024}
}
```
