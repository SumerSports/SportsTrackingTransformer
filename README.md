
# Attention is All You Need, for Sports Tracking Data

## Introduction

The rapid advancement of spatial tracking technologies in sports has led to an unprecedented surge in high-quality, high-volume tracking data across all levels of play. While this data has catalyzed innovations in sports analytics, current methodologies often struggle with a fundamental challenge: the player-ordering problem. This issue arises from the dynamic nature of team sports, where player roles and formations are fluid and can vary between games, making it difficult to create consistent input structures for machine learning models.

This paper introduces a transformative approach to sports analytics by applying Transformer architectures to address these challenges. Our framework operates end-to-end on raw player tracking data, naturally handles unordered collections of player vectors, and is inherently designed to learn pairwise spatial interactions between players. Using the NFL's 2024 Big Data Bowl dataset, we demonstrate our approach's effectiveness in predicting tackle locations, showing significant improvements over commonly used architectures, particularly in generalizing to diverse game situations.

The repository contains our implementation and experimental results, providing a foundation for future research in sports analytics using Transformer-based architectures. We believe this approach can be extended beyond American football to other team sports, offering a more robust and generalizable framework for analyzing player tracking data.

![Simple Architecture Diagram](./paper/Sumer%20Sports%20Transformer%20Simple%20Arch.jpg)

Our key contributions include:

* A minimal-feature-engineering approach to handling the player-ordering problem
* An end-to-end Transformer architecture adapted for sports tracking data
* Empirical evidence showing superior generalization compared to existing methods
* Open-source implementation for reproducibility and further research

We hope this work catalyzes a shift in sports analytics research methodologies, advancing our ability to derive meaningful insights from tracking data across various sports domains.

## Paper and Workshop Materials

This repository accompanies our research paper:
- **Full Paper**: [`Attention is All You Need, for Sports Tracking Data.pdf`](./paper/Attention%20is%20All%20You%20Need,%20for%20Sports%20Tracking%20Data.pdf)
- **LaTeX Source**: [`Attention is All You Need, for Sports Tracking Data.tex`](./paper/Attention%20is%20All%20You%20Need,%20for%20Sports%20Tracking%20Data.tex)
- **Workshop Slides**: [`CMSAC Workshop Notes.pdf`](./paper/CMSAC%20Workshop%20Notes.pdf) - Presented at the Carnegie Mellon Sports Analytics Conference

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
# Run full pipeline (downloads data, trains models, generates results)
uv run dvc repro
```

This runs all stages including training 24 model configurations (~8-12 hours on GPU).

**Note:** The original NFL Big Data Bowl 2024 dataset was removed from Kaggle by the host. For reproducibility, we've hosted the dataset in [GitHub Releases](https://github.com/SumerSports/SportsTrackingTransformer/releases/tag/data-v1.0). The DVC pipeline automatically downloads it from there.


## Pre-trained Models

Pre-trained models are available in [GitHub Releases](https://github.com/SumerSports/SportsTrackingTransformer/releases/tag/models-v1.0) (135MB compressed):
- Zoo Architecture best model (M128_L2)
- Transformer best model (M512_L2)
- Test set predictions for both models

Download and extract to skip training:
```bash
wget https://github.com/SumerSports/SportsTrackingTransformer/releases/download/models-v1.0/best_models.tar.gz
tar -xzf best_models.tar.gz
rm best_models.tar.gz
```

## Results

Pre-computed results are available in the `results/` directory. See [`results/RESULTS.md`](results/RESULTS.md) for:
- Detailed explanation of the ADE (Average Displacement Error) metric
- Comprehensive model comparison across all 24 trained configurations
- Per-event performance breakdown
- Analysis by frames before tackle

**Quick Summary (Test Set):**
- Zoo Architecture: 5.71 yards ADE
- Transformer: 4.57 yards ADE
- **Improvement: 1.14 yards (20.0%)**

**What is ADE?** Average Displacement Error measures the mean Euclidean distance (in yards) between predicted and actual tackle locations. Lower is better. See [`results/RESULTS.md`](results/RESULTS.md) for the full mathematical definition and interpretation.

## Analysis

The notebook [`results/animated_results_visualization.ipynb`](results/animated_results_visualization.ipynb) contains interactive visualizations and detailed analysis of model predictions.

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
