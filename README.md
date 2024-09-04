
# Attention is All You Need, for Sports Tracking Data

## Introduction

The world of sports tracking data is exploding, and there is a strong need to finding a Deep Neural Net Framework that can be applied to a wide-range of Sports Tracking problems in a reproducible, scaleable, and generalized way. At SumerSports, we were faced with this challenge 2 years ago, we started by looking at the commonly-used architecture in the NFL at that time: the Zoo Model that won the 2020 Big Data Bowl. After implementing and deeply breaking down this approach, we found it that while it was a great architecture for the problem and dataset it was designed for, it had many shortcomings in generalizability and being overly-regularized. From there, we discovered that many of the shortcomings of The Zoo Model can be solved by applying the Transformer Encoder Architecture to the data. This repository demonstrates how to apply the Transformer Encoder Architecture to sports tracking data and compares it to the Zoo Model. The problem we chose to solve is predicting tackle location using the 2024 Big Data Bowl dataset as it is a much more rich dataset than the one in 2020. We believe these results will generalize to other problems using NFL tracking data but also to other sports and look forward to seeing it applied elsewhere.

## Getting Started With The Code

This repo uses `uv` to manage python version, environment, and dependencies. Start off by [installing it](https://docs.astral.sh/uv/getting-started/installation/):

```bash
# On macOS and Linux.
$ curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
$ powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After cloning the repo and opening a new terminal inside the repo workspace, perform the following steps:

1. Run `uv python install` to install the python version pinned in the `.python-version` file.
2. Run `uv sync` to create a `.venv` and populate it with the locked dependencies.
3. Run `uv venv` and `source .venv/bin/activate` to create and activate the virtual environment.


The notebook `results_analysis` has some viz and results.