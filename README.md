# Attention is All You Need, for Sports Spacial Tracking Data
## Introduction

The world of sports tracking data is exploding, and there is a strong need to finding a Deep Neural Net Framework that can be applied to a wide-range of Sports problems in a simple and efficient way. At SumerSports, we were faced with this challenge 2 years ago, we started by looking at the commonly-used State-of-The-Art Architecture at that time: the Zoo Model that won the 2020 Big Data Bowl. After implementing and deeply breaking down this approach, we found it that while it was a great architecture for the problem and dataset it was designed for, it had many shortcomings and did not *generalize* or scale well to new problems with much more data available well at all. From there, we discovered that many of the shortcomings of The Zoo Model can be solved by applying the Transformer Encoder Architecture to the data. This repository demonstrates how to apply the Transformer Encoder Architecture to sports tracking data and compares it to the Zoo Model. The problem we chose to solve is predicting tackle location using the 2024 Big Data Bowl dataset as it is a much more rich dataset than the one in 2020. We believe these results will generalize to other problems using NFL tracking data but also to other sports and look forward to seeing it applied elsewhere.

## Getting Started With The Code
This repo uses `rye` to manage environment, code, and dependencies. Start off by [installing it](https://rye-up.com/guide/installation/):
```bash
curl -sSf https://rye-up.com/get | bash
```
Follow instructions to add `rye` to your path and open a new terminal.

After cloning the repo, use the command `rye sync`. This will create a `.venv` and populate it with the needed dependencies.
If you want to activate the venv, you can do `. .venv/bin/activate` else just use `rye run` to ensure a command runs in the venv

You will need to use `az login` or manual `config.local` and then `dvc pull dvc.yaml:pick_best_models` to grab the latest run.
If you want to skip this, you can also just use `rye run dvc repro` to kick off the pipeline from scratch.

The notebook `results_analysis` has some viz and results.