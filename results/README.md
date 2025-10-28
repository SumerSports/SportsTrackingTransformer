# Results

All results are in `results.json` as a flat array of objects (easily convertible to dataframe).

## Format

```json
[
  {
    "split": "train|val|test|test-{event}",
    "metric": "ade_yards",
    "zoo": ...,
    "transformer": ...,
    "improvement_pct": ...,
    "improvement_yards": ...,
    "n_plays": ...,
    "n_frames": ...
  }
]
```

**Notes:**
- Only `mirrored=False` data is used (mirrored data is horizontal flip augmentation - we filter to avoid double-counting the same physical plays)
- `n_frames` = number of predictions evaluated (includes predictions from both zoo and transformer models)
- For main splits (train/val/test): `n_frames` includes all frames across all plays
- For event splits (test-tackle, test-fumble, etc.): we only evaluate at the specific frame where the event occurred, so `n_frames = 2 × n_plays` (one prediction per model)
- Event splits are sorted by `n_plays` (descending)

## Main Results

**Test Set (Overall):**
- Zoo: 5.71 yards ADE
- Transformer: 4.57 yards ADE
- **Improvement: 1.14 yards (20.0%)**
- 1,871 plays / 160,124 predictions

**Test Set (By Event - at event frame only):**
- `test-tackle`: 74.7% improvement (1,497 plays)
- `test-out_of_bounds`: 68.7% improvement (272 plays)
- `test-touchdown`: 54.1% improvement (67 plays)
- `test-qb_slide`: 47.4% improvement (21 plays)
- `test-fumble`: 41.1% improvement (14 plays)

## Metric

**ADE (Average Displacement Error):** Mean Euclidean distance between predicted and actual tackle locations.

Formula: `mean(sqrt((x_pred - x_true)² + (y_pred - y_true)²))`

Standard metric for spatial prediction tasks in computer vision and robotics.

## Regenerating

To regenerate these results from the trained models:

```bash
uv run python src/generate_results_summary.py
```
