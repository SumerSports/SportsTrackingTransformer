# Model Results

**To regenerate these results:**
```bash
uv run python src/generate_results_summary.py
```
This generates `results.json` (performance metrics) and `model_comparison.json` (all trained models).

---

## Metric: Average Displacement Error (ADE)

**ADE** measures the mean Euclidean distance in yards between predicted and actual tackle locations.

$$\text{ADE} = \frac{1}{N}\sum_{i=1}^{N} \sqrt{(x_{\text{pred}}^{(i)} - x_{\text{true}}^{(i)})^2 + (y_{\text{pred}}^{(i)} - y_{\text{true}}^{(i)})^2}$$

where $N$ is the number of predictions, and $(x_{\text{pred}}, y_{\text{pred}})$ and $(x_{\text{true}}, y_{\text{true}})$ are the predicted and ground truth tackle locations in yards.

This is a standard metric for spatial prediction tasks in computer vision and robotics. Lower values indicate more accurate predictions.

---

## Model Selection

Our goal was to compare Transformer and Zoo architectures at **similar model sizes** for a fair evaluation. We conducted a comprehensive hyperparameter sweep (model_dim ∈ {32, 128, 512}, num_layers ∈ {1, 2, 4, 8}) across both architectures, training 24 models total. However, we discovered that Zoo architecture does not benefit from scaling, necessitating comparison between the best version of each architecture.

### What We Found

**Zoo architecture does not scale well.** The best Zoo model is the relatively small **M128_L2** (72K params, 4.2M FLOPs, 5.71 yards ADE). Larger Zoo models consistently performed worse - for example, scaling up to M512_L8 (5.9M params, 256M FLOPs) resulted in **5.93 yards ADE**, actually degrading performance despite being 81× larger.

**Transformer architecture scales effectively.** The best Transformer model is **M512_L2** (6.6M params, 139M FLOPs, 4.57 yards ADE), showing clear improvements with increased model capacity.

### Fair Comparison

Given Zoo's inability to scale, we selected:
- **Zoo M128_L2**: Best Zoo configuration (72K params, 5.71 yards)
- **Transformer M512_L2**: Best Transformer configuration (6.6M params, 4.57 yards)

While these models differ in size (Transformer is 91× larger in parameters, 33× larger in FLOPs), this comparison uses the **optimal version of each architecture**.

**Importantly:** Even if we were **forced to pick a smaller Transformer** than the best Zoo model, the **Transformer M32_L4** (52K params, 1.1M FLOPs, **4.90 yards ADE**) still **handily outperforms** the best Zoo M128_L2 (72K params, 4.2M FLOPs, **5.71 yards ADE**) - an improvement of **0.81 yards (14%)** with fewer parameters and 4× fewer FLOPs.

This demonstrates that the Transformer's architectural advantages (self-attention over all players) are fundamental, not just a result of model size.

### Complete Model Comparison (All 24 Trained Models)

The table below shows all hyperparameter configurations tested, demonstrating that:
1. **Zoo does not scale** - performance degrades with increased capacity (↑ indicates worse performance than smaller model)
2. **Transformer scales well** - performance generally improves with increased capacity
3. **Architecture matters more than size** - small Transformers outperform large Zoo models

| Architecture | Model Dim | Layers | Params | Inference FLOPs | Test ADE (yards) | Val Loss |
|--------------|-----------|--------|--------|-----------------|------------------|----------|
| **Zoo** | 32 | 1 | 2,910 | 180K | 6.68 | 3.896 |
| **Zoo** | 32 | 2 | 5,086 | 305K | 6.22 | 3.566 |
| **Zoo** | 32 | 4 | 11,678 | 556K | 6.03 | 3.428 |
| **Zoo** | 32 | 8 | 24,862 | 1.1M | 6.70 ↑ | 3.903 |
| **Zoo** | 128 | 1 | 39,222 | 2.2M | 5.86 | 3.311 |
| **Zoo** | **128** | **2** | **72,502** | **4.2M** | **5.71** | **3.221** ← **Best Zoo** |
| **Zoo** | 128 | 4 | 172,598 | 8.2M | 5.82 ↑ | 3.305 |
| **Zoo** | 128 | 8 | 372,790 | 16.2M | 5.85 ↑ | 3.347 |
| **Zoo** | 512 | 1 | 599,190 | 32.6M | 5.72 | 3.229 |
| **Zoo** | 512 | 2 | 1,125,526 | 64.4M | 5.76 ↑ | 3.282 |
| **Zoo** | 512 | 4 | 2,705,558 | 128.4M | 5.80 ↑ | 3.292 |
| **Zoo** | 512 | 8 | 5,865,622 | 256.3M | 5.93 ↑ | 3.374 |
| | | | | | | |
| **Transformer** | 32 | 1 | 14,358 | 287K | 5.14 | 2.885 |
| **Transformer** | 32 | 2 | 27,062 | 564K | 5.12 | 2.846 |
| **Transformer** | **32** | **4** | **52,470** | **1.1M** | **4.90** | **2.748** ← **Smaller than Zoo** |
| **Transformer** | 32 | 8 | 103,286 | 2.2M | 4.90 | 2.701 |
| **Transformer** | 128 | 1 | 220,206 | 4.4M | 4.69 | 2.588 |
| **Transformer** | 128 | 2 | 418,478 | 8.8M | 4.64 | 2.567 |
| **Transformer** | 128 | 4 | 815,022 | 17.5M | 4.60 | 2.561 |
| **Transformer** | 128 | 8 | 1,608,110 | 34.9M | 4.65 | 2.577 |
| **Transformer** | 512 | 1 | 3,485,838 | 69.8M | 4.69 | 2.616 |
| **Transformer** | **512** | **2** | **6,638,222** | **139.1M** | **4.57** | **2.554** ← **Best Transformer** |
| **Transformer** | 512 | 4 | 12,942,990 | 277.7M | 4.61 | 2.567 |
| **Transformer** | 512 | 8 | 25,552,526 | 555.0M | 4.66 | 2.588 |

### Key Observations

**1. Zoo Architecture Fails to Scale:**
- Optimal at **M128_L2** (72K params → 5.71 yards)
- Scaling to **M512_L8** (5.9M params, 81× larger) → **5.93 yards** - performance degrades
- Consistent pattern: adding capacity hurts performance beyond M128_L2

**2. Transformer Architecture Scales Successfully:**
- **M32_L1** (14K params) → 5.14 yards
- **M512_L2** (6.6M params) → 4.57 yards
- Clear improvement trend with model capacity (11% better performance at 460× larger size)

**3. Architectural Superiority at Matched Sizes:**
- **Transformer M32_L4**: 52K params, 1.1M FLOPs → **4.90 yards**
- **Zoo M128_L2**: 72K params, 4.2M FLOPs → **5.71 yards**
- Transformer wins with **fewer parameters and 4× fewer FLOPs**

**4. Why Zoo Doesn't Scale:**

The Zoo architecture processes pairwise interactions between offensive (10) and defensive (11) players through a fixed 10×11 grid structure. Each interaction is treated independently, preventing the model from capturing higher-order relationships involving multiple players. Adding more parameters cannot overcome this fundamental architectural limitation.

In contrast, the Transformer's self-attention mechanism computes interactions between all 22 players simultaneously, allowing each player's representation to be informed by the entire field context. This flexible architecture naturally benefits from additional capacity, enabling the model to learn increasingly sophisticated spatial patterns.

---

## Overall Performance

Comparing the selected best models (Zoo M128_L2 vs Transformer M512_L2), the Transformer substantially outperforms the Zoo baseline across all data splits:

| Split | Zoo (yards) | Transformer (yards) | Improvement | Plays |
|-------|-------------|---------------------|-------------|-------|
| **Train** | 4.90 | 3.97 | **19.0%** (0.93 yards) | 8,735 |
| **Val** | 5.81 | 4.68 | **19.4%** (1.13 yards) | 1,872 |
| **Test** | 5.71 | 4.57 | **20.0%** (1.14 yards) | 1,871 |

---

## Test Set Performance by Event Type

The Transformer shows particularly strong improvements on specific tackle events, especially when predicting the exact tackle frame:

| Event | Zoo (yards) | Transformer (yards) | Improvement | Plays |
|-------|-------------|---------------------|-------------|-------|
| **Tackle** | 4.03 | 1.02 | **74.7%** (3.01 yards) | 1,497 |
| **Out of Bounds** | 5.37 | 1.68 | **68.7%** (3.69 yards) | 272 |
| **Touchdown** | 4.99 | 2.29 | **54.1%** (2.70 yards) | 67 |
| **QB Slide** | 5.00 | 2.63 | **47.4%** (2.37 yards) | 21 |
| **Fumble** | 5.14 | 3.03 | **41.1%** (2.11 yards) | 14 |

The Transformer's self-attention mechanism is particularly effective at capturing the complex spatial relationships between all players simultaneously, leading to more accurate tackle location predictions.

---

## Data Files

### results.json

Performance metrics for selected models across all data splits.

**Format:**
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
- Only `mirrored=False` data is used (mirrored data is horizontal flip augmentation - filtered to avoid double-counting)
- `n_frames` = total number of predictions evaluated (includes both models)
- For main splits (train/val/test): `n_frames` includes all frames across all plays
- For event splits (test-tackle, test-fumble, etc.): only the specific frame where the event occurred is evaluated, so `n_frames = 2 × n_plays`
- Event splits are sorted by `n_plays` (descending)

### model_comparison.json

Comprehensive comparison of all 24 trained models with computational metrics.

**Format:**
```json
[
  {
    "model_type": "zoo|transformer",
    "model_dim": ...,
    "num_layers": ...,
    "params": ...,
    "inference_flops": ...,
    "test_ade_yards": ...,
    "val_loss": ...
  }
]
```

**Notes:**
- `params`: Total trainable parameters
- `inference_flops`: Floating point operations per single frame inference (batch_size=1)
- Calculated using Facebook's `fvcore.nn.FlopCountAnalysis`
- Sorted by model_type, then params
