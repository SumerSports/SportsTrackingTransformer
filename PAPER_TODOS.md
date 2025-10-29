# Paper Corrections TODO

This document tracks all corrections needed in the paper PDF: `Attention is All You Need, for Sports Tracking Data.pdf`

## Critical Fixes (Must Fix Before Publication)

### 1. ✅ Fix Table 2: Change MSE to ADE
**Location:** Page 8, Table 2 header

**Current:**
```
Table 2: Test Set Event-Frame Performance Comparison (Mean Squared Error)
```

**Should be:**
```
Table 2: Test Set Event-Frame Performance Comparison (Average Displacement Error in Yards)
```

**Impact:** This is the main known issue. The metric shown in the table is actually ADE (yards), not MSE. All values in the table are correct, just the header is wrong.

---

### 2. ✅ Fix Abstract: Change "MAE" to "ADE"
**Location:** Page 1, Abstract, last paragraph

**Current:**
> "outperforms TZA by 20.4% in our score of **Mean Average Error (MAE)**"

**Should be:**
> "outperforms TZA by 20.0% in our score of **Average Displacement Error (ADE)**"

**Issues:**
- Incorrect metric name: Should be "ADE" not "MAE"
- MAE typically means "Mean Absolute Error" in ML literature
- Also fix percentage: 20.4% → 20.0% (see #3 below)

---

### 3. ✅ Fix Abstract: Correct Improvement Percentage
**Location:** Page 1, Abstract, last paragraph

**Current:**
> "outperforms TZA by **20.4%**"

**Should be:**
> "outperforms TZA by **20.0%**"

**Calculation:**
- Zoo: 5.71 yards ADE
- Transformer: 4.57 yards ADE
- Improvement: (5.71 - 4.57) / 5.71 = 1.14 / 5.71 = 0.1996 = **19.96% ≈ 20.0%**

**Note:** 20.4% appears to be a typo. The actual improvement is 20.0% (or 19.96% if being precise).

---

## Medium Priority Fixes

### 4. ⚠️ Fix Loss Function Naming
**Location:** Page 7, Section 3.2, paragraph 1

**Current:**
> "both models share the same AdamW optimizer, learning rate, and **Huber Loss** function. The Huber Loss balances L1 Loss for outliers with L2 Loss for predictions close to the target"

**Suggested fix:**
> "both models share the same AdamW optimizer, learning rate, and **SmoothL1Loss** function (also known as Huber Loss). The SmoothL1Loss balances L1 Loss for outliers with L2 Loss for predictions close to the target, using a delta threshold of 1.0 (PyTorch default)"

**Rationale:**
- Code uses `torch.nn.SmoothL1Loss()` (models.py:322)
- While "Huber Loss" is correct conceptually, using the PyTorch class name improves reproducibility
- Adding the delta parameter (1.0) provides complete implementation details

---

### 5. ⚠️ Fix Event Names in Table 2
**Location:** Page 8, Table 2

**Current events listed:**
- ball snap
- handoff
- pass caught ← **Issue: This event doesn't exist in the data**
- first contact
- out of bounds
- tackle

**Actual events in the dataset:**
- ball_snap
- handoff
- run ← **Missing from paper**
- pass_arrived ← **Not in paper**
- pass_outcome_caught ← **Paper calls this "pass caught"**
- first_contact
- out_of_bounds
- tackle

**Suggested fix:** Either:

**Option A (Recommended):** Match the actual event names used in code/results:
```
Event: ball_snap, handoff, run, pass_arrived, pass_outcome_caught, first_contact, out_of_bounds, tackle
```

**Option B:** Add a note explaining aggregation:
```
Note: Event names shown are simplified for readability. "pass caught" aggregates "pass_arrived" and "pass_outcome_caught" events from the raw data.
```

---

## Minor Fixes / Clarifications

### 6. 📝 Clarify Model Naming Convention
**Location:** Throughout paper

**Current:** Models referred to as "M{dim}_L{layers}" (e.g., M128_L2)

**Code checkpoint names:** `M{dim}_L{layers}_LR{lr}` (e.g., M128_L2_LR1e-04)

**Suggested clarification:** Add footnote on first usage:
> "Models are named by their hyperparameters: M{model_dim}_L{num_layers}. Checkpoint files include the learning rate (e.g., M128_L2_LR1e-04) but we omit it here for brevity."

---

### 7. 📝 Add Metric Definition Earlier
**Location:** Consider moving to Section 3 (Methods)

**Current:** ADE is defined in the Results section

**Suggestion:** Define ADE in the Methods section (Section 3.2) before presenting results:

```markdown
**Evaluation Metric: Average Displacement Error (ADE)**

We evaluate model performance using Average Displacement Error (ADE), a standard metric for spatial prediction tasks:

$$\text{ADE} = \frac{1}{N}\sum_{i=1}^{N} \sqrt{(x_{\text{pred}}^{(i)} - x_{\text{true}}^{(i)})^2 + (y_{\text{pred}}^{(i)} - y_{\text{true}}^{(i)})^2}$$

where $N$ is the number of predictions, and coordinates are measured in yards. Lower ADE indicates more accurate tackle location predictions.
```

---

## Verification Checklist

Before finalizing paper, verify these consistency checks:

- [ ] All references to "MSE" or "Mean Squared Error" changed to "ADE" / "Average Displacement Error"
- [ ] All references to "MAE" or "Mean Average Error" changed to "ADE" / "Average Displacement Error"
- [ ] Abstract improvement percentage is 20.0% (not 20.4%)
- [ ] Loss function is called "SmoothL1Loss" or "Huber Loss (SmoothL1Loss)"
- [ ] Table 2 header says "Average Displacement Error" not "Mean Squared Error"
- [ ] Event names in Table 2 match actual data events (or clarified as aggregated)
- [ ] All numerical results match `results/RESULTS.md`

---

## Cross-Reference: Code is Correct

**Important:** The code implementation is correct. All issues listed above are in the paper PDF only.

**Code verification:**
- ✅ Uses `torch.nn.SmoothL1Loss()` - correct
- ✅ Calculates ADE in `generate_results_summary.py:22-55` - correct
- ✅ Event names from actual tracking data - correct
- ✅ 20.0% improvement in results - correct
- ✅ Test set: Zoo 5.71 yards, Transformer 4.57 yards - correct

---

## Quick Summary for Paper Authors

**What's wrong:** The paper has some terminology inconsistencies and one incorrect table header (MSE → ADE).

**What's right:** All the actual numbers, results, and conclusions are correct.

**Priority:** Fix Table 2 header (MSE → ADE) and Abstract (MAE → ADE, 20.4% → 20.0%) before publication.
