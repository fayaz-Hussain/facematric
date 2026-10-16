# FaceMetric — Scoring Math Quality Assurance Report (Phase 1)

**Date:** 2026-10-14  
**Scope:** Verification of the Golden Ratio scoring engine against PRD mathematical formulas, independent oracle validation, numerical boundaries, bilateral splitting discontinuity, and celebrity breakdown audit.  
**Execution Environment:** `qa/.venv` (Python 3.12.10, Pytest 7.4.3). Main production codebase was not modified.

---

## 1. Reference Table Verification

The formula under test is:
$$\text{score}(r) = \max\left(0.0, \, \min\left(100.0, \, 100.0 \times \left(1.0 - \frac{|r - \phi|}{\phi}\right)\right)\right), \quad \phi = 1.618$$

The plan's exact reference values were executed against `GoldenRatioScorer._score_feature`:

| Measured Ratio ($r$) | Plan Target Score (2 dec) | Production Scorer Output (`fs.score`) | Exact Formula (Unrounded) | Notes & Boundary Conditions |
| :---: | :---: | :---: | :---: | :--- |
| **1.618** | **100.00** | **100.0** | 100.0000 | Exact $\phi$ target |
| **1.500** | **92.71** | **92.7** | 92.7070 | Sub-$\phi$ ratio |
| **1.520** | **93.94** | **93.9** | 93.9431 | PRD Section 9 example ratio |
| **1.700** | **94.93** | **94.9** | 94.9320 | PRD Section 9 example ratio |
| **1.000** | **61.80** | **61.8** | 61.8047 | Linear deviation |
| **0.000** | **0.00** | **0.0** | 0.0000 | Lower zero-score boundary |
| **3.236** | **0.00** | **0.0** | 0.0000 | Upper zero-score boundary ($2\phi$) |
| **4.000** | **0.00** | **0.0** | -47.2188 | Clamped extreme ratio |

### PRD Example Values Audit (82.3 / 74.1)
- An exhaustive AST scan of all 30 test files in `backend/tests/` confirmed that **zero tests assert 82.3 or 74.1**.
- The PRD Section 9 example response payload contains numbers mathematically unreproducible by any valid formula. The backend implementation adheres to the mathematical formula, not the PRD text examples ([FM-BUG-001](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/BUGS.md#FM-BUG-001)).

---

## 2. Clamping, Invalid Inputs & Determinism

All tests are implemented in [qa/tests/test_phase1_scoring_math.py](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/tests/test_phase1_scoring_math.py):

1. **Upper & Lower Clamping:** Ratios ranging from $0.0$ to $10^6$ return scores strictly within $[0.0, 100.0]$.
2. **Controlled Error on Negative Ratios:** Ratios $< 0$ raise a controlled `ValueError: Measured ratio must be non-negative` via `FeatureScoreInternal.__post_init__`, preventing garbage input propagation.
3. **Non-Finite Inputs:** `NaN`, `+inf`, and `-inf` are handled gracefully via independent oracle logic without unhandled crashes.
4. **Determinism:** Over 10 consecutive executions on identical landmark inputs, `GoldenRatioScorer.score_all` produces bit-identical overall scores and feature breakdowns (0.000000 drift).
5. **Independent Oracle Comparison:** An independent implementation in [qa/eval/oracle.py](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/eval/oracle.py) evaluated 50 random ratios across $[0.1, 3.5]$. Production output matched the oracle within standard 1-decimal rounding ($\pm 0.1$ pt) across all 50 trials.

---

## 3. Strict Known Bug Verification (`xfail(strict=True)`)

Two tests in `qa/tests/test_phase1_scoring_math.py` were written to capture known bugs. Both tests **strictly failed** as required:

### 3.1. [FM-BUG-009](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/BUGS.md#FM-BUG-009): Overall Score Overweights Asymmetric Features
- **Test:** `test_overall_score_equals_equal_weighted_mean_of_8_features`
- **Result:** `XFAIL` (Strictly failed as expected).
- **Failure Details:**
  When a bilateral feature splits into Left and Right entries ($>10\%$ asymmetry), `len(feature_scores)` increases from 8 to 9. Line 175 of `golden_ratio_scorer.py` computes:
  $$\text{overall\_score} = \frac{\sum_{i=1}^{N} \text{score}_i}{N}, \quad N \in \{8, 9, 10\}$$
  Instead of preserving equal 12.5% weight for each of the 8 feature categories, the asymmetric feature receives two full slots ($2/9 = 22.2\%$ or $2/10 = 20.0\%$), while symmetric features are penalized down to $1/9 = 11.1\%$.

### 3.2. [FM-BUG-007](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/BUGS.md#FM-BUG-007): Ratio 7 Right Side Hardcoded to Left Side
- **Test:** `test_ratio_7_measures_right_side_independently`
- **Result:** `XFAIL` (Strictly failed as expected).
- **Failure Details:**
  In `golden_ratio_scorer.py:574`, `_measure_eye_to_brow_distance_bilateral` contains:
  ```python
  ratio_right = ratio_left  # Hardcoded!
  ```
  Modifying right eye and eyebrow landmarks (points 300, 386, 374) produced `r7_right_after == r7_right_init`. The right side is completely unmeasured, structurally preventing asymmetry detection on this feature.

---

## 4. Bilateral Split Rule Quantification & Mathematical Midpoint Flaw

### 4.1. The 10% Discontinuity Jump
Using synthetic landmarks on Ratio 3 (Eye Spacing ÷ Eye Width) where average ratio is maintained at $\phi = 1.618$ and all other 7 features remain fixed:

| Target Asymmetry | Measured Asymmetry | Breakdown Entries | Left Ratio / Score | Right Ratio / Score | Overall Score | Behavioral Impact |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **9.0%** | 9.00% | **8** | N/A (averaged) | N/A (averaged) | **51.80** | Symmetric ($N=8$); average score = 100.0 (weighted 12.5%). |
| **10.0%** | 10.00% | **9** | 1.537 / 95.0 | 1.699 / 95.0 | **56.00** | **Discontinuity jump!** Splits into 2 entries, increasing category weight from 12.5% to 22.2%. |
| **11.0%** | 11.00% | **9** | 1.529 / 94.5 | 1.707 / 94.5 | **55.90** | Smooth continuation after split ($N=9$). |

> [!WARNING]
> **Artificial Discontinuity:** A mere 1% increase in facial asymmetry from 9.0% to 10.0% causes the overall score to **jump upwards by +4.20 points (+8.1%)** simply because the high-scoring eye feature receives double representation in the divisor.

### 4.2. [FM-BUG-022](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/BUGS.md#FM-BUG-022): Ratio 2 Midpoint Geometry Bug
During bilateral quantification, an additional major mathematical flaw was uncovered in Ratio 2 (`_measure_cheekbone_to_jaw_bilateral`, lines 500–516):
- The implementation defines `face_center = (face_left + face_right) / 2` and `jaw_center = (jaw_left + jaw_right) / 2`.
- It measures `face_width_left = dist(face_left, face_center)` and `face_width_right = dist(face_center, face_right)`.
- By basic Euclidean geometry, the distance from an endpoint to its midpoint is identically $\frac{1}{2} \text{width}$:
  $$\|A - \frac{A+B}{2}\| \equiv \|\frac{A+B}{2} - B\| \equiv \frac{1}{2}\|A - B\|$$
- Therefore, `face_width_left` is identically equal to `face_width_right`, and `jaw_width_left` is identically equal to `jaw_width_right`.
- **Verdict:** Ratio 2 asymmetry is **permanently and mathematically locked to 0.0%** on every possible face.

### Summary of Claimed Bilateral Features:
1. **Ratio 2 (Cheekbone to Jaw):** Mathematically locked to 0.0% asymmetry by midpoint definition ([FM-BUG-022](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/BUGS.md#FM-BUG-022)).
2. **Ratio 7 (Eye to Brow):** Hardcoded to 0.0% asymmetry via `ratio_right = ratio_left` ([FM-BUG-007](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/BUGS.md#FM-BUG-007)).
3. **Ratio 3 (Eye Spacing):** The **only** bilateral feature that can ever split in production.

---

## 5. Celebrity Image Breakdown Audit

Three real celebrity photographs from `celebrity_dataset/pins_Adriana Lima/` were processed through real MediaPipe detection and `GoldenRatioScorer.score_all`:

| Image | Dimensions | Breakdown Entries | Overall Score | Split Features Detected | Asymmetry Details |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `Adriana Lima0_0.jpg` | $550 \times 668$ | **8** | 51.8 | None | All bilateral features $\le 10\%$. Equal weighting maintained. |
| `Adriana Lima101_3.jpg` | $302 \times 320$ | **9** | 52.4 | **Eye Spacing ÷ Eye Width** | Left: ratio 1.184 (score 73.2)<br>Right: ratio 1.403 (score 86.7)<br>**Asymmetry: 16.9%** ($>10\%$) |
| `Adriana Lima102_4.jpg` | $146 \times 155$ | **8** | 49.4 | None | All bilateral features $\le 10\%$. Equal weighting maintained. |

**Key Empirical Finding:** Real photographs trigger the 9-entry split in production (`Adriana Lima101_3.jpg`), directly confirming that real user faces experience the divisor skew and weighting distortion identified in [FM-BUG-009](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/BUGS.md#FM-BUG-009).

---

## 6. Phase 1 Test Execution Command & Summary

To reproduce all findings:
```powershell
.\qa\.venv\Scripts\pytest.exe qa/tests/test_phase1_scoring_math.py -v
```

**Results:**
- **21 passed**
- **2 xfailed (strict=True)**: `test_overall_score_equals_equal_weighted_mean_of_8_features` and `test_ratio_7_measures_right_side_independently`
- **0 failed**
- Duration: 10.68s
