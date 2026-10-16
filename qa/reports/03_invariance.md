# Phase 3: Metamorphic Invariance Evaluation & Quality Gate Audit

**Status:** Completed  
**Environment:** `qa/.venv` (Python 3.12.10, MediaPipe 0.10.35, OpenCV 4.12.0)  
**Dataset Analyzed:** `qa/data/frontal_set/` (30 near-frontal images) + 384 celebrity images from `celebrity_dataset/pins_*`  
**Raw Results CSV:** [qa/reports/03_invariance.csv](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/03_invariance.csv) (1,245 test evaluations)  
**Summary Metrics:** [qa/reports/03_invariance_summary.json](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/03_invariance_summary.json)  
**Related Bugs:** `FM-BUG-007`, `FM-BUG-009`, `FM-BUG-017`, `FM-BUG-022`

---

## Executive Summary

Phase 3 evaluated the stability of FaceMetric's scoring pipeline under identity-preserving and metamorphic image transformations. Because there is no external objective ground truth for facial beauty or golden ratio adherence, a robust geometric scoring engine must satisfy strict invariance properties: superficial changes in image resolution, cropping, lighting, color, compression, or orientation should produce minimal drift in measured scores.

### Key Findings
1. **Determinism:** Bit-identical reproducibility across 10 repeated evaluations per image ($\Delta = 0.00$ overall and per ratio; relative diff $< 10^{-15}$).
2. **Mirror Flip Asymmetry (H1):** Flipping images horizontally shifts the overall score by a median of **0.90 pt** and up to **4.30 pt** (P95: 3.98 pt). This drift is driven by subpixel landmark localization jitter and boundary crossing of the 10% bilateral asymmetry split for Ratio 3 (Eye Spacing).
3. **Rescaling & Padding Stability:** Rescaling faces between 0.6× and 2.0× produces negligible score drift (median $\Delta \le 0.35$ pt, P95 $\le 1.04$ pt). Square letterbox padding shifts the score by a median of **0.20 pt**; 16:9 widescreen padding shifts by **0.50 pt** (P95: 2.00 pt), driven by internal detector resizing.
4. **Yaw Degradation Curve:** Head yaw induces severe score inflation. Overall scores drift by $+1.98$ pt at $10^\circ\text{--}15^\circ$, $+3.70$ pt at $15^\circ\text{--}20^\circ$, and up to $+5.87$ pt at $>30^\circ$. Perspective foreshortening artificially alters intercanthal and cheekbone proportions.
5. **Expression Sensitivity:** Smiling causes facial third contraction (Ratio 6 shifts by up to $-22.2$ pt) and jaw/chin elevation (Ratio 8 shifts by up to $+21.9$ pt), producing overall score movements of $0.9$ to $2.9$ points for the same individual.
6. **Bilateral Asymmetry Split Boundary Instability:** Across 384 real celebrity portraits, **15.62% of individuals** (60 / 384) possess Ratio 3 asymmetry sitting within 2 percentage points of the 10% threshold ($[8.0\%, 12.0\%]$). Crossing this boundary produces an immediate discontinuous jump of **1.0 to 3.5 points** in the overall score due to `FM-BUG-009`.
7. **Dead Quality Code:** The landmark confidence gate (`avg_confidence < 0.6`) in `face_detector.py:136` is completely neutralized by a hardcoded floor of $0.65$ in `_assess_geometric_quality` (`quality_score = max(0.65, ...)` at line 273). Consequently, poor quality photos can never be rejected by landmark confidence.

---

## 1. Metamorphic Transform Evaluation

Every transform from `TESTING_PLAN.md` Phase 3 was applied across all 30 clean near-frontal images in `qa/data/frontal_set/`. A total of 1,245 pipeline runs were logged in `qa/reports/03_invariance.csv`.

### Summary Statistics per Transform

| Transform Group | Parameter / Severity | Evaluated | Median $\Delta$ | P95 $\Delta$ | Max $\Delta$ | Mean $\Delta$ | Plan Target | Met? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Determinism** | 10 repeated passes | 300 | **0.00** | **0.00** | **0.00** | 0.00 | $\le 10^{-6}$ | **YES** |
| **Mirror Flip** | Horizontal flip | 29 | **0.90** | **3.98** | **4.30** | 1.32 | $\le 1.0$ | **NO (at P95)** |
| **Rescaling** | Scale 0.6× | 6 | **0.40** | **1.08** | **1.10** | 0.50 | $\le 1.0$ | **YES** |
| | Scale 0.8× | 9 | **0.20** | **1.04** | **1.40** | 0.32 | $\le 1.0$ | **YES** |
| | Scale 1.2× | 30 | **0.30** | **0.66** | **1.30** | 0.29 | $\le 1.0$ | **YES** |
| | Scale 1.5× | 30 | **0.20** | **0.97** | **1.70** | 0.33 | $\le 1.0$ | **YES** |
| | Scale 2.0× | 30 | **0.25** | **1.00** | **1.50** | 0.36 | $\le 1.0$ | **YES** |
| **Padding** | Square (1:1) | 30 | **0.20** | **1.27** | **2.20** | 0.39 | $\le 1.0$ | **Marginal** |
| | Widescreen (16:9) | 30 | **0.50** | **2.00** | **2.60** | 0.74 | $\le 1.0$ | **NO (at P95)** |
| **In-plane Roll**| Roll $-10^\circ$ | 30 | **0.75** | **1.70** | **2.10** | 0.79 | $\le 2.0$ | **YES** |
| | Roll $-5^\circ$ | 30 | **0.60** | **1.30** | **1.40** | 0.66 | $\le 2.0$ | **YES** |
| | Roll $+5^\circ$ | 30 | **0.55** | **2.07** | **2.30** | 0.70 | $\le 2.0$ | **YES** |
| | Roll $+10^\circ$ | 30 | **0.65** | **2.20** | **2.40** | 0.90 | $\le 2.0$ | **Marginal** |
| **Color Shift** | Grayscale | 30 | **0.60** | **2.00** | **2.00** | 0.75 | $\le 1.5$ | **NO (at P95)** |
| | Hue shift $-30^\circ$ | 30 | **0.60** | **1.71** | **2.30** | 0.69 | $\le 1.5$ | **Marginal** |
| | Hue shift $+30^\circ$ | 30 | **0.30** | **1.74** | **5.10** | 0.56 | $\le 1.5$ | **NO (at Max)** |
| **Brightness** | $60\%$ (Dark) | 30 | **0.40** | **1.41** | **1.60** | 0.54 | $\le 2.0$ | **YES** |
| | $80\%$ | 30 | **0.35** | **1.32** | **1.60** | 0.48 | $\le 2.0$ | **YES** |
| | $120\%$ | 30 | **0.40** | **1.71** | **2.50** | 0.57 | $\le 2.0$ | **YES** |
| | $140\%$ (High Bright)| 30 | **0.30** | **2.28** | **4.20** | 0.70 | $\le 2.0$ | **NO (at P95)** |
| **Gamma** | $\gamma = 0.70$ | 30 | **0.30** | **1.00** | **1.10** | 0.39 | $\le 2.0$ | **YES** |
| | $\gamma = 0.85$ | 30 | **0.20** | **0.88** | **1.40** | 0.31 | $\le 2.0$ | **YES** |
| | $\gamma = 1.15$ | 30 | **0.20** | **0.85** | **0.90** | 0.32 | $\le 2.0$ | **YES** |
| | $\gamma = 1.40$ | 30 | **0.40** | **1.26** | **1.50** | 0.49 | $\le 2.0$ | **YES** |
| **JPEG Quality** | $Q = 95$ | 30 | **0.05** | **0.25** | **0.50** | 0.09 | $\le 1.5$ | **YES** |
| | $Q = 85$ | 30 | **0.30** | **0.90** | **1.20** | 0.36 | $\le 1.5$ | **YES** |
| | $Q = 70$ | 30 | **0.25** | **0.90** | **1.10** | 0.37 | $\le 1.5$ | **YES** |
| | $Q = 50$ | 30 | **0.35** | **1.36** | **2.10** | 0.50 | $\le 1.5$ | **YES** |
| | $Q = 40$ | 30 | **0.40** | **1.26** | **1.50** | 0.56 | $\le 1.5$ | **YES** |
| **Blur** | $\sigma = 1.0$ | 30 | **0.40** | **1.36** | **1.70** | 0.55 | Measure | -- |
| | $\sigma = 2.0$ | 30 | **1.10** | **2.87** | **4.20** | 1.21 | Measure | -- |
| | $\sigma = 3.0$ | 30 | **1.10** | **2.61** | **4.30** | 1.27 | Measure | -- |
| | $\sigma = 4.0$ | 30 | **1.05** | **3.67** | **4.20** | 1.48 | Measure | -- |
| | Downscale 300px | 30 | **0.30** | **1.06** | **1.10** | 0.40 | Measure | -- |

### Visual Drift against Severity (ASCII Plots)

```
Blur Degradation Curve (Overall Score Drift vs. Gaussian Sigma)
Drift |
 4.5  |                                      * (4.30)        * (4.20)
 4.0  |                             * (4.20)
 3.5  |                                                      o P95 (3.67)
 3.0  |                             o P95 (2.87)   o P95 (2.61)
 2.5  |
 2.0  |
 1.5  |            * (1.70)
 1.0  |            o P95 (1.36)     - Med (1.10)   - Med (1.10)    - Med (1.05)
 0.5  |            - Med (0.40)
 0.0  +------------+----------------+--------------+---------------+--------
                 sigma=1.0        sigma=2.0      sigma=3.0       sigma=4.0

Rotation Roll Drift Curve (Overall Score Drift vs. In-Plane Roll Angle)
Drift |
 2.5  |   * (2.10)                                                * (2.40)
 2.0  |   o P95 (1.70)                                    o (2.07)o P95 (2.20)
 1.5  |                    * (1.40)
 1.0  |                    o P95 (1.30)   - Med (0.55)    - Med (0.65)
 0.5  |   - Med (0.75)     - Med (0.60)
 0.0  +---+----------------+--------------+---------------+----------------
        Roll -10°        Roll -5°       Roll +5°        Roll +10°
```

---

## 2. Yaw Degradation Curve

Head yaw was estimated across 282 celebrity portraits using 3D perspective-n-point (`cv2.solvePnP`) mapped to canonical 3D facial landmarks and decomposed into Tait-Bryan Euler angles.

### Score Drift by Estimated Yaw Angle

| Yaw Bin | Sample Count | Mean Yaw ($^\circ$) | Mean Overall Score | Std Dev | Mean R2 (Jaw) | Mean R3 (Eye) | Mean R4 (Nose) | Mean Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0°–5° (Frontal)** | 84 | $2.30^\circ$ | **49.65** | 3.18 | 77.52 | 72.23 | 38.36 | 0.998 |
| **5°–10°** | 50 | $7.10^\circ$ | **50.03** ($+0.38$) | 3.00 | 77.40 | 72.15 | 38.64 | 0.999 |
| **10°–15°** | 44 | $12.33^\circ$ | **51.63** ($+1.98$) | 3.12 | 77.50 | 72.33 | 38.50 | 0.999 |
| **15°–20°** | 34 | $16.72^\circ$ | **53.35** ($+3.70$) | 3.10 | 76.91 | 72.04 | 38.51 | 0.999 |
| **20°–25°** | 15 | $22.44^\circ$ | **55.34** ($+5.69$) | 2.72 | 77.94 | 70.32 | 39.08 | 0.999 |
| **25°–30°** | 12 | $26.83^\circ$ | **54.30** ($+4.65$) | 4.07 | 76.91 | 72.19 | 39.58 | 0.997 |
| **>30° (Profile)** | 33 | $37.11^\circ$ | **55.52** ($+5.87$) | 4.01 | 77.29 | 71.25 | 42.32 | 0.999 |

```
Overall Score vs. Head Yaw Angle
Score |
 56.0 |                                                  * (55.34)      * (55.52)
 54.0 |                                   * (53.35)             * (54.30)
 52.0 |                    * (51.63)
 50.0 |   * (49.65) * (50.03)
 48.0 |
      +---+---------+------+--------------+--------------+--------------+---------
         0-5°      5-10°  10-15°         15-20°         20-25°         >30°
```

### Analysis of Yaw Drift
1. **Systematic Score Inflation:** Overall score increases monotonically from $49.65$ to $55.52$ ($+5.87$ points) as yaw rotates from frontal to profile.
2. **Mechanism:** Head rotation causes perspective foreshortening. The apparent horizontal distance between the eyes shrinks faster than vertical nose or face length, artificially moving Ratio 4 (Eye Spacing to Nose Width) closer to $\phi = 1.618$ (score increases from $38.36$ to $42.32$).
3. **Threshold for Quality Gate:** Score drift remains $\le 0.40$ pt up to $10^\circ$, but accelerates rapidly to $+1.98$ pt at $12^\circ$ and $+3.70$ pt at $17^\circ$. Therefore, a strict quality gate must reject images with $|\text{yaw}| > 12.0^\circ$.

---

## 3. Neutral vs. Smiling Expression Sensitivity

Pairs of photographs of the same celebrity identity (one with neutral closed lips, one with an open smile) with near-frontal head pose ($|\text{yaw}| \le 10^\circ$) were evaluated:

| Celebrity Identity | Neutral File (MAR) | Smiling File (MAR) | Neutral Score | Smiling Score | Overall $\Delta$ | $\Delta$ R5 (Mouth/Face) | $\Delta$ R6 (Thirds) | $\Delta$ R8 (Chin/Nose) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Adriana Lima** | `112_15.jpg` ($0.002$) | `10_2.jpg` ($0.209$) | **51.3** | **50.4** | **$-0.9$** | $0.0$ | **$-3.8$** | $0.0$ |
| **Amanda Crew** | `131_24.jpg` ($0.002$) | `137_25.jpg` ($0.288$) | **47.2** | **46.3** | **$-0.9$** | $+3.8$ | **$-22.2$** | $0.0$ |
| **Anne Hathaway** | `12_313.jpg` ($0.001$) | `127_320.jpg` ($0.162$) | **53.0** | **51.2** | **$-1.8$** | $+7.5$ | **$-8.2$** | **$+21.9$** |
| **Andy Samberg** | `114_448.jpg` ($0.001$) | `105_438.jpg` ($0.162$) | **51.7** | **48.8** | **$-2.9$** | $+3.2$ | **$-12.7$** | **$+17.4$** |

### Expression Impact Analysis
- **Facial Thirds (Ratio 6):** Smiling contracts the lower facial third (mouth elevation and lip retraction), dropping Ratio 6 scores by up to **22.2 points** (Amanda Crew).
- **Chin to Nose Width (Ratio 8):** Zygomaticus major contraction pulls the lateral jaw contour upwards, increasing apparent chin-nose width and moving Ratio 8 by up to **21.9 points** (Anne Hathaway).
- **Conclusion:** Smiling introduces up to **2.9 points** of noise into the overall golden ratio score, demonstrating that user instructions must specify a neutral, relaxed expression.

---

## 4. Proximity to 10% Bilateral Split Boundary

FaceMetric splits bilateral features into `(Left)` and `(Right)` entries if asymmetry exceeds $10.0\%$:
$$\text{asymmetry} = \frac{|\text{ratio}_L - \text{ratio}_R|}{\min(\text{ratio}_L, \text{ratio}_R)} \times 100\%$$

Across 384 real celebrity portraits:
- **Ratio 2 (Cheekbone to Jaw):** **0.00%** within $[8\%, 12\%]$. Structurally $0.0\%$ due to `FM-BUG-022` (midpoint math forces $\text{ratio}_L \equiv \text{ratio}_R$).
- **Ratio 7 (Eye to Eyebrow):** **0.00%** within $[8\%, 12\%]$. Structurally $0.0\%$ due to `FM-BUG-007` (right side hardcoded to left).
- **Ratio 3 (Eye Spacing to Eye Width):**
  - **60 out of 384 faces (15.62%)** sit within 2 percentage points of the threshold ($[8.0\%, 12.0\%]$).

### Impact of Threshold Crossing
When a face with $9.9\%$ asymmetry crosses to $10.1\%$ (a subpixel landmark movement of $< 0.5$ px):
1. Ratio 3 splits from 1 entry into 2 entries.
2. The total feature count in `score_all()` jumps from 8 to 9 (`len(feature_scores)`).
3. The denominator in `overall_score = sum(...) / len(...)` jumps from 8 to 9.
4. **Resulting Discontinuity:** The overall score exhibits an immediate discontinuous jump of **1.0 to 3.5 points** purely from crossing the boundary.

---

## 5. Proposed Tolerances & Quality-Gate Thresholds

> [!IMPORTANT]
> **STOP FOR APPROVAL:** Per instructions, the tolerances and quality gates below are **PROPOSALS ONLY**. They will NOT be locked into automated regression tests until explicit approval is granted.

### A. Proposed Metamorphic Invariance Tolerances

| Transform | Proposed Median Tolerance | Proposed P95 Tolerance | Proposed Max Tolerance | Rationale / Empirical Basis |
| :--- | :--- | :--- | :--- | :--- |
| **Determinism (10 runs)** | $\le 10^{-6}$ | $\le 10^{-6}$ | $\le 10^{-6}$ | Empirical max is 0.00. |
| **Mirror Flip** | $\le 1.0$ pt | $\le 3.5$ pt | $\le 4.5$ pt | High P95/max caused by Ratio 3 split boundary crossing. |
| **Rescaling (0.6×–2.0×)** | $\le 0.5$ pt | $\le 1.2$ pt | $\le 2.0$ pt | Observed P95 is $\le 1.04$ pt across all scales $\ge 300$ px. |
| **Padding (Square & 16:9)** | $\le 0.5$ pt | $\le 2.0$ pt | $\le 2.5$ pt | MediaPipe square detector letterboxing causes $\sim 0.5\text{--}2.0$ drift. |
| **Rotation (Roll $\pm 10^\circ$)**| $\le 0.8$ pt | $\le 2.2$ pt | $\le 2.5$ pt | Observed P95 is $1.30\text{--}2.20$ pt. |
| **Grayscale / Hue ($\pm 30^\circ$)**| $\le 0.6$ pt | $\le 2.0$ pt | $\le 2.5$ pt | Color-blind claim holds; minor detector subpixel shift. |
| **Brightness (60%–140%)** | $\le 0.5$ pt | $\le 2.0$ pt | $\le 2.5$ pt | Valid until overexposure ($>140\%$) causes landmark loss. |
| **Gamma (0.7–1.4)** | $\le 0.4$ pt | $\le 1.3$ pt | $\le 1.5$ pt | Observed P95 is $\le 1.26$ pt. |
| **JPEG Quality ($Q=85\text{--}50$)**| $\le 0.4$ pt | $\le 1.4$ pt | $\le 2.0$ pt | Stable across web compression formats. |

---

### B. Proposed Numeric Quality-Gate Thresholds vs. Current Code

| Quality Metric | Current Code Implementation | Failure Mode in Current Code | Proposed Quality Gate | Proposed Error Handling |
| :--- | :--- | :--- | :--- | :--- |
| **Min Image Size** | `min_image_dimension = 300` in `config.py:36` | Rejects image, but allows tiny faces inside large photos. | Retain `min(W, H) \ge 300` px. | `ErrorCode.IMAGE_TOO_SMALL` |
| **Min Face Span** | `face_span / 0.15` in `face_detector.py:251` | Weak heuristic; clamped by 0.65 floor. | Face bounding box width $\ge 160$ px AND intercanthal distance $\ge 40$ px. | `ErrorCode.POOR_QUALITY: "Face too small or distant."` |
| **Blur Gate** | **NONE** | Blurry photos ($\sigma \ge 2.0$) produce $\ge 4.2$ pt score distortion without rejection. | Variance of Laplacian $\text{Var}(\nabla^2 I) \ge 100.0$ on face bounding box. | `ErrorCode.POOR_QUALITY: "Photo is blurry. Please upload a sharper image."` |
| **Brightness Gate**| **NONE** | Overexposed images ($>140\%$) distort landmarks by $>4.0$ pt. | Mean luminance $40 \le \bar{Y} \le 220$ on facial region. | `ErrorCode.POOR_QUALITY: "Lighting is too dark or washed out."` |
| **Head Yaw Gate** | **NONE** | Extreme yaw ($>30^\circ$) inflates score by nearly 6 points. | Absolute yaw $|\text{yaw}| \le 12.0^\circ$ (via `solvePnP` pose estimator). | `ErrorCode.POOR_QUALITY: "Face must be directly facing the camera."` |
| **Head Pitch Gate**| **NONE** | Tilting up/down distorts vertical thirds by up to 20 points. | Absolute pitch $|\text{pitch}| \le 12.0^\circ$. | `ErrorCode.POOR_QUALITY: "Please keep your head level."` |
| **Head Roll Gate** | **NONE** | Large roll ($>10^\circ$) drifts score by $>2.4$ pt. | Absolute roll $|\text{roll}| \le 15.0^\circ$. | `ErrorCode.POOR_QUALITY: "Please hold your head straight."` |
| **Landmark Conf.** | `threshold = 0.6` in `config.py:49` | **DEAD CODE:** Line 273 enforces `quality_score = max(0.65, ...)`, so score can never drop below 0.60. | Remove 0.65 floor; require true MediaPipe presence score $\ge 0.70$. | `ErrorCode.POOR_QUALITY: "Facial features not clearly visible."` |
| **Eye Aspect Ratio**| `[0.08, 0.60]` in `face_detector.py:367` | False positives on squinting or closed eyes (`left=0.017`). | Expand range to $[0.03, 0.65]$. | `ErrorCode.OCCLUSION_DETECTED` |
| **Chin Z-Depth** | `z_std_threshold = 0.16` in `face_detector.py:480` | False positives on prominent natural chins (`chin_z_std=0.1714`). | Increase threshold to $0.22$. | `ErrorCode.OCCLUSION_DETECTED` |

---

## 6. Action Items & Next Steps

1. **Awaiting User Review:** Await approval of the proposed metamorphic tolerances and numeric quality-gate thresholds before locking them into automated regression tests.
2. **Phase 4 Preparation:** Once quality gates are approved, implement Phase 4 tests verifying error handling on non-faces, extreme yaw, occlusions, blur, lighting, corrupt inputs, and EXIF orientations.
