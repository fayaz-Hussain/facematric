# FaceMetric — Comprehensive Validation & Test Plan Final Report

**Date:** 2026-10-15  
**Audience:** Antigravity AI Coding Agent & Human Engineering Lead  
**Governing Documents:** [FaceMatric-prd.md](file:///c:/Users/FayazHussain/Desktop/FaceMatric/FaceMatric-prd.md) (Draft v4), [TESTING_PLAN.md](file:///c:/Users/FayazHussain/Desktop/FaceMatric/TESTING_PLAN.md) (v2), [AGENTS.md](file:///c:/Users/FayazHussain/Desktop/FaceMatric/AGENTS.md)  
**Execution Environment:** Windows 10/11 x64, PowerShell 7, `qa/.venv` (Python 3.12.10) & `backend/venv` (Python 3.10.11), Node.js Playwright (`qa/e2e`)

---

## 1. Executive Summary & Verification vs Validation

This final report concludes the comprehensive Quality Assurance, verification, and empirical validation of the **FaceMetric** biometric facial analysis application. Over nine progressive testing phases (Phase 0 through Phase 8), the system was audited against its mathematical claims, API contracts, machine learning pipelines, and user-facing frontend.

Per `TESTING_PLAN.md`, this report strictly bifurcates two critical engineering questions:
- **Verification (Did we build the system right?):** The backend faithfully implements the mathematical golden ratio scoring formula ($100 \times [1 - |r - \phi| / \phi]$), MediaPipe 468-point mesh extraction, and ephemeral data handling with zero disk/log retention. API latency for core scoring is **282 ms** (P50), easily meeting the $<3.0\text{ s}$ SLA. However, contract defects break Base64 JSON ingestion (`FM-BUG-028`), create dual-schema error payloads (`FM-BUG-027`), and leave stale mocks in legacy unit tests.
- **Validation (Did we build the right system?):** The empirical and geometric foundations suffer from severe architectural and statistical flaws:
  1. *Geometric Validity:* Anatomical landmark selection introduces structural anomalies. Reusing lateral jaw landmarks (172/397) as chin width causes Ratio 8 to measure $\approx 3.3$ ($>2\times \phi$), depressing feature scores to $0.0$ (`FM-BUG-026`). Ratio 7 hardcodes right-eye measurements to left (`FM-BUG-007`), and Ratio 2 midpoint math structurally forces $0.0\%$ asymmetry (`FM-BUG-022`).
  2. *Look-Alike Matching:* Production code completely bypasses thresholding and artificially clamps look-alike confidence to $0.50 + \text{sim}$ in $[0.60, 0.96]$ (`FM-BUG-014`). Any face—including 100% of unknown non-celebrities—receives a forced look-alike. On held-out test splits, Top-1 accuracy is **54.9%** (far below the $90\%$ target), and the reference database reuses 100% of $64\times 64$ Bollywood crops that fail production upload gates (`FM-BUG-024`).
  3. *Statistical Calibration:* The percentile rank engine relies entirely on an assumed theoretical normal distribution $\mathcal{N}(70, 10)$ (`FM-BUG-015`). Measured frontal portraits average a score of $56.4 \pm 5.1$. Consequently, a user scoring $70$ is in the 99th empirical percentile, yet the app reports them as the 50th percentile.

---

## 2. Acceptance Criteria Evaluation Matrix

Per the mandate: *Do not claim any target is met unless a CSV or report artifact in `qa/reports/` demonstrates it.*

| Area | Criterion (from `TESTING_PLAN.md`) | Measured Result | Status | Primary Evidence & Reproducing Artifacts |
| :--- | :--- | :--- | :---: | :--- |
| **Scoring Math** | All unit tests pass; matches independent oracle to $10^{-6}$ relative | Production formula matches oracle to $< 10^{-12}$ relative. However, legacy unit suite has 1 stale failure (`FM-BUG-002`) and 1 integration mock crash (`FM-BUG-003`). | **PARTIALLY MET** | [qa/reports/01_scoring_math.md](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/01_scoring_math.md); [qa/tests/test_phase1_scoring_math.py](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/tests/test_phase1_scoring_math.py) |
| **Score Range** | Every valid response has all scores strictly within $[0, 100]$ | Clamping confirmed in code (`max(0.0, min(100.0, score))`). Across 600 transform runs in `03_invariance.csv`, all scores remained in $[0.0, 98.4]$. | **MET** | [qa/reports/03_invariance.csv](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/03_invariance.csv); [qa/reports/01_scoring_math.md](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/01_scoring_math.md) |
| **Measurements** | All 8 ratios have visually verified endpoints and written definitions | Written definitions verified. Endpoints plotted on 20 frontal photos. However, Ratio 8 reuses jaw points 172/397, Ratio 7 copies left eye, and Ratio 6 uses forehead landmark 10. | **NOT MET** *(Flawed Definitions)* | [qa/reports/02_measurements.md](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/02_measurements.md); [qa/reports/phase2_nose_width_comparison.csv](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/phase2_nose_width_comparison.csv) |
| **Determinism** | Same image $\rightarrow$ identical numerical output | 10 repeated evaluations of identical frontal image yielded identical landmarks and identical overall score down to $0.000000$ difference. | **MET** | [qa/tests/test_phase1_scoring_math.py](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/tests/test_phase1_scoring_math.py); [qa/reports/08_api_e2e.md](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/08_api_e2e.md) |
| **Invariance** | Each image transform within human-approved tolerances from Phase 3 | Horizontal flip: median $\Delta = 0.52$. Re-compression (Q85): median $\Delta = 0.00$. Rescaling (0.75x): median $\Delta = 0.00$. Pitch/yaw $>15^\circ$ causes $>12.0$ score drops. | **PENDING APPROVAL** | [qa/reports/03_invariance.csv](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/03_invariance.csv); [qa/reports/03_invariance.md](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/03_invariance.md) |
| **Same-Person Stability** | Overall-score $\text{SD} \le 3.0$ on near-frontal photos | Evaluated across 72 photos of 12 identities: Mean within-person $\mathbf{\text{SD} = 1.81 \pm 0.72}$. 11 of 12 identities ($91.7\%$) achieved $\text{SD} \le 3.0$. $\text{ICC}(1) = 0.854$. | **MET** | [qa/reports/05_same_person_stability.csv](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/05_same_person_stability.csv); [qa/reports/05_validity.md](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/05_validity.md) |
| **Percentile Engine** | Monotonic, bounded, documented reference; signal-to-noise reported | Formula is monotonic and bounded in $(0, 100)$. However, it uses uncalibrated $\mathcal{N}(70, 10)$, shifting median user percentile by 49 points. 1 score point moves reported rank by 3.99%. | **NOT MET** *(Uncalibrated)* | [qa/reports/05_validity.md](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/05_validity.md); [qa/reports/05_validity_summary.json](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/05_validity_summary.json) |
| **Celebrity Top-1** | $\ge 90\%$ on held-out test split of known identities | Evaluated on 2,131 held-out test images across 187 identities: Top-1 Accuracy is **54.9%** (1,170 / 2,131) under production pipeline; 57.4% under aligned pipeline. | **NOT MET** *(54.9% vs $\ge 90\%$)* | [qa/reports/celebrity_predictions.csv](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/celebrity_predictions.csv); [qa/reports/07_celebrity_eval.md](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/07_celebrity_eval.md) |
| **Returned-Match Precision** | $\ge 95\%$ on known + unknown mixture at chosen threshold | At PRD threshold 0.55: Precision = **76.2%** (95% CI: $[74.1\%, 78.2\%]$). Precision reaches $95.2\%$ only at threshold 0.72 (coverage drops to 16.8%). Production clamps bypass this. | **NOT MET** *(76.2% vs $\ge 95\%$)* | [qa/reports/07_celebrity_eval.md](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/07_celebrity_eval.md); [qa/eval/celebrity_eval.py](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/eval/celebrity_eval.py) |
| **False-Match Rate (FMR)** | Meets declared limit against unknown faces | Evaluated on 200 unknown LFW faces: At threshold 0.55, FMR is **34.5%**. In active production code (due to artificial confidence clamping), FMR is **100.0%**. | **NOT MET** *(34.5% vs $<5\%$)* | [qa/reports/07_celebrity_eval.md](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/07_celebrity_eval.md); [qa/tests/test_phase7_celebrity_matcher.py](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/tests/test_phase7_celebrity_matcher.py) |
| **Error Handling** | Every documented error returns a controlled, contract-correct response | All 7 error codes reachable (`NO_IMAGE`, `INVALID_REQUEST`, `INVALID_IMAGE`, `FILE_TOO_LARGE`, `NO_FACE_DETECTED`, `POOR_QUALITY`, `PROCESSING_TIMEOUT` 408). Base64 body broken (`FM-BUG-028`). | **PARTIALLY MET** | [qa/tests/test_phase8_api.py](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/tests/test_phase8_api.py); [qa/reports/08_api_e2e.md](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/08_api_e2e.md) |
| **Privacy & Hygiene** | No image retention; no embeddings or images in logs | Zero image files retained in temp/upload directories after requests. Zero base64 strings, landmark coordinate arrays, or 512-d embeddings logged. | **MET** | [qa/tests/test_phase8_api.py](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/tests/test_phase8_api.py); [qa/reports/08_api_e2e.md](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/08_api_e2e.md) |
| **Performance** | P0 path $< 3\text{ s}$ on test machine | Core scoring path (landmarks + scoring + symmetry): **P50 = 282.2 ms**, **P95 = 371.4 ms**. PRD target $< 3000\text{ ms}$ is exceeded by $87.6\%$. | **MET** | [qa/reports/08_api_benchmarks.json](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/08_api_benchmarks.json); [qa/reports/08_api_e2e.md](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/08_api_e2e.md) |

---

## 3. Verdict on Hypotheses H1–H14

Each hypothesis formulated in `TESTING_PLAN.md` was investigated against source code, git history, and empirical test harnesses:

| ID | Hypothesis Statement | Verdict | Evidence & Code Location |
| :--- | :--- | :---: | :--- |
| **H1** | PRD API example scores ($1.52 \rightarrow 82.3$, $1.70 \rightarrow 74.1$) cannot be produced by the formula ($93.94, 94.93$). | **CONFIRMED** | Mathematical proof: For $r=1.52$, $100 \times (1 - 0.098/1.618) = 93.94$. For $r=1.70$, $100 \times (1 - 0.082/1.618) = 94.93$. PRD numbers 82.3 and 74.1 are mathematically impossible under the formula. Code in [golden_ratio_scorer.py:209](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/golden_ratio_scorer.py#L209) implements the formula correctly and outputs 93.94. (`FM-BUG-001`) |
| **H2** | MediaPipe landmarks are normalized; computing distances on raw $x, y$ distorts ratios on non-square images. | **REFUTED** | [face_detector.py:112-113](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/face_detector.py#L112-L113) explicitly denormalizes landmarks to pixel coordinates (`lm.x * width`, `lm.y * height`). Distances are computed in 2D Euclidean pixel space, protecting against aspect-ratio stretching. |
| **H3** | Ratio 6 uses a "hairline", but MediaPipe has no hairline landmark; code uses an upper forehead proxy (landmark 10). | **CONFIRMED** | [golden_ratio_scorer.py:55, 451](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/golden_ratio_scorer.py#L55): `FACE_TOP = 10`, `hairline = points[self.FACE_TOP]`. Landmark 10 sits at the apex of the forehead mesh, significantly below the anatomical trichion for high foreheads or receding hairlines. (`FM-BUG-016`) |
| **H4** | Ratio 4 (nose $\div$ mouth) and Ratio 5 (mouth $\div$ face width) are structurally far from $\phi$ in real faces, capping overall scores well below 100. | **CONFIRMED** | Anatomical proportions dictate human nose/mouth ratio $\approx 0.65\text{--}0.75$ (score $\approx 40\text{--}46$) and mouth/face ratio $\approx 0.35\text{--}0.40$ (score $\approx 22\text{--}25$). In empirical testing, mean overall score across 30 clean frontal portraits is **$56.4 \pm 5.1$**, and maximum observed score was 63.8. Scores above 85 are biologically unreachable. |
| **H5** | Feature scores are not clamped, so ratios $> 2\phi$ give negative scores. | **REFUTED** | [golden_ratio_scorer.py:210](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/golden_ratio_scorer.py#L210) explicitly clamps: `score = max(0.0, min(100.0, score))`. Line 178 also clamps overall score. |
| **H6** | Reference celebrity embeddings built offline with different preprocessing than production request path. | **CONFIRMED** | Offline script [preprocess_celebrity_database.py:95-100](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/scripts/preprocess_celebrity_database.py#L95-L100) forces raw OpenCV resize to $(160, 160)$ without landmark face alignment. Production path ([main.py:459](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/main.py#L459)) applies EXIF orientation, JPEG re-compression, and MediaPipe bounding box cropping. This domain shift degrades cosine similarity. (`FM-BUG-013`) |
| **H7** | Per-celebrity embeddings averaged without L2-normalizing first; 0.55 threshold uncalibrated; "confidence" is just cosine similarity; embeddings are 512-d Facenet512. | **PARTIALLY REFUTED, PARTIALLY CONFIRMED** | *Refuted:* Embeddings are L2-normalized both before and after averaging ([preprocess_celebrity_database.py:128-136](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/scripts/preprocess_celebrity_database.py#L128-L136)).<br>*Confirmed:* Uses 512-d `Facenet512`. Threshold 0.55 is uncalibrated. Furthermore, [celebrity_matcher.py:507](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/celebrity_matcher.py#L507) replaced raw cosine similarity with an arbitrary ad-hoc linear shift `0.50 + sim` clamped to $[0.60, 0.96]$. (`FM-BUG-014`) |
| **H8** | Occlusion detection has no real mechanism: Face Mesh returns full mesh on masked/sunglassed faces. | **CONFIRMED (Fundamental) / PARTIALLY REFUTED (Heuristic)** | MediaPipe returns 468 landmarks regardless of occlusion. However, [face_detector.py:277-491](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/face_detector.py#L277-L491) implements heuristic checks on $z$-depth variance and 2D bounding ratios. These heuristics are uncalibrated and yield high false positives on clean portraits with minor styling. (`FM-BUG-017`) |
| **H9** | Celebrity result shape is inconsistent: PRD Step 5 says `null` below threshold, API example returns object with `name: null`. | **CONFIRMED** | PRD Step 5 specifies `null`. PRD Section 9 contract specifies `{ name: null, confidence: 0.42, message: ... }`. Code in [main.py:564](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/main.py#L564) returns `null` if matcher returns `None`, but `{ name: null }` if matcher returns below threshold. In frontend [results.js:166](file:///c:/Users/FayazHussain/Desktop/FaceMatric/frontend/src/js/results.js#L166), receiving `null` skips UI updates and leaves the static `#1 Tom Holland (92%)` placeholder visible. (`FM-BUG-010`) |
| **H10** | EXIF orientation is ignored and/or HEIC is unsupported despite claims. | **EXIF REFUTED; HEIC CONFIRMED** | EXIF orientation is handled via `ImageOps.exif_transpose(image)` in [image_processor.py:145](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/image_processor.py#L145). HEIC is claimed in `SUPPORTED_FORMATS`, but `pillow-heif` is not installed; uploading HEIC triggers `UnidentifiedImageError` and HTTP 400. (`FM-BUG-006`) |
| **H11** | The 0.55 comparison is made on a rounded value, so the UI can show "0.55" next to "No match". | **REFUTED (Backend) / CONFIRMED (Frontend)** | Backend compares unrounded float. However, frontend [results.js:170](file:///c:/Users/FayazHussain/Desktop/FaceMatric/frontend/src/js/results.js#L170) formats confidence via `Math.round(confidence * 100) + '%'`. A confidence of 0.546 rounds to 55%. |
| **H12** | Percentile engine uses assumed Normal(70, 10), not real data, so "score higher than X% of people" is unsupported. | **CONFIRMED** | [percentile_engine.py:26-36](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/percentile_engine.py#L26-L36) hardcodes `mean=70.0, std=10.0`. No empirical distribution is referenced. The user percentile is a pure mathematical artifact of the theoretical assumption. (`FM-BUG-015`) |
| **H13** | Status guides are stale: claim API gateway, celebrity matching and frontend are not implemented, but later files show they were completed. | **CONFIRMED** | Git history confirms guides were created at commit `48799ca` (Oct 11). Later commits (`14637f8`, `931551b`, `d54c749`, `851c3cb`) implemented `/api/analyze`, database loading, timeout middleware, and frontend integration without updating the status files. |
| **H14** | Existing unit tests are component-level with synthetic/mocked inputs, allowing mathematical conflicts (H1) or bad percentile assumptions (H12) to pass undetected. | **CONFIRMED** | Unit tests in `backend/tests/unit/` use synthetic landmark arrays (`np.zeros((468, 3))`) or mock data. In mutation testing (`00c_test_audit.md`), 7 of 18 semantic mutations survived undetected (38.9% mutation survival rate). (`FM-BUG-002`, `FM-BUG-003`) |

---

## 4. Complete Bug Log Grouped by Proposed Fix Batch

All 29 logged bugs from [qa/reports/BUGS.md](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/BUGS.md) are consolidated below into 5 cohesive engineering fix batches.

### Batch 1: Scoring Math & Facial Geometry Defects

#### `FM-BUG-001`: PRD API Example Scores Contradict PRD Mathematical Formula
- **Severity:** Spec-Only (Specification contradiction)
- **Component:** Specification / Golden Ratio Scoring
- **Expected:** PRD Section 9 payload examples should match the Section 8 formula ($r=1.52 \rightarrow 93.94$, $r=1.70 \rightarrow 94.93$).
- **Actual:** PRD Section 9 gives 82.3 and 74.1. Code in [golden_ratio_scorer.py:209](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/golden_ratio_scorer.py#L209) implements formula correctly.
- **Evidence / Repro:** `qa/tests/test_phase1_scoring_math.py::TestScoringMathReferenceTable::test_reference_table_against_oracle`

#### `FM-BUG-007`: Ratio 7 (Eye-to-Brow) Bilateral Measurement Hardcoded to Left Eye
- **Severity:** S3 (Moderate)
- **Component:** Golden Ratio Scoring
- **Expected:** Right eyebrow-to-eye distance measured using right eye landmarks (landmarks 295, 386).
- **Actual:** [golden_ratio_scorer.py:574](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/golden_ratio_scorer.py#L574) explicitly assigns `ratio_right = ratio_left`. Right eye asymmetry is structurally impossible (permanently 0.0%).
- **Evidence / Repro:** `qa/tests/test_phase1_scoring_math.py::test_ratio7_right_side_hardcoded_to_left_xfail`

#### `FM-BUG-008`: Shadowed Duplicate Method Definitions in `GoldenRatioScorer`
- **Severity:** S3 (Moderate)
- **Component:** Code Quality / Scorer
- **Expected:** Single clean method implementation per ratio.
- **Actual:** [golden_ratio_scorer.py:276-385](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/golden_ratio_scorer.py#L276-L385) defines bilateral methods that are silently re-defined and overwritten at lines 486–577, leaving unreachable dead code.
- **Evidence / Repro:** Line inspection in [golden_ratio_scorer.py:276-385, 486-577](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/golden_ratio_scorer.py#L276).

#### `FM-BUG-009`: Asymmetric Bilateral Splitting Biases Overall Golden Ratio Score
- **Severity:** S2 (Major)
- **Component:** Scoring Math
- **Expected:** Overall score is the unweighted mean of the 8 facial features.
- **Actual:** When asymmetry exceeds 10%, [golden_ratio_scorer.py:175](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/golden_ratio_scorer.py#L175) appends both Left and Right scores to `feature_scores` and divides by `len(feature_scores)` (9 or 10), doubling the weighting of the asymmetric feature.
- **Evidence / Repro:** `qa/tests/test_phase1_scoring_math.py::test_asymmetric_split_biases_overall_score_xfail`

#### `FM-BUG-015`: Percentile Distribution Assumes Theoretical $\mathcal{N}(70, 10)$ without Empirical Basis
- **Severity:** S2 (Major)
- **Component:** Percentile Scoring / Statistics
- **Expected:** Percentile calculated against empirical distribution of facial golden ratio scores.
- **Actual:** [percentile_engine.py:26-36](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/percentile_engine.py#L26-L36) hardcodes `norm.cdf(overall_score, 70.0, 10.0)`. Empirical population mean is $56.4 \pm 5.1$, distorting reported percentiles by ~49 points.
- **Evidence / Repro:** `qa/reports/05_validity.md`; `qa/eval/phase5_validity_eval.py`

#### `FM-BUG-016`: Anatomical Hairline Approximated by Forehead Landmark 10
- **Severity:** S3 (Moderate)
- **Component:** Facial Landmarks / Ratio 6
- **Expected:** True anatomical trichion used for upper facial third.
- **Actual:** [golden_ratio_scorer.py:442-450](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/golden_ratio_scorer.py#L442-L450) uses landmark 10 (top center of mesh contour), systematically underestimating forehead height.
- **Evidence / Repro:** Visual overlay analysis in `qa/reports/02_measurements.md`

#### `FM-BUG-022`: Ratio 2 Bilateral Measurement Midpoint Math Enforces 0.0% Asymmetry
- **Severity:** S2 (Major)
- **Component:** Golden Ratio Scoring / Bilateral Asymmetry
- **Expected:** Left and right cheekbone-to-jaw proportions measured relative to true facial midline (subnasale/philtrum).
- **Actual:** [golden_ratio_scorer.py:500-516](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/golden_ratio_scorer.py#L500-L516) defines `face_center = (left + right) / 2`. Euclidean distance from each side to the segment midpoint is mathematically identical ($\frac{1}{2}\|A - B\|$). Asymmetry is structurally guaranteed to be $0.0\%$.
- **Evidence / Repro:** `qa/eval/test_bilateral_quantification.py`; `qa/reports/01_scoring_math.md`

#### `FM-BUG-026`: Ratio 8 Reuses Lateral Jaw Landmarks (172/397), Making Ratio Biologically Double $\phi$
- **Severity:** S2 (Major)
- **Component:** Golden Ratio Scoring / Landmark Semantics
- **Expected:** Ratio 8 (Chin to Nose Width) measured using true chin landmarks (menton / chin crest ~40–60 px).
- **Actual:** [golden_ratio_scorer.py:470](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/golden_ratio_scorer.py#L470) reuses landmarks 172 and 397 (the lateral gonial jaw points from Ratio 2). Measured ratio is $3.298 \pm 0.267$ ($>2\times \phi$), depressing feature scores to $0.0$ on $100\%$ of human faces.
- **Evidence / Repro:** `qa/reports/phase2_nose_width_comparison.csv`; `qa/reports/02_measurements.md`

#### `FM-BUG-029`: Face Shape Classifier Unhandled Gap Intervals Cause Massive Fallthrough Bias Toward 'Oval'
- **Severity:** S3 (Moderate)
- **Component:** Face Shape Classifier / SymmetryClassifier
- **Expected:** Contiguous partitioning of face morphometry covering intermediate length/width ratios and jawline angles.
- **Actual:** [symmetry_classifier.py:140-164](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/symmetry_classifier.py#L140-L164) leaves large gaps ($1.10 < \text{ratio} < 1.40$ and angles $130^\circ\text{--}140^\circ$), silently falling through to `OVAL` on $>70\%$ of all images.
- **Evidence / Repro:** `qa/tests/test_phase5_validity.py::TestFaceShapeClassifierBoundaries::test_unclassified_middle_ratio_defaults_to_oval`

---

### Batch 2: Celebrity Look-Alike Pipeline Defects

#### `FM-BUG-004`: Celebrity Look-Alike Matching Completely Disabled at Runtime in `backend/venv`
- **Severity:** S1 (Critical)
- **Component:** Environment / Celebrity Matcher
- **Expected:** `deepface` installed and functional.
- **Actual:** `deepface` is missing in `backend/venv`. [celebrity_matcher.py:230](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/celebrity_matcher.py#L230) catches `ModuleNotFoundError`, logs an error, and returns `None`, silently crippling look-alike matching.
- **Evidence / Repro:** `backend/venv/Scripts/python.exe -c "import deepface"`

#### `FM-BUG-011`: Celebrity Matching Threshold Conflict in Configuration
- **Severity:** S3 (Moderate)
- **Component:** Configuration
- **Expected:** Uniform threshold configuration matching PRD Step 3c ($0.55$).
- **Actual:** [config.py:44](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/config.py#L44) sets `0.35`, [celebrity_matcher.py:288](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/celebrity_matcher.py#L288) defaults to `0.55`, and active code clamps top matches to $\ge 0.55$.
- **Evidence / Repro:** Code comparison between `config.py` and `celebrity_matcher.py`.

#### `FM-BUG-013`: Preprocessing Disparity between Offline Database and Production Pipeline
- **Severity:** S2 (Major)
- **Component:** Preprocessing / Celebrity Embeddings Parity
- **Expected:** Database centroids and query embeddings generated using identical detection, cropping, and normalization.
- **Actual:** Centroids in `celebrity_embeddings.pkl` were built with raw unaligned $160\times 160$ OpenCV resizing. Production uses MediaPipe crop + JPEG compression, reducing cosine similarity.
- **Evidence / Repro:** [scripts/generate_celebrity_embeddings.py:73-82](file:///c:/Users/FayazHussain/Desktop/FaceMatric/scripts/generate_celebrity_embeddings.py#L73-L82) vs [celebrity_matcher.py:347-380](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/celebrity_matcher.py#L347-L380).

#### `FM-BUG-014`: Artificial Confidence Clamping Removes "No Match" Pathway
- **Severity:** S2 (Major)
- **Component:** Celebrity Matching / Calibration
- **Expected:** Low-similarity queries ($<0.55$) return "No match found".
- **Actual:** [celebrity_matcher.py:507](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/celebrity_matcher.py#L507) forces $c_1 = \max(0.60, \min(0.96, 0.50 + \text{sim}))$, guaranteeing that every face receives a confidence between 60% and 96%.
- **Evidence / Repro:** `qa/tests/test_phase7_celebrity_matcher.py::test_confidence_clamping_forces_match_on_unknown`

#### `FM-BUG-021`: Bollywood Celebrity Dataset Resolution Fails Production Minimum Dimensions
- **Severity:** S3 (Moderate)
- **Component:** Dataset Compatibility / Ingestion
- **Expected:** Dataset images pass production validation gates.
- **Actual:** All images in `celebrity_dataset/[Name]/*.jpg` are cropped to $64\times 64$ px. Production gate requires $\ge 300\times 300$ px, throwing `IMAGE_TOO_SMALL`.
- **Evidence / Repro:** `celebrity_dataset/Aamir_Khan/0.jpg`; [backend/app/components/image_processor.py:65](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/image_processor.py#L65).

#### `FM-BUG-023`: Confidence Thresholding Method `get_best_match` is Dead Code in Production
- **Severity:** S2 (Major)
- **Component:** Celebrity Matching / API Orchestration
- **Expected:** Threshold filtering applied to query matches.
- **Actual:** `CelebrityMatcher.get_best_match()` implements thresholding, but is never called by `main.py`. The active route calls `find_match()` $\rightarrow$ `_find_best_match()`, which completely bypasses threshold checks.
- **Evidence / Repro:** [backend/app/main.py:528](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/main.py#L528); [backend/app/components/celebrity_matcher.py:276](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/celebrity_matcher.py#L276).

#### `FM-BUG-024`: Production Resolution Check Rejects 89.1% of Reference Dataset Images
- **Severity:** S2 (Major)
- **Component:** Input Processing / Dataset Parity
- **Expected:** Reference database images are valid according to production input rules.
- **Actual:** 1,827 of 2,050 reference images (89.1%) in `celebrity_dataset/` are smaller than $300\times 300$ px and would be rejected with HTTP 400 if uploaded via `/api/analyze`.
- **Evidence / Repro:** `qa/reports/06_dataset_audit.md`; `qa/reports/07_celebrity_eval.md`.

#### `FM-BUG-025`: Displayed Confidence Formula Saturates at 96% for All Matches Above 0.46
- **Severity:** S2 (Major)
- **Component:** Celebrity Matching / Calibration
- **Expected:** Linear or calibrated probabilistic mapping reflecting cosine similarity.
- **Actual:** For any match with similarity $\ge 0.46$, $0.50 + \text{sim} \ge 0.96$, saturating at the 0.96 clamp. In empirical testing, 99.8% of matches display an identical, flat "96%".
- **Evidence / Repro:** `qa/reports/celebrity_predictions.csv`; `qa/eval/celebrity_eval.py`.

---

### Batch 3: UI, Legal & Documentation Defects

#### `FM-BUG-010`: Results UI Shows Fake "Tom Holland (92%)" when Celebrity Match is Null
- **Severity:** S1 (Critical)
- **Component:** Frontend Results Page
- **Expected:** When `celebrity_match` is `null`, look-alike card is hidden or displays "No match found".
- **Actual:** [results.js:166](file:///c:/Users/FayazHussain/Desktop/FaceMatric/frontend/src/js/results.js#L166) only updates the DOM if `celebrity && celebrity.name`. When `null`, the card is untouched, leaving static HTML `#1 Tom Holland (92%)` visible to the user.
- **Evidence / Repro:** `qa/e2e/tests/frontend.spec.js` (Test 5); [frontend/results.html:475](file:///c:/Users/FayazHussain/Desktop/FaceMatric/frontend/results.html#L475).

#### `FM-BUG-012`: Disclaimer Divergence from PRD Section 15
- **Severity:** S4 (Minor)
- **Component:** Legal / Compliance / UI
- **Expected:** PRD Section 15 verbatim text displayed across all screens.
- **Actual:** `analyzing.html` contains no disclaimer. `index.html` and `results.html` display divergent text (*"This analysis is based on facial geometry and proportions, not skin color, ethnicity, or beauty standards."*).
- **Evidence / Repro:** `qa/e2e/tests/frontend.spec.js` (Test 4); [frontend/results.html:496](file:///c:/Users/FayazHussain/Desktop/FaceMatric/frontend/results.html#L496).

#### `FM-BUG-019`: Quality Check Fix Summary Contradicts Active Configuration
- **Severity:** S3 (Moderate)
- **Component:** Documentation / Configuration
- **Expected:** Documentation matches codebase constants.
- **Actual:** `QUALITY_CHECK_FIX_SUMMARY.md` claims quality threshold was lowered to 0.3. [config.py:38](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/config.py#L38) sets `0.6`.
- **Evidence / Repro:** `QUALITY_CHECK_FIX_SUMMARY.md` vs `backend/app/config.py:38`.

#### `FM-BUG-020`: AI Status Documentation and Commit Messages Contain False Claims
- **Severity:** S4 (Minor)
- **Component:** Documentation Integrity
- **Expected:** Accurate reporting of test suite pass rates and feature status.
- **Actual:** `TESTING_STATUS.md` and commit `d54c749` claim "All 764/764 tests passing". Actual unit suite had 592 tests (with 1 failure) and integration suite crashed during collection.
- **Evidence / Repro:** `qa/reports/00_recon.md`; `git log -n 1 d54c749`.

---

### Batch 4: Environment, Deployment & Ingestion Defects

#### `FM-BUG-005`: Integration Test Suite Inexecutable in `backend/venv` due to Missing `httpx`
- **Severity:** S1 (Critical)
- **Component:** Environment / Test Infrastructure
- **Expected:** `httpx` installed to support Starlette `TestClient`.
- **Actual:** `httpx` is missing in `backend/venv`, causing all 8 integration test suites (172 tests) to fail during collection with `RuntimeError`.
- **Evidence / Repro:** `backend/venv/Scripts/pytest.exe tests/integration/`

#### `FM-BUG-006`: HEIC Image Decoding Fails Despite Claimed Support
- **Severity:** S3 (Moderate)
- **Component:** Image Processing
- **Expected:** iPhone HEIC uploads decoded and processed successfully.
- **Actual:** `pillow-heif` is not installed. Uploading HEIC produces `UnidentifiedImageError` and HTTP 400 `INVALID_IMAGE`.
- **Evidence / Repro:** `qa/tests/test_phase4_quality_gates.py::test_heic_upload_xfail`

#### `FM-BUG-017`: Occlusion and Lighting Heuristics Uncalibrated
- **Severity:** S3 (Moderate)
- **Component:** Quality Validation
- **Expected:** Occlusion filters detect actual masks/sunglasses without rejecting clean portraits.
- **Actual:** Heuristic $z$-depth variance triggers false-positive rejections on clean portraits with mild styling or standard glasses.
- **Evidence / Repro:** `qa/reports/00b_env_followup.md`; [face_quality_validator.py:175](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/face_quality_validator.py#L175).

#### `FM-BUG-018`: Severe Dependency Drift between `backend/venv` and `backend/requirements.txt`
- **Severity:** S2 (Major)
- **Component:** Environment / Deployment
- **Expected:** Virtual environment mirrors pinned `requirements.txt`.
- **Actual:** 28 packages differ in version. `deepface` and `httpx` are absent from `backend/venv`.
- **Evidence / Repro:** `qa/reports/00a_env.md`; `qa/reports/00b_env_followup.md`.

#### `FM-BUG-027`: Error Response Schema Inconsistency (Nested "detail" vs Top-Level ErrorResponse)
- **Severity:** S2 (Major)
- **Component:** API Gateway / Contract Compliance
- **Expected:** Uniform top-level error format `{"status": "error", "error_code": "...", ...}` across all errors.
- **Actual:** Errors raised inside `_process_analysis` are re-raised via `HTTPException(detail=e.to_dict())`, nesting the payload under `{"detail": {"status": ...}}`. Gateway errors return top-level format.
- **Evidence / Repro:** `qa/tests/test_phase4_quality_gates.py::test_contract_top_level_error_schema_on_corrupt_file`

#### `FM-BUG-028`: Base64 Image Input Broken for POST Request Bodies
- **Severity:** S2 (Major)
- **Component:** API Gateway / Request Parsing
- **Expected:** Submitting Base64 JSON or form body `{"image_base64": "..."}` is accepted by `/api/analyze`.
- **Actual:** [main.py:128](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/main.py#L128) declares `image_base64: Optional[str] = None` without `Body()` or `Form()`, causing FastAPI to expect a URL query parameter. Submitting via body returns HTTP 400 `NO_IMAGE`.
- **Evidence / Repro:** `qa/tests/test_phase8_api.py::test_base64_in_post_body_contract`

---

### Batch 5: Test Suite Integrity & Stale Mocks

#### `FM-BUG-002`: Regression Failure in Unit Test Suite due to Ad-hoc Calibration Scaling
- **Severity:** S2 (Major)
- **Component:** Celebrity Matching / Test Suite
- **Expected:** Unit test `test_rank_celebrities_finds_best_match` passes.
- **Actual:** Fails with `assert 0.96 == 1.0`. Commit `851c3cb` introduced ad-hoc clamping to 0.96 without updating the test assertion.
- **Evidence / Repro:** `pytest backend/tests/unit/test_rank_celebrities.py -k test_rank_celebrities_finds_best_match`

#### `FM-BUG-003`: Integration Test Failure in `/api/analyze` Mock Pipeline
- **Severity:** S2 (Major)
- **Component:** API Gateway Orchestration / Test Suite
- **Expected:** Integration test `test_successful_response_structure` passes.
- **Actual:** Fails with HTTP 500 (`TypeError: 'Mock' object is not iterable`). Commit `851c3cb` iterated over `top_matches`, but test mocked it with a non-iterable `MagicMock()`.
- **Evidence / Repro:** `pytest backend/tests/integration/test_api_endpoint_integration.py -k test_successful_response_structure`

---

## 5. Decisions Still Owed by the Human Lead

Per `AGENTS.md` ("Never loosen a tolerance... Stop and ask me... Fixes happen only after I approve them"), the following architectural and mathematical decisions require explicit approval before fixes can be applied:

1. **Spec Conflict on PRD Section 9 Examples (`FM-BUG-001`):**  
   *Decision:* Formally approve deprecating the contradictory 82.3 and 74.1 examples in PRD Section 9 and standardizing on the Section 8 mathematical formula ($93.94$ and $94.93$).
2. **Bilateral Ratio Asymmetry Weighting (`FM-BUG-009`):**  
   *Decision:* When a bilateral ratio deviates by $>10\%$, should the system:
   - *(Option A - Recommended):* Average the left and right measurements first so the overall score remains the unweighted arithmetic mean of 8 features.
   - *(Option B):* Retain the current behavior of dividing by 9 or 10, explicitly acknowledging that asymmetric features carry double weight.
3. **Ratio 8 Landmark Redefinition (`FM-BUG-026`):**  
   *Decision:* Ratio 8 currently reuses lateral jaw points 172/397 as chin width, producing ratios $>3.2$ and scores of $0.0$.
   - *(Option A - Recommended):* Redefine chin width using true anatomical chin landmarks (landmark 152 menton, or 200/420 mandibular sulcus).
   - *(Option B):* Keep 172/397, but rename the feature to "Jaw Width $\div$ Nose Width" and adjust the ideal target ratio accordingly.
4. **Ratio 4 and 8 Nose Width Landmarks:**  
   *Decision:* Code currently uses 219/439 (inner alar base). Phase 2 demonstrated that outer alar curvature points 129/358 provide measurements conforming much better to classical facial aesthetics. Approve switching to 129/358 or maintaining 219/439.
5. **Ratio 2 Bilateral Math Fix (`FM-BUG-022`):**  
   *Decision:* Approve replacing the midpoint formula with true distance to facial midline (subnasale landmark 2 / philtrum landmark 168) to unblock asymmetry detection on Ratio 2.
6. **Celebrity Matcher Calibration Policy (`FM-BUG-014`, `FM-BUG-023`, `FM-BUG-025`):**  
   *Decision:* Remove the artificial linear shift $0.50 + \text{sim}$ and clamp $[0.60, 0.96]$. Select the production operating threshold:
   - Threshold 0.55: Coverage 85.3%, Precision 76.2%, FMR 34.5%.
   - Threshold 0.65: Coverage 42.1%, Precision 89.1%, FMR 6.2%.
   - Threshold 0.72: Coverage 16.8%, Precision 95.2%, FMR 1.5%.
7. **Approval of Quality Gate Thresholds (Phase 3 Proposal):**  
   *Decision:* Approve locking in the proposed numerical quality gates:
   - Minimum image dimension: $250\times 250$ px (or allow $64\times 64$ for Bollywood dataset).
   - Pose yaw gate: $[-10.0^\circ, +10.0^\circ]$.
   - Pose roll gate: $[-8.0^\circ, +8.0^\circ]$.
   - Remove the hardcoded 0.65 floor on landmark confidence in [face_detector.py:273](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/face_detector.py#L273).
8. **Percentile Distribution Calibration (`FM-BUG-015`):**  
   *Decision:* Approve replacing theoretical $\mathcal{N}(70, 10)$ with empirical percentile lookup tables derived from the benchmark facial dataset.
9. **Face Shape Ground Truth Labels:**  
   *Decision:* Human lead needs to provide ratings in [qa/data/faceshape_labels.csv](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/data/faceshape_labels.csv) (template created in Phase 5 with 60 images across 3 reviewer columns) to unblock the human agreement analysis for face shape classification.
10. **Legal Disclaimer Alignment (`FM-BUG-012`):**  
    *Decision:* Approve inserting the PRD Section 15 educational disclaimer verbatim into `analyzing.html`, `index.html`, and `results.html`.

---

## 6. Limitations of the Evaluation

To prevent misinterpretation of this audit, the following scientific and operational limitations must be recognized:

1. **Golden Ratio is Not Attractiveness:** Proximity to $\phi \approx 1.618$ is an anthropometric geometric curiosity popularized in art and cosmetic surgery. Scientific facial perception literature does not support $\phi$ as a universal or objective determinant of biological beauty, personal attractiveness, or human worth. The PRD Section 15 disclaimer is essential.
2. **Demographic and Celebrity Dataset Bias:** The evaluation relied on PinsFaceRecognition and the Bollywood Localized Face Dataset. Both datasets represent professional entertainment figures photographed with studio lighting, professional makeup, and flattering poses. These models cannot be assumed to generalize across all ages, ethnicities, and non-studio smartphone captures.
3. **MediaPipe Landmark Approximations:** MediaPipe Face Mesh is a monocular 2D/3D surface regression model. It estimates landmarks on a generic topology:
   - Landmark 10 represents the upper forehead contour, not the biological hairline (trichion); hairline-dependent ratios are sensitive to hairstyles and hair loss.
   - Landmarks 172/397 represent the lateral gonial jaw, not the chin (gnathion).
   - Occlusion handling does not semantically segment sunglasses or masks; it relies on proxy bounding heuristics.
4. **Same-Person Multi-Photo Sample Size:** Within-person repeatability was evaluated on 72 photos across 12 celebrity identities. While confirming low score standard deviation ($\text{SD} = 1.81$), clinical validation of metric repeatability would require longitudinal multi-camera captures with consenting volunteers across controlled illuminations.

---

## 7. Complete Reproduction Steps

All nine phases can be fully reproduced using the following commands in PowerShell.

### 7.1 Virtual Environments Setup
```powershell
# Python environments:
# 1. qa/.venv: Contains full pipeline, deepface, tensorflow, imagehash, pillow-heif
# 2. backend/venv: Production environment baseline

# Activate QA environment
.\qa\.venv\Scripts\Activate.ps1
```

### 7.2 Phase Execution Commands

```powershell
# Phase 0 & 0.5: Test Suite Audit & Baseline Checks
cd backend
..\qa\.venv\Scripts\pytest.exe tests/unit/ -v
..\qa\.venv\Scripts\pytest.exe --cov=app --cov-branch
cd ..

# Phase 1: Scoring Math & Oracle Tests
.\qa\.venv\Scripts\pytest.exe qa/tests/test_phase1_scoring_math.py -v

# Phase 2: Measurement Verification & Landmark Overlays
.\qa\.venv\Scripts\python.exe qa/eval/phase2_measurements.py

# Phase 3: Invariance & Metamorphic Testing
.\qa\.venv\Scripts\python.exe qa/eval/phase3_invariance.py

# Phase 4: Quality Gates & Error Handling
.\qa\.venv\Scripts\pytest.exe qa/tests/test_phase4_quality_gates.py -v

# Phase 5: Repeatability, Validity & Percentile
.\qa\.venv\Scripts\python.exe qa/eval/phase5_validity_eval.py
.\qa\.venv\Scripts\pytest.exe qa/tests/test_phase5_validity.py -v

# Phase 6: Celebrity Dataset Audit
.\qa\.venv\Scripts\python.exe qa/eval/phase6_dataset_audit.py

# Phase 7: Celebrity Matcher Evaluation
.\qa\.venv\Scripts\python.exe qa/eval/celebrity_eval.py
.\qa\.venv\Scripts\pytest.exe qa/tests/test_phase7_celebrity_matcher.py -v

# Phase 8: API Integration, Privacy & Latency
.\qa\.venv\Scripts\pytest.exe qa/tests/test_phase8_api.py -v
.\qa\.venv\Scripts\python.exe qa/eval/phase8_api_eval.py

# Phase 8: Playwright Frontend E2E
cd qa/e2e
npx playwright test
cd ../..
```

---

## 8. Final Verdict & Sign-Off

The FaceMetric verification and validation plan is **COMPLETE**. All empirical data, test scripts, and reports are committed and referenced. Production code and legacy tests remain untouched in strict compliance with `AGENTS.md`.

Remediation work may commence across Batches 1 through 5 once the human lead approves the decisions outlined in Section 5.
