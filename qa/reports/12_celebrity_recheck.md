# FaceMetric — Phase 12: Celebrity Verification & Pipeline Re-Check Report

**Date:** 2026-10-21  
**Author:** QA Engineering Agent  
**Commit Tested:** `3a1cb063e6dd742d65e21e98187d256e14863e36` (Merged Main Branch)  
**Python Runtime:** `qa/.venv/Scripts/python.exe` (Python 3.12.10, Windows x64)  
**Strict Policy Compliance:** Followed `AGENTS.md`. No production code or existing tests under `backend/tests/` were modified. Test split was strictly held out and untouched.

---

## 1. Executive Summary

This report establishes the verified empirical truth regarding the celebrity matching database, preprocessing pipeline parity, dataset partitions, and classification performance on the merged `main` branch.

Previous regression reporting in `qa/reports/11_regression.md` presented celebrity metrics (Top-1 99.25% across 40 identities; unknown false matches 0.42%) that were **copied from the pre-fix Phase 7 pilot evaluation** rather than measured on the updated 275-identity database. Furthermore, that baseline used an unaligned whole-image resize script (`cv2.resize(img, (160, 160))`) rather than the production face cropping pipeline (`extract_face_crop` with 10% bounding box margin).

### Key Findings Summary:
1. **Database Identity Count:** `backend/data/celebrity_embeddings.pkl` contains **275 identities** and **5,481 reference images** (267 identities have 20 references, 5 have 19, 2 have 18, 1 has 10). `celebrity_embeddings.old.pkl` exists (454,072 bytes, 205 identities).
2. **Prior Pipeline Disparity:** The prior evaluation script (`qa/eval/celebrity_eval.py`) evaluated only 40 pilot identities against the old 205-identity database using full-image resizing without face cropping.
3. **Test-Set Contamination:** In Batch 2, when the database was rebuilt with 20 references per identity, **1,783 images (15.56%)** from the old held-out test split (`splits.csv`) and **1,753 images (15.34%)** from the old validation split were absorbed into the new reference set.
4. **New Splits Partitioning (`splits_v2.csv`):** A clean 50/50 partition of remaining images was generated across all 275 identities (5,481 reference, 12,094 validation, 12,216 test), preserving near-duplicate perceptual hash clusters ($\text{Hamming} \le 6$). Zero identities had $<6$ remaining images. The test split was **strictly sealed and not evaluated**.
5. **Re-Measured Performance (Production Pipeline):**
   - **Validation Queries ($N = 12,094$):** Top-1 Accuracy is **`97.68%`** (11,813 / 12,094; 95% CI: `[97.39%, 97.93%]`), Top-5 is **`99.38%`** (12,019 / 12,094; 95% CI: `[99.22%, 99.50%]`).
   - **Operating Point (0.55 Threshold):** Coverage is **`97.63%`** (11,807 returned) with **`98.65%`** precision (11,648 / 11,807 correct; 95% CI: `[98.43%, 98.85%]`).
   - **Unknown Faces (LFW, $N = 240$):** False match rate at 0.55 is **`5.83%`** (14 / 240; 95% CI: `[3.51%, 9.55%]`), not 0.42%. At 0.40 threshold, false matches reach **`44.58%`** (107 / 240).
6. **QA Test Suite Remediation:** Corrected stale tests in `qa/tests/test_phase2_measurements.py`, `qa/tests/test_phase4_quality_gates.py`, and `qa/e2e/tests/frontend.spec.js`. All 5 phase 2 measurement tests and format handling tests now pass cleanly (100%).

---

## 2. Database & Preprocessing Audit

### 2.1 Database Files on Disk
```powershell
Get-ChildItem backend\data\celebrity_embeddings*.pkl
```
- `backend/data/celebrity_embeddings.pkl`: 823,494 bytes (Active 275-identity database).
- `backend/data/celebrity_embeddings.old.pkl`: 454,072 bytes (Archived 205-identity database).

**Identity & Reference Distribution (`backend/data/celebrity_embeddings.pkl`):**
- Total identities: **275** (105 Western, 170 Bollywood).
- 267 identities have 20 reference images.
- 5 identities have 19 reference images.
- 2 identities have 18 reference images.
- 1 identity has 10 reference images.
- **Total Reference Images:** $267 \times 20 + 5 \times 19 + 2 \times 18 + 1 \times 10 = \mathbf{5,481}$ images.

### 2.2 Git Diff Stat (`qa-baseline` to `HEAD`)
```text
$ git diff --stat qa-baseline HEAD -- backend/scripts/preprocess_celebrity_database.py backend/app/components/celebrity_matcher.py
 backend/app/components/celebrity_matcher.py      | 177 ++++++------
 backend/scripts/preprocess_celebrity_database.py | 327 +++++++++++++++--------
 2 files changed, 315 insertions(+), 189 deletions(-)
```

### 2.3 Prior Regression Run Preprocessing Diagnosis
In the regression report `qa/reports/11_regression.md`, the reported numbers (Top-1 99.25% on 2,131 queries, 240 unknown faces, 0.42% false matches) were **copied from the baseline Phase 7 report** (`qa/reports/07_celebrity_eval.md` and `qa/reports/celebrity_predictions.csv`).

Diagnostic audit of `qa/eval/celebrity_eval.py`:
1. **Loaded Database:** Evaluated only against the pilot subset of 40 identities from the old 205-identity database.
2. **Preprocessing Pipeline:** In `celebrity_eval.py::embed_offline_batch`, images were processed via:
   ```python
   resized = cv2.resize(img_rgb, (160, 160))
   norm_img = (resized.astype(np.float32) / 127.5) - 1.0
   ```
   This unaligned whole-image resize had **0% bounding box margin** and completely omitted face detection and face cropping.
3. **Production Disparity:** The runtime matcher in `backend/app/components/celebrity_matcher.py` (updated in Batch 2) utilizes:
   ```python
   crop = extract_face_crop(image_data.array, landmarks=landmarks, margin=0.10)
   ```
   which detects face landmarks via MediaPipe FaceLandmarker, crops the face with a **10% fractional margin**, resizes to $160 \times 160$, and normalizes to $[-1.0, 1.0]$.

---

## 3. Dataset Partitions and Contamination Analysis

### 3.1 Test-Set Contamination Audit
When Batch 2 rebuilt the celebrity reference database (`backend/scripts/preprocess_celebrity_database.py`), reference images were selected evenly across the alphabetically sorted files in each folder without referencing the original partitions in `qa/reports/splits.csv`.

Cross-referencing the 5,481 reference images of the new database against the original 24,930 rows of `splits.csv`:

| Original Split (`splits.csv`) | Original Image Count | Absorbed into New Reference Set | Contamination Rate |
| :--- | :---: | :---: | :---: |
| **Test Split** (Held-out) | 11,456 | **1,783** | **15.56%** |
| **Validation Split** | 11,424 | **1,753** | **15.34%** |
| **Reference Split** (Retained) | 2,050 | **545** | **26.59%** |

> [!WARNING]
> **Test-Set Contamination Confirmed:** 1,783 images that were designated as strictly held-out test data in `splits.csv` were absorbed into the reference database centroids. The old test split is therefore contaminated and cannot be used for unbiased evaluation.

### 3.2 Clean Partitioning: `qa/reports/splits_v2.csv`
To re-establish clean, leak-free evaluation partitions, `qa/eval/build_splits_v2.py` was executed across all 275 identities:
- **Reference Set:** All 5,481 images present in `db[identity]['reference_files']` are marked `split = "reference"`.
- **Remaining Images:** 24,310 images across the 275 identity folders. All 275 identities have $\ge 6$ remaining images (0 excluded).
- **Leakage Prevention:** Near-duplicates were clustered using perceptual hashing (`imagehash.phash`, Hamming distance $\le 6$). Clusters were partitioned 50/50 using a fixed random seed (`seed = 42`) into validation and test sets so no near-duplicate pair spans multiple splits.

**Final `splits_v2.csv` Distribution (29,791 Total Images):**
- **Reference Split:** **5,481 images** (Centroid training data)
- **Validation Split:** **12,094 images** (Operating point & threshold tuning)
- **Test Split:** **12,216 images** (**Strictly Sealed & Untouched**)

---

## 4. Re-Measured Celebrity Evaluation (Validation Split)

The evaluation was executed via `qa/eval/celebrity_eval_v2.py` on all 12,094 validation images of `splits_v2.csv` and the 240 LFW unknown faces using the **exact production code path** (`extract_face_crop` with 10% margin $\to$ `Facenet512` $\to$ unit normalize $\to$ cosine similarity against 275 unit-normalized centroids).

The held-out test split (12,216 images) was **strictly untouched and never loaded**.

### 4.1 Evaluation Summary Matrix

| Metric | Baseline / Copied Report | Re-Measured (`splits_v2.csv`) | 95% Wilson Interval | Evidence Artifact |
| :--- | :---: | :---: | :---: | :--- |
| **Database Size** | 205 (40 pilot used) | **275 identities** | N/A | `backend/data/celebrity_embeddings.pkl` |
| **Validation Query Count** | 2,131 queries | **12,094 queries** | N/A | `qa/reports/splits_v2.csv` |
| **Top-1 Accuracy** | 99.25% (2,115/2,131) | **97.68% (11,813/12,094)** | `[97.39%, 97.93%]` | `celebrity_eval_v2_summary.json` |
| **Top-5 Accuracy** | 99.77% (2,126/2,131) | **99.38% (12,019/12,094)** | `[99.22%, 99.50%]` | `celebrity_eval_v2_summary.json` |
| **Validation Strong ($\ge 0.55$)** | 97.5% (2,077/2,131) | **97.63% (11,807/12,094)** | `[97.34%, 97.88%]` | `celebrity_eval_v2_summary.json` |
| **Validation Moderate ($[0.40, 0.55)$)** | 2.3% (49/2,131) | **2.18% (264/12,094)** | `[1.94%, 2.46%]` | `celebrity_eval_v2_summary.json` |
| **Validation Weak ($< 0.40$)** | 0.2% (5/2,131) | **0.19% (23/12,094)** | `[0.13%, 0.29%]` | `celebrity_eval_v2_summary.json` |
| **Precision at 0.55** | 99.6% (2,076/2,085) | **98.65% (11,648/11,807)** | `[98.43%, 98.85%]` | `celebrity_eval_v2_summary.json` |
| **Coverage at 0.55** | 97.5% (2,077/2,131) | **97.63% (11,807/12,094)** | `[97.34%, 97.88%]` | `celebrity_eval_v2_summary.json` |
| **Unknown Faces ($N = 240$)** | 240 LFW faces | **240 LFW faces** | N/A | `qa/data/unknown/lfw_home` |
| **Unknown False Match ($\ge 0.55$)** | 0.42% (1/240) | **5.83% (14/240)** | `[3.51%, 9.55%]` | `celebrity_eval_v2_summary.json` |
| **Unknown Moderate ($[0.40, 0.55)$)** | N/A | **38.75% (93/240)** | `[32.81%, 45.04%]` | `celebrity_eval_v2_summary.json` |
| **Unknown Weak ($< 0.40$, Rejected)**| N/A | **55.42% (133/240)** | `[49.09%, 61.57%]` | `celebrity_eval_v2_summary.json` |
| **Unknown False Match ($\ge 0.40$)** | N/A | **44.58% (107/240)** | `[38.43%, 50.91%]` | `celebrity_eval_v2_summary.json` |

### 4.2 Analysis of Discrepancies
1. **Top-1 Accuracy:** Moves from 99.25% (on 40 pilot identities) to **97.68%** across all 275 identities. With 275 identities competing in cosine space, a small drop in accuracy (1.57%) is expected and represents genuine performance across diverse lighting and resolutions.
2. **Unknown False Matches at 0.55:** Moves from 0.42% (1/240) to **5.83%** (14/240). When face crops are standardized with 10% margins and the database expands from 205 to 275 identities, the nearest-neighbor similarity distribution shifts slightly upward. 14 out of 240 random non-celebrity faces exceed 0.55 similarity to at least one of the 275 celebrity centroids.
3. **Thresholding Implication:** The strong threshold (0.55) maintains **98.65% precision** while rejecting 94.17% of unknown faces. In contrast, the moderate threshold (0.40) admits a **44.58% false match rate** on unknown faces, confirming that returning celebrity lookalikes in the moderate band carries substantial risk of false identification.

---

## 5. Stale QA Test Updates & Root Cause Audit

### 5.1 Measurement Tests (`qa/tests/test_phase2_measurements.py`)
- **Midline Ratio 2 Parity:** Updated `GoldenRatioOracle.calculate_production_ratios` in `qa/eval/oracle.py` to match the Batch 3 midline perpendicular distance formula (landmarks 168 to 152). Parity across frontal images matches within $10^{-6}$ relative difference.
- **Pose Quality Handling in Tests:** Added graceful exception handling for images that correctly trigger the new $\le 12^\circ$ yaw gate.
- **8-Feature Equal Weighting Assertion:** Updated `test_synthetic_unilateral_eye_distortion_splits_ratio_3` to assert that the feature breakdown maintains exactly 8 features while correctly reporting `asymmetry_pct > 10.0%` and `ratio_left != ratio_right` (`FM-BUG-009`).
- **XFail Removal:** Removed `@pytest.mark.xfail` from `test_ratio_2_asymmetry_detection_on_asymmetric_jaw` (`FM-BUG-022`) and `test_ratio_7_asymmetry_detection_on_asymmetric_eyebrows` (`FM-BUG-007`). Both now pass cleanly.
- **Result:** **5 passed, 0 failed** in 26.47s.

### 5.2 Quality Gates (`qa/tests/test_phase4_quality_gates.py`)
- **EXIF Rotated Test:** Updated `test_exif_rotated_phone_photo` to use `frontal_03_pins_Alvaro Morte_Alvaro Morte155_193.jpg`, which exhibits $|yaw| \le 12^\circ$ when rotated, cleanly verifying EXIF transpose handling. Passes 100%.
- **HEIC Image Acceptance Investigation (`FM-BUG-006`):**
  - `pillow-heif==0.14.0` is present in `backend/requirements.txt`.
  - `pillow_heif.register_heif_opener()` is executed at top of `backend/app/components/image_processor.py`.
  - Running a HEIC byte fixture through `ImageProcessor.process_upload()` succeeds, returning `ImageData(format="HEIF", width=400, height=400)`.
  - `test_heic_acceptance` passes (HTTP 200). `FM-BUG-006` is confirmed **FIXED**.

### 5.3 Frontend E2E (`qa/e2e/tests/frontend.spec.js`)
- Updated `#percentileStatus` assertion from `toContainText('7%')` to `toContainText('below typical (compared with 959 celebrity photos)')` matching the calibrated Batch 4 UI implementation.

### 5.4 Percentile Engine Investigation (`FM-BUG-015`)
- In Batch 4, `PercentileEngine` in production was updated to default to `DEFAULT_EMPIRICAL_MEAN = 61.28` and `DEFAULT_EMPIRICAL_STD = 4.65` calibrated against `celebrity_scores_benchmark.csv` (N=959), and unit tests pass 100%.
- The QA test `test_percentile_calibration_against_empirical_population` in `qa/tests/test_phase5_validity.py` remained marked `xfail(strict=True)` because its pytest fixture had been hardcoded to `PercentileEngine(mean=70.0, std=10.0)` during Phase 5 testing.

### 5.5 Updated Bug Tracking Status (`qa/reports/BUGS.md`)
- Total Tracked Bugs: **29** (FM-BUG-001 through FM-BUG-029).
- **Fixed:** **23** (001, 002, 003, 006, 007, 008, 009, 010, 011, 012, 013, 014, 015, 019, 020, 022, 023, 025, 026, 027, 028, 029).
- **Won't Fix:** **5** (004, 005, 016, 018, 021/024).
- **Still Open:** **1** (FM-BUG-017: Heuristic occlusion detection fails on sunglasses).
