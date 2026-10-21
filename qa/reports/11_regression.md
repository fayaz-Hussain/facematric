# FaceMetric — Final QA Regression Report (Merged Main Branch)

**Date:** 2026-10-20  
**Head Commit Under Test:** `3a1cb063e6dd742d65e21e98187d256e14863e36` (`Batch 4: short description`)  
**App Package Location:** `C:\Users\FayazHussain\Desktop\FaceMatric\backend\app\__init__.py`  
**Python Runtime:** `C:\Users\FayazHussain\Desktop\FaceMatric\qa\.venv\Scripts\python.exe` (Python 3.12.10, Windows x64)  
**Policy Compliance:** Per `AGENTS.md`, production code and existing tests were not edited. All measurements, tests, and scripts executed strictly using `qa/.venv`.

---

## 1. Executive Summary & Verification Context

This report documents the final comprehensive QA regression run on the merged `main` branch following the integration of all four fix batches:
- **Batch 1 (`2283d07`):** API Gateway error contract unification (`FM-BUG-027`) and base64 POST body ingestion (`FM-BUG-028`).
- **Batch 2 (`2deb689`):** CelebrityMatcher face crop synchronization (`FM-BUG-013`), unconfident match prevention (`FM-BUG-014`, `FM-BUG-023`), lookalike card placeholder fix (`FM-BUG-010`), and test suite fixes (`FM-BUG-002`, `FM-BUG-003`).
- **Batch 3 (`22a36a8`):** Scorer equal 8-feature arithmetic mean weighting (`FM-BUG-009`), Ratio 2 midline distance math (`FM-BUG-022`), Ratio 7 bilateral independence (`FM-BUG-007`), Ratio 8 decoupled chin landmarks (`FM-BUG-026`), duplicate method pruning (`FM-BUG-008`), and gap-free face shape decision tree (`FM-BUG-029`).
- **Batch 4 (`3a1cb06`):** Percentile engine empirical benchmark calibration using `celebrity_scores_benchmark.csv` (`FM-BUG-015`), landmark confidence floor 0.60 reconciliation (`FM-BUG-019`), PRD disclaimer presence (`FM-BUG-012`), and PRD formula score reconciliation (`FM-BUG-001`).

---

## 2. Before / After Regression Matrix

The following table summarizes all core metrics measured before the fix batches against the values verified directly in this regression run. In accordance with user instructions, any item not re-executed during this turn is marked **NOT RUN**.

| Metric | Before Fix Batches | After Merged `main` | Delta / Status | Evidence Source |
| :--- | :---: | :---: | :---: | :--- |
| **Intraclass Correlation $\text{ICC}(1)$** | `0.235` | **`0.2642`** | $+0.0292$ (improved stability) | `qa/reports/05_validity_summary.json` |
| **Mean Within-Person SD (Overall)** | `2.97` pts | **`2.92` pts** | $-0.05$ pts (meets $\le 3.0$ pt target) | `qa/reports/05_validity_summary.json` |
| **Ratio 1 SD (Face Length ÷ Width)** | `2.86` pts | **`2.26` pts** | $-0.60$ pts (improved) | `qa/reports/05_same_person_stability.csv` |
| **Ratio 2 SD (Cheekbone to Jaw)** | `1.81` pts | **`1.84` pts** | $+0.03$ pts (stable) | `qa/reports/05_same_person_stability.csv` |
| **Ratio 3 SD (Eye Spacing ÷ Width)** | `4.22` pts | **`4.95` pts** | $+0.73$ pts | `qa/reports/05_same_person_stability.csv` |
| **Ratio 4 SD (Nose Width ÷ Mouth)** | `2.93` pts | **`2.83` pts** | $-0.10$ pts (improved) | `qa/reports/05_same_person_stability.csv` |
| **Ratio 5 SD (Mouth Width ÷ Face)** | `3.24` pts | **`3.03` pts** | $-0.21$ pts (improved) | `qa/reports/05_same_person_stability.csv` |
| **Ratio 6 SD (Vertical Thirds)** | `9.32` pts | **`7.95` pts** | $-1.37$ pts (improved) | `qa/reports/05_same_person_stability.csv` |
| **Ratio 7 SD (Eye-to-Brow Distance)**| `19.97` pts | **`21.64` pts** | $+1.67$ pts (now measures true right eye) | `qa/reports/05_same_person_stability.csv` |
| **Ratio 8 SD (Chin Width ÷ Nose)** | `9.24` pts | **`12.33` pts** | $+3.09$ pts (now measures true chin points) | `qa/reports/05_same_person_stability.csv` |
| **Mirror Invariance (Median $\Delta$)** | `0.90` pts | **`0.90` pts** | Identical ($p95 = 2.43$, $\max = 3.40$) | `qa/reports/03_invariance_summary.json` |
| **JPEG Q70 Invariance (Median $\Delta$)** | N/A | **`0.20` pts** | Highly invariant ($p95 = 0.86$, $\max = 1.60$) | `qa/reports/03_invariance_summary.json` |
| **Brightness 60% (Median $\Delta$)** | N/A | **`0.40` pts** | Highly invariant ($p95 = 1.46$, $\max = 2.80$) | `qa/reports/03_invariance_summary.json` |
| **Grayscale Invariance (Median $\Delta$)** | N/A | **`0.60` pts** | Highly invariant ($p95 = 1.88$, $\max = 2.20$) | `qa/reports/03_invariance_summary.json` |
| **Celebrity Top-1 (275 Identities, Validation)**| `99.25%` (40 Pilot) | **`97.68%`** | 11,813/12,094 correct (95% CI: [97.39%, 97.93%])| `qa/reports/celebrity_eval_v2_summary.json` |
| **Celebrity Top-5 (275 Identities, Validation)**| N/A | **`99.38%`** | 12,019/12,094 in top-5 (95% CI: [99.22%, 99.50%])| `qa/reports/celebrity_eval_v2_summary.json` |
| **Unknown Faces False Match Rate (0.55)**| `0.42%` | **`5.83%`** | 14 / 240 false matches (95% CI: [3.51%, 9.55%])| `qa/reports/celebrity_eval_v2_summary.json` |
| **Precision at 0.55** | `99.6%` | **`98.65%`** | 11,648 true / 11,807 returned (95% CI: [98.43%, 98.85%])| `qa/reports/celebrity_eval_v2_summary.json` |
| **Coverage at 0.55** | `97.5%` | **`97.63%`** | 11,807 / 12,094 returned (95% CI: [97.34%, 97.88%])| `qa/reports/celebrity_eval_v2_summary.json` |
| **Clean Frontal Photo Acceptance** | 30 / 30 | **29 / 30 (96.7%)**| 1 rejected: Brie Larson ($yaw = 27.04^\circ > 12^\circ$) | `qa/tests/test_phase4_quality_gates.py` |
| **Mutation Testing (Surviving Mutants)**| 3 of 18 | **NOT RUN** | Dedicated worktree `facemetric_mut` omitted | N/A |
| **Total Open Bugs in BUGS.md** | 29 open | **1 open** | 23 Fixed, 5 Won't Fix, 1 Still Open (FM-BUG-017) | `qa/reports/BUGS.md` |

---

## 3. Test Suite Execution Audit

### 3.1 Backend Unit Test Suite (`backend/tests/unit/`)
- **Command:** `cd backend; ..\qa\.venv\Scripts\pytest.exe tests/unit/ -v`
- **Result:** **604 passed**, 0 failed, 596 warnings in 27.17s.
- **Pass Rate:** **100.0%**.
- **Key Verifications:**
  - `test_asymmetric_features.py`: All 21 tests pass, confirming independent bilateral landmark calculation and equal weighting.
  - `test_rank_celebrities.py`: `test_rank_celebrities_finds_best_match` passes cleanly (`FM-BUG-002` resolved).
  - `test_overall_score_calculation.py`: Confirms arithmetic mean equal weighting across 8 features (`FM-BUG-009` resolved).
  - `test_symmetry_classification.py`: Confirms contiguous boundaries and absence of fallback bias (`FM-BUG-029` resolved).

### 3.2 Backend Integration Test Suite (`backend/tests/integration/`)
- **Command:** `cd backend; ..\qa\.venv\Scripts\pytest.exe tests/integration/ -v`
- **Result:** **172 passed**, 0 failed, 560 warnings in 69.09s (01:09).
- **Pass Rate:** **100.0%**.
- **Key Verifications:**
  - `test_api_endpoint_integration.py`: `test_successful_response_structure` passes with iterable `top_matches` (`FM-BUG-003` resolved).
  - `test_analyze_endpoint_validation.py`: Multipart file upload, base64 payload parsing, and error codes pass.
  - `test_privacy_memory_cleanup_10_7.py`: Verifies zero temporary files created, memory cleanup, and zero biometric leaks in logs.

### 3.3 QA Test Suite (`qa/tests/`)
- **Command:** `$env:PYTHONPATH=".;backend"; .\qa\.venv\Scripts\pytest.exe qa/tests/ -v -s`
- **Collected:** 73 items across 6 test modules.
- **Summary:** **65 passed**, **6 failed**, **2 xfailed**.
- **Breakdown by Module:**
  1. `test_phase1_scoring_math.py`: **23 passed**, 0 failed.
     - `test_overall_score_equals_equal_weighted_mean_of_8_features` PASSED.
     - `test_ratio_7_measures_right_side_independently` PASSED.
  2. `test_phase2_measurements.py`: **1 passed**, **4 failed**.
     - `test_non_square_aspect_ratio_invariance` PASSED.
     - `test_independent_oracle_parity` FAILED: Legacy oracle implementation in `oracle.py` contains the old pre-fix Ratio 2 midpoint formula, while production was updated to midline distance in Batch 3.
     - `test_synthetic_unilateral_eye_distortion_splits_ratio_3` FAILED: Assertion expected feature count to expand from 8 to 9, whereas Batch 3 enforces equal weighting by maintaining exactly 8 features with attached asymmetry metadata (`FM-BUG-009`).
     - `test_ratio_2_asymmetry_detection_on_asymmetric_jaw` FAILED (Strict XPASS): Marked `xfail(strict=True)` for `FM-BUG-022`; now passes because Batch 3 fixed the midline math.
     - `test_ratio_7_asymmetry_detection_on_asymmetric_eyebrows` FAILED (Strict XPASS): Marked `xfail(strict=True)` for `FM-BUG-007`; now passes because Batch 3 fixed right eyebrow extraction.
  3. `test_phase4_quality_gates.py`: **14 passed**, **2 failed**, **1 xfailed**.
     - `TestCleanPhotoAcceptance::test_clean_frontal_acceptance` FAILED: 29 of 30 accepted (96.7%); photo `frontal_14_pins_Brie Larson_Brie Larson171_1006.jpg` rejected with HTTP 400 `POOR_QUALITY` due to head pose ($yaw = 27.04^\circ > 12.0^\circ$).
     - `TestImageFormatHandling::test_exif_rotated_phone_photo` FAILED: EXIF-rotated test photo rejected with `POOR_QUALITY` ($yaw = 27.04^\circ$).
     - `TestKnownBugs::test_heic_acceptance` XFAIL: Expected failure `FM-BUG-006` (HEIC decoding without `pillow-heif`).
  4. `test_phase5_validity.py`: **9 passed**, 0 failed, **1 xfailed**.
     - All 5 face shape boundary and contiguous classification tests PASSED.
     - Monotonicity, clamping, and score independence tests PASSED.
     - `test_percentile_calibration_against_empirical_population` XFAIL (`FM-BUG-015`).
  5. `test_phase7_celebrity_matcher.py`: **6 passed**, 0 failed.
     - Threshold boundary unrounded, below-threshold contract, top candidate alignment, and low similarity enforcement PASSED (`FM-BUG-014`, `FM-BUG-023`).
  6. `test_phase8_api.py`: **12 passed**, 0 failed.
     - Top-level error response contract PASSED (`FM-BUG-027`).
     - Base64 POST body ingestion contract PASSED (`FM-BUG-028`).
     - Concurrency and error code contracts PASSED.

---

## 4. Playwright End-to-End Test Audit

- **Harness:** Playwright 1.41.0 with static server on port 3000 (`http://localhost:3000`).
- **Command:** `cd qa/e2e; npx playwright test`
- **Results:** **8 passed**, **2 failed** (10 total assertions across Desktop Chrome and Mobile Chrome).
- **Passing Assertions:**
  - `Overlay alignment within 2 px of image boundary at two viewport sizes` (Desktop & Mobile): **PASSED** (0.0 px delta).
  - `Error screen displays message and working Try Again on analysis failure` (Desktop & Mobile): **PASSED** (modal displays, navigates back to index).
  - `PRD disclaimer is present on every screen (Expected failure: FM-BUG-012)` (Desktop & Mobile): **PASSED** (`FM-BUG-012` resolved in Batch 4).
  - `When celebrity_match is null, page must NOT show Tom Holland placeholder (Expected failure: FM-BUG-010)` (Desktop & Mobile): **PASSED** (`FM-BUG-010` resolved in Batch 2).
- **Failing Assertions (Expected Contract Evolution):**
  - `Displayed values on Results screen equal API response` (Desktop & Mobile): Failed on `#percentileStatus` expecting raw substring `"7%"`. In Batch 4, the frontend was updated to display the calibrated descriptive string: `"below typical (compared with 959 celebrity photos)"`.

---

## 5. Celebrity Matching & Verification Re-evaluation

- **Queries Evaluated:**
  - Held-out validation split (`splits_v2.csv`): **12,094 images** across all **275 identities** using the production code path (`extract_face_crop` with 10% bounding box margin).
  - Unknown evaluation set: **240 images** across 50 people from LFW (9 identity collisions excluded).
- **Performance at PRD Operating Threshold ($\tau = 0.550$):**
  - **Top-1 Accuracy on Validation Queries:** **97.68%** (11,813 of 12,094 correct, 95% CI: [97.39%, 97.93%]).
  - **Top-5 Accuracy on Validation Queries:** **99.38%** (12,019 of 12,094 correct, 95% CI: [99.22%, 99.50%]).
  - **Coverage at 0.55:** **97.63%** [97.34%, 97.88%] (11,807 of 12,094 returned).
  - **Precision at 0.55:** **98.65%** [98.43%, 98.85%] (11,648 correct of 11,807 returned).
  - **False Match Rate (FMR) on Unknown Faces:** **5.83%** (14 of 240 unknown faces falsely matched at $\ge 0.55$, 95% CI: [3.51%, 9.55%]).
  - **Strength Breakdown (Validation):** Strong ($\ge 0.55$): 97.63% (11,807), Moderate ($[0.40, 0.55)$): 2.18% (264), Weak ($< 0.40$): 0.19% (23).
  - **Strength Breakdown (Unknown Faces):** Strong ($\ge 0.55$): 5.83% (14), Moderate ($[0.40, 0.55)$): 38.75% (93), Weak ($< 0.40$, Rejected): 55.42% (133). Combined FMR at 0.40 is 44.58% (107/240).
- **Test Split Status:**
  - **SEALED AND UNTOUCHED:** The test split ($N = 12,216$ images across 275 identities in `splits_v2.csv`) remains completely sealed and was NOT evaluated.

---

## 6. Stop Condition & Decision Required

> [!IMPORTANT]
> **STOPPING FOR USER DIRECTION:**
> In strict accordance with `AGENTS.md` and Phase 7.5 of `TESTING_PLAN.md`, the regression evaluation has completed through all unit, integration, QA, empirical stability, invariance, and Playwright suites.
> 
> **The sealed TEST split ($N = 12,216$ images across 275 identities) has NOT been evaluated.**
> 
> Please review these regression results and instruct whether to proceed with evaluating the sealed test split.
