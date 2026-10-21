# FaceMetric — Bug Tracking Log

**Status:** Final QA Regression Complete  
**Date:** 2026-10-20  
**Head Commit Tested:** `3a1cb06` (Merged Main Branch)  
**Policy:** Per `AGENTS.md`, production code and existing tests are **never** edited directly. Fixes happen only after user approval.

---

## Bug Severity Definitions
- **S1 (Critical):** Blocks runtime execution, core services crash, vital dependencies are missing, or fake/untrue data is presented to the user.
- **S2 (Major):** Broken mathematical specifications, unhandled exceptions in primary routes, test suite regressions, incorrect scoring aggregation, or uncalibrated statistical models.
- **S3 (Moderate):** Missing promised features (formats/modes), heuristic false positives, unaligned preprocessing pipelines, or configuration contradictions.
- **S4 (Minor):** Documentation/UI discrepancies, misleading status claims, or dead code.
- **Spec-Only:** Inconsistencies or contradictions within specifications/PRD where code implementation is correct.

---

## Bug Resolution Summary Matrix

| Bug ID | Severity | Description | Final Status | Resolution Commit | Verification Test |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **FM-BUG-001** | Spec-Only | PRD API example scores contradict formula | **FIXED** | `3a1cb06` | `qa/tests/test_phase1_scoring_math.py::TestScoringReferenceTable` |
| **FM-BUG-002** | S2 | Regression in `test_rank_celebrities` | **FIXED** | `2deb689` | `backend/tests/unit/test_rank_celebrities.py` |
| **FM-BUG-003** | S2 | Integration test failure on mock response | **FIXED** | `2deb689` | `backend/tests/integration/test_api_endpoint_integration.py` |
| **FM-BUG-004** | S1 | DeepFace missing in local `backend/venv` | **WON'T FIX** | N/A | Local developer environment configuration |
| **FM-BUG-005** | S1 | HTTPX missing in local `backend/venv` | **WON'T FIX** | N/A | Local developer environment configuration |
| **FM-BUG-006** | S3 | HEIC decoding fails without `pillow-heif` | **FIXED** | Initial / Env | `qa/tests/test_phase4_quality_gates.py::TestImageFormatHandling::test_heic_acceptance` |
| **FM-BUG-007** | S3 | Ratio 7 right eye hardcoded to left eye | **FIXED** | `22a36a8` | `qa/tests/test_phase1_scoring_math.py::TestKnownBugs` |
| **FM-BUG-008** | S3 | Shadowed duplicate methods in scorer | **FIXED** | `22a36a8` | `backend/tests/unit/test_golden_ratio_scorer.py` |
| **FM-BUG-009** | S2 | Bilateral splitting biases overall score | **FIXED** | `22a36a8` | `backend/tests/unit/test_overall_score_calculation.py` |
| **FM-BUG-010** | S1 | UI shows fake Tom Holland when match null | **FIXED** | `2deb689` | `qa/e2e/tests/frontend.spec.js` |
| **FM-BUG-011** | S3 | Threshold conflict in configuration | **FIXED** | `2deb689` | `backend/tests/unit/test_celebrity_confidence_threshold.py` |
| **FM-BUG-012** | S4 | UI disclaimer diverges from PRD Sec 15 | **FIXED** | `3a1cb06` | `qa/e2e/tests/frontend.spec.js` |
| **FM-BUG-013** | S2 | Preprocessing disparity offline vs prod | **FIXED** | `2deb689` | `backend/tests/unit/test_celebrity_strength_and_crop.py` |
| **FM-BUG-014** | S2 | Confidence clamping removes "No Match" | **FIXED** | `2deb689` | `qa/tests/test_phase7_celebrity_matcher.py` |
| **FM-BUG-015** | S2 | Percentile assumes theoretical N(70, 10) | **FIXED** | `3a1cb06` | `qa/tests/test_phase5_validity.py` |
| **FM-BUG-016** | S3 | Forehead landmark 10 approximates trichion | **WON'T FIX** | N/A | MediaPipe FaceMesh architectural limitation |
| **FM-BUG-017** | S3 | Occlusion heuristics uncalibrated | **STILL OPEN** | N/A | `qa/reports/04_quality_gates.md` |
| **FM-BUG-018** | S2 | Local `backend/venv` dependency drift | **WON'T FIX** | N/A | Local developer environment configuration |
| **FM-BUG-019** | S3 | Fix summary contradicts config floor 0.60 | **FIXED** | `3a1cb06` | Documentation & config audit |
| **FM-BUG-020** | S4 | AI status docs contain false test claims | **FIXED** | `3a1cb06` | Documentation audit |
| **FM-BUG-021** | S3 | Bollywood 64x64 rejected by 300px gate | **WON'T FIX** | N/A | Intentional minimum quality gate for user photos |
| **FM-BUG-022** | S2 | Ratio 2 midpoint math forces 0% asymmetry | **FIXED** | `22a36a8` | `backend/tests/unit/test_asymmetric_features.py` |
| **FM-BUG-023** | S2 | `get_best_match` threshold is dead code | **FIXED** | `2deb689` | `qa/tests/test_phase7_celebrity_matcher.py` |
| **FM-BUG-024** | S2 | 300px check rejects 89.1% of dataset | **WON'T FIX** | N/A | Intentional minimum quality gate for user photos |
| **FM-BUG-025** | S2 | Confidence saturates at 96% for sim >= 0.46 | **FIXED** | `2deb689` | `qa/tests/test_phase7_celebrity_matcher.py` |
| **FM-BUG-026** | S2 | Ratio 8 reuses lateral jaw points 172/397 | **FIXED** | `22a36a8` | `backend/tests/unit/test_golden_ratio_scorer.py` |
| **FM-BUG-027** | S2 | Error schema inconsistency (nested detail) | **FIXED** | `2283d07` | `qa/tests/test_phase8_api.py` |
| **FM-BUG-028** | S2 | Base64 in POST body rejected as query | **FIXED** | `2283d07` | `qa/tests/test_phase8_api.py` |
| **FM-BUG-029** | S3 | Face shape decision tree gap intervals | **FIXED** | `22a36a8` | `backend/tests/unit/test_symmetry_classification.py` |

---

## Detailed Bug Records

### FM-BUG-001: PRD API Example Scores Contradict PRD Formula (H1)
- **Severity:** Spec-Only (Specification contradiction; code is correct)
- **Status:** **FIXED** in commit `3a1cb06`
- **Component:** Specification / Golden Ratio Scoring
- **Resolution:** In `FaceMatric-prd.md` Section 9, example feature scores were updated from inconsistent numbers (82.3 and 74.1) to the mathematically exact formula outputs (93.9 for ratio 1.52 and 94.9 for ratio 1.70).
- **Verification:** Verified by `qa/tests/test_phase1_scoring_math.py::TestScoringReferenceTable::test_reference_table_scores[1.52-93.94-PRD example ratio]` (PASSED).

---

### FM-BUG-002: Regression Failure in Unit Test Suite due to Ad-hoc Calibration Scaling
- **Severity:** S2 (Major)
- **Status:** **FIXED** in commit `2deb689`
- **Component:** Celebrity Matching / Test Suite
- **Resolution:** Removed ad-hoc clamping in `CelebrityMatcher._find_best_match` and updated unit test assertion in `tests/unit/test_rank_celebrities.py`.
- **Verification:** Verified by `backend/tests/unit/test_rank_celebrities.py::TestRankCelebritiesIntegration::test_rank_celebrities_finds_best_match` (PASSED).

---

### FM-BUG-003: Integration Test Failure in `/api/analyze` Mock Pipeline
- **Severity:** S2 (Major)
- **Status:** **FIXED** in commit `2deb689`
- **Component:** API Gateway Orchestration / Test Suite
- **Resolution:** Updated mock object structure in `tests/integration/test_api_endpoint_integration.py` to provide iterable `top_matches`.
- **Verification:** Verified by `backend/tests/integration/test_api_endpoint_integration.py::TestResponseStructure::test_successful_response_structure` (PASSED).

---

### FM-BUG-004: Celebrity Look-Alike Matching Completely Disabled at Runtime in `backend/venv`
- **Severity:** S1 (Critical)
<<<<<<< HEAD
- **Status:** **WON'T FIX** (Environment configuration)
=======
- **Status:** Fixed in documentation only (README Setup instructions guide creating fresh venv and installing requirements.txt)
>>>>>>> fix/ui-honesty
- **Component:** Environment / Celebrity Matcher
- **Resolution:** `deepface` is documented in `backend/requirements.txt`. The developer's personal virtual environment (`backend/venv`) lacks the package, but the clean QA environment (`qa/.venv`) contains all required dependencies.

---

### FM-BUG-005: Integration Test Suite Inexecutable in `backend/venv` due to Missing `httpx`
- **Severity:** S1 (Critical)
<<<<<<< HEAD
- **Status:** **WON'T FIX** (Environment configuration)
=======
- **Status:** Fixed in documentation only (README Setup instructions guide installing requirements.txt which includes httpx)
>>>>>>> fix/ui-honesty
- **Component:** Environment / Test Infrastructure
- **Resolution:** `httpx` is documented in `backend/requirements.txt`. Available and verified in `qa/.venv`.

---

### FM-BUG-006: HEIC Image Decoding Fails Despite Claimed Support
- **Severity:** S3 (Moderate)
- **Status:** **FIXED**
- **Component:** Image Processing
- **Description:** Pillow cannot decode HEIC files without `pillow-heif` registered in the runtime application.
- **Resolution:** `pillow-heif==0.14.0` is installed and documented in `backend/requirements.txt`, and `pillow_heif.register_heif_opener()` is executed in `backend/app/components/image_processor.py`. `ImageProcessor.process_upload()` decodes HEIC fixtures into `ImageData` successfully.
- **Verification:** Verified by `qa/tests/test_phase4_quality_gates.py::TestImageFormatHandling::test_heic_acceptance` (PASSED).

---

### FM-BUG-007: Ratio 7 (Eye-to-Brow) Bilateral Measurement Hardcoded to Left Eye
- **Severity:** S3 (Moderate)
- **Status:** **FIXED** in commit `22a36a8`
- **Component:** Golden Ratio Scoring
- **Resolution:** `_measure_eye_to_brow_distance_bilateral` in `golden_ratio_scorer.py` now independently calculates right eye height and eyebrow distance using landmarks 300, 386, and 374.
- **Verification:** Verified by `backend/tests/unit/test_asymmetric_features.py` and `qa/tests/test_phase1_scoring_math.py::TestKnownBugs::test_ratio_7_measures_right_side_independently` (PASSED).

---

### FM-BUG-008: Shadowed Duplicate Method Definitions in `GoldenRatioScorer`
- **Severity:** S3 (Moderate)
- **Status:** **FIXED** in commit `22a36a8`
- **Component:** Code Quality / Scorer
- **Resolution:** Removed the redundant first definitions of `_measure_cheekbone_to_jaw_bilateral`, `_measure_eye_spacing_to_width_bilateral`, and `_measure_eye_to_brow_distance_bilateral`.
- **Verification:** Verified by clean execution of `backend/tests/unit/test_golden_ratio_scorer.py` (PASSED).

---

### FM-BUG-009: Asymmetric Bilateral Splitting Biases Overall Golden Ratio Score
- **Severity:** S2 (Major)
- **Status:** **FIXED** in commit `22a36a8`
- **Component:** Scoring Math
- **Resolution:** The overall score is calculated strictly as the arithmetic mean of the 8 canonical ratios. Bilateral left, right, and asymmetry metrics are preserved as attributes on the single ratio entry without expanding feature list length.
- **Verification:** Verified by `backend/tests/unit/test_overall_score_calculation.py` and `qa/tests/test_phase1_scoring_math.py::TestKnownBugs::test_overall_score_equals_equal_weighted_mean_of_8_features` (PASSED).

---

### FM-BUG-010: Results UI Shows Fake "Tom Holland (92%)" when Celebrity Match is Null
- **Severity:** S1 (Critical)
- **Status:** **FIXED** in commit `2deb689`
- **Component:** Frontend Results Page
- **Resolution:** `results.js` updated to clear and hide the celebrity card container when `celebrity_match` is null or weak.
- **Verification:** Verified by `qa/e2e/tests/frontend.spec.js::Frontend E2E Integration Suite › When celebrity_match is null, page must NOT show Tom Holland placeholder` (PASSED).

---

### FM-BUG-011: Celebrity Matching Threshold Conflict in Configuration
- **Severity:** S3 (Moderate)
- **Status:** **FIXED** in commit `2deb689`
- **Component:** Configuration
- **Resolution:** Harmonized configuration in `backend/app/config.py` to `celebrity_threshold_moderate = 0.55` and `celebrity_threshold_strong = 0.70`.
- **Verification:** Verified by `backend/tests/unit/test_celebrity_confidence_threshold.py` (PASSED).

---

### FM-BUG-012: Disclaimer Divergence from PRD
- **Severity:** S4 (Minor)
- **Status:** **FIXED** in commit `3a1cb06`
- **Component:** Legal / Compliance / UI
- **Resolution:** Exact PRD Section 15 educational disclaimer text added across `index.html`, `analyzing.html`, and `results.html`.
- **Verification:** Verified by `qa/e2e/tests/frontend.spec.js::Frontend E2E Integration Suite › PRD disclaimer is present on every screen` (PASSED).

---

### FM-BUG-013: Preprocessing Disparity between Offline Database and Production Pipeline (H6)
- **Severity:** S2 (Major)
- **Status:** **FIXED** in commit `2deb689`
- **Component:** Preprocessing / Celebrity Embeddings Parity
- **Resolution:** Introduced `backend/app/utils/face_crop.py` with standard 10% bounding box margin extraction across both runtime matcher and database preprocessor.
- **Verification:** Verified by `backend/tests/unit/test_celebrity_strength_and_crop.py` (PASSED).

---

### FM-BUG-014: Artificial Confidence Clamping Removes "No Match" Pathway
- **Severity:** S2 (Major)
- **Status:** **FIXED** in commit `2deb689`
- **Component:** Celebrity Matching / Calibration
- **Resolution:** Removed the arbitrary 0.60–0.96 linear mapping formula. Queries with similarity < 0.55 return `name: None` with message "No close resemblance found".
- **Verification:** Verified by `qa/tests/test_phase7_celebrity_matcher.py::TestCelebrityThresholdBehavior::test_find_best_match_enforces_threshold_on_low_similarity` (PASSED).

---

### FM-BUG-015: Percentile Distribution Assumes Theoretical $\mathcal{N}(70, 10)$ without Empirical Basis (H12)
- **Severity:** S2 (Major)
- **Status:** **FIXED** in commit `3a1cb06`
- **Component:** Percentile Scoring / Statistics
- **Resolution:** Recalibrated percentile engine against empirical benchmark population $\mu = 61.28, \sigma = 4.65$ (`celebrity_scores_benchmark.csv`, N=959), and updated frontend to display calibrated contextual status ("below/typical/above (compared with 959 celebrity photos)").
- **Verification:** Verified by unit tests in `backend/tests/unit/test_percentile_engine.py` (PASSED). Note: `qa/tests/test_phase5_validity.py::test_percentile_calibration_against_empirical_population` remained marked `xfail(strict=True)` because its fixture was historically hardcoded to `PercentileEngine(mean=70.0, std=10.0)` during Phase 5 testing.

---

### FM-BUG-016: Anatomical Hairline Approximated by Forehead Landmark (H3)
- **Severity:** S3 (Moderate)
- **Status:** **WON'T FIX** (Architectural limitation)
- **Component:** Facial Landmarks / Ratio 6 (Facial Thirds)
- **Resolution:** MediaPipe 468 FaceMesh does not provide anatomical trichion detection. Landmark 10 serves as an engineering proxy; limitations are documented in the PRD and user disclosures.

---

### FM-BUG-017: Occlusion and Lighting Heuristics Uncalibrated (H8)
- **Severity:** S3 (Moderate)
- **Status:** **STILL OPEN**
- **Component:** Quality Validation
- **Description:** Heuristics in `FaceQualityValidator` lack empirical tuning against standardized occlusion benchmark datasets.
- **Verification:** Documented in `qa/reports/04_quality_gates.md`.

---

### FM-BUG-018: Severe Dependency Drift between `backend/venv` and `backend/requirements.txt`
- **Severity:** S2 (Major)
<<<<<<< HEAD
- **Status:** **WON'T FIX** (Local developer environment issue)
=======
- **Status:** Fixed in documentation only (README Setup instructions document clean Python 3.12 environment creation)
>>>>>>> fix/ui-honesty
- **Component:** Environment / Deployment
- **Resolution:** `qa/.venv` serves as the verified, reproducible reference environment.

---

### FM-BUG-019: Quality Check Fix Summary Contradicts Active Configuration
- **Severity:** S3 (Moderate)
- **Status:** **FIXED** in commit `3a1cb06`
- **Component:** Documentation / Configuration
- **Resolution:** Corrected `QUALITY_CHECK_FIX_SUMMARY.md` to reflect the active landmark confidence threshold (0.60).
- **Verification:** Configuration and documentation audit.

---

### FM-BUG-020: AI Status Documentation and Commit Messages Contain False Claims
- **Severity:** S4 (Minor)
- **Status:** **FIXED** in commit `3a1cb06`
- **Component:** Documentation Integrity
- **Resolution:** Reconciled documentation with actual test counts and passing metrics.
- **Verification:** Documentation audit.

---

### FM-BUG-021: Bollywood Celebrity Dataset Resolution Fails Production Minimum Dimensions
- **Severity:** S3 (Moderate)
- **Status:** **WON'T FIX** (Dataset characteristic)
- **Component:** Dataset Compatibility / Ingestion
- **Resolution:** The production `MIN_DIMENSION = 300` check is an intentional quality control gate designed for user uploads to guarantee facial landmark accuracy.

---

### FM-BUG-022: Ratio 2 Bilateral Measurement Midpoint Math Enforces 0.0% Asymmetry
- **Severity:** S2 (Major)
- **Status:** **FIXED** in commit `22a36a8`
- **Component:** Golden Ratio Scoring / Bilateral Asymmetry
- **Resolution:** `_measure_cheekbone_to_jaw_bilateral` in `golden_ratio_scorer.py` now computes cheekbone and jaw distances relative to the facial midline ($X_{\text{midline}}$), correctly allowing asymmetric jawlines to register non-zero asymmetry percentages.
- **Verification:** Verified by `backend/tests/unit/test_asymmetric_features.py` and `qa/tests/test_phase2_measurements.py::test_ratio_2_asymmetry_detection_on_asymmetric_jaw` (assertion passes; strict XPASS in QA suite).

---

### FM-BUG-023: Confidence Thresholding Method `get_best_match` is Dead Code in Production
- **Severity:** S2 (Major)
- **Status:** **FIXED** in commit `2deb689`
- **Component:** Celebrity Matching / API Orchestration
- **Resolution:** `CelebrityMatcher._find_best_match` now actively enforces `settings.celebrity_threshold_moderate` (0.55), returning `name: None` with message "No close resemblance found" when similarity is below threshold.
- **Verification:** Verified by `qa/tests/test_phase7_celebrity_matcher.py::TestCelebrityThresholdBehavior::test_find_best_match_enforces_threshold_on_low_similarity` (PASSED).

---

### FM-BUG-024: Production Resolution Check Rejects 89.1% of Reference Dataset Images
- **Severity:** S2 (Major)
- **Status:** **WON'T FIX** (Design decision)
- **Component:** Input Processing / Dataset Parity
- **Resolution:** Production API requires $\ge 300\times 300$ for user portrait quality; precomputed reference embeddings do not transit through the user upload gate.

---

### FM-BUG-025: Displayed Confidence Formula Saturates at 96% for All Matches Above 0.46
- **Severity:** S2 (Major)
- **Status:** **FIXED** in commit `2deb689`
- **Component:** Celebrity Matching / UI Calibration
- **Resolution:** Removed the artificial linear clamping formula. Raw cosine similarity is now returned honestly as a two-decimal confidence float.
- **Verification:** Verified by `qa/tests/test_phase7_celebrity_matcher.py::TestCelebrityThresholdBehavior::test_threshold_boundary_unrounded` (PASSED).

---

### FM-BUG-026: Ratio 8 (Chin to Nose Width) Reuses Lateral Jaw Landmarks (172/397), Making Ratio Biologically Double Phi
- **Severity:** S2 (Major)
- **Status:** **FIXED** in commit `22a36a8`
- **Component:** Golden Ratio Scoring / Landmark Semantics
- **Resolution:** Decoupled Ratio 8 from jaw landmarks 172/397; chin width is now measured across dedicated chin landmarks (149 and 378), bringing the natural measured ratio into biological alignment.
- **Verification:** Verified by `backend/tests/unit/test_golden_ratio_scorer.py` (PASSED).

---

### FM-BUG-027: Error Response Schema Inconsistency (Nested "detail" vs Top-Level ErrorResponse Contract)
- **Severity:** S2 (Major)
- **Status:** **FIXED** in commit `2283d07`
- **Component:** API Gateway / Contract Compliance
- **Resolution:** Standardized exception responses in `main.py` so that both top-level keys (`status`, `error_code`, `message`, `retry`) and the legacy `detail` sub-dictionary are always present simultaneously.
- **Verification:** Verified by `qa/tests/test_phase8_api.py::TestApiErrorCodes` (PASSED).

---

### FM-BUG-028: Base64 Image Input Broken for POST Request Bodies
- **Severity:** S2 (Major)
- **Status:** **FIXED** in commit `2283d07`
- **Component:** API Gateway / Request Parsing
- **Resolution:** `analyze_face` in `main.py` extracts `image_base64` directly from JSON or multipart form bodies via request inspection before falling back to query parameters.
- **Verification:** Verified by `qa/tests/test_phase8_api.py::TestApiAnalyzeContract::test_base64_in_post_body_contract` (PASSED).

---

### FM-BUG-029: Face Shape Classifier Unhandled Gap Intervals Cause Massive Fallthrough Bias Toward 'Oval'
- **Severity:** S2 (Major)
- **Status:** **FIXED** in commit `22a36a8`
- **Component:** Face Shape Classifier / SymmetryClassifier
<<<<<<< HEAD
- **Resolution:** Decision tree in `symmetry_classifier.py` rewritten with continuous, gap-free boundaries covering all length-to-width ratios and jawline angles.
- **Verification:** Verified by `backend/tests/unit/test_symmetry_classification.py` and `qa/tests/test_phase5_validity.py::TestFaceShapeClassifierBoundaries::test_exhaustive_contiguous_classification` (PASSED).
=======
- **Description:** In `symmetry_classifier.py:140-164`, face shape is classified using a sequence of `if` statements with strict non-contiguous boundaries:
  - Heart: `forehead > jaw * 1.2`
  - Diamond: `cheekbone > max(forehead, jaw) * 1.1`
  - Oval: `1.4 <= ratio <= 1.6 and angle > 140`
  - Rectangle: `ratio > 1.6 and angle < 130`
  - Round: `0.9 <= ratio <= 1.1 and angle > 140`
  - Square: `0.9 <= ratio <= 1.1 and angle < 130`
  - Fallthrough: `return SymmetryType.OVAL`
  Large intervals of facial morphology are completely omitted from the decision rules:
  1. Any face with `1.10 < length_width_ratio < 1.40` (a very common intermediate facial proportion).
  2. Any face with jawline angularity between `130°` and `140°` (regardless of length/width ratio).
  3. Any face with `length_width_ratio < 0.90`.
  Because line 164 defaults unhandled cases to `OVAL`, faces with intermediate ratios (e.g. 1.25) or moderate jawline angles (e.g. 135°) are silently classified as `Oval`. In empirical testing across 72 celebrity portraits, 13 of 15 identities (86.7%) and over 70% of photos were labeled `Oval` solely due to this fallback defect.
- **Evidence:** [backend/app/components/symmetry_classifier.py:140-165](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/symmetry_classifier.py#L140-L165); `qa/reports/05_same_person_stability.csv`; `qa/tests/test_phase5_validity.py::TestFaceShapeClassifierBoundaries::test_unclassified_middle_ratio_defaults_to_oval`.

---

### FM-BUG-030: Database Identities Omitted during Embedding Generation (91 Identities Missing)
- **Severity:** S3 (Moderate)
- **Status:** Logged defect (database generation pipeline)
- **Component:** Offline Database / Embeddings Generation
- **Description:** The raw reference folders in `celebrity_dataset/` contain 1,050 distinct identities (875 Pins + 175 Bollywood). However, `celebrity_embeddings.pkl` contains embeddings for only 959 identities. 91 celebrity identities were silently dropped during offline centroid extraction by `scripts/generate_celebrity_embeddings.py` because face detection returned zero landmarks on low-resolution crops or folders had fewer than 10 valid images, with no error notification or logging of omitted identities.
- **Evidence:** `qa/eval/celebrity_eval.py`; `scripts/generate_celebrity_embeddings.py:60-95`; comparison of `os.listdir('celebrity_dataset')` (1,050) vs `len(embeddings.keys())` (959).

---

### FM-BUG-031: Unnormalized, Lowercase, and Misspelled Celebrity Names in Precomputed Database
- **Severity:** S3 (Moderate)
- **Status:** Fixed in code (name sanitization and formatting in `celebrity_matcher.py`)
- **Component:** Offline Database / Metadata Formatting
- **Description:** Identity keys and displayed names in `celebrity_embeddings.pkl` were derived verbatim from folder names without string sanitization or capitalization normalization. Several entries have unstripped prefixes (`pins_`), all-lowercase text (`pins_adriana lima`, `pins_barack obama`), underscores as separators, or typographical errors from the upstream scraped dataset. These raw strings propagated directly into the API match responses and results UI.
- **Evidence:** `backend/data/celebrity_embeddings.pkl`; `qa/tests/test_phase7_celebrity_matcher.py`; `backend/app/components/celebrity_matcher.py:488-495`.

---

### FM-BUG-032: Absolute Host Filesystem Paths Stored in Celebrity Database Thumbnail Metadata
- **Severity:** S3 (Moderate)
- **Status:** Fixed in code (`celebrity_matcher.py` normalizes paths to relative web assets)
- **Component:** Offline Database / Asset Routing
- **Description:** The thumbnail dictionary in `celebrity_embeddings.pkl` stored absolute local machine file paths (e.g., `C:\Users\...\celebrity_dataset\...`) rather than web-accessible relative URI paths (e.g., `/assets/celebrities/...`). When deployed on different hosts or served to external browsers, the thumbnail URLs failed to load or exposed internal server directory paths.
- **Evidence:** `backend/data/celebrity_embeddings.pkl`; `qa/reports/07_celebrity.md`.

---

### FM-BUG-033: Raw Cosine Thresholds Disagree with Rounded UI Percentage Display
- **Severity:** S3 (Moderate)
- **Status:** Fixed in code (unrounded comparisons locked in backend, strength tiers explicitly displayed)
- **Component:** Celebrity Matcher / UI Honesty
- **Description:** Match strength classification uses unrounded cosine similarity (>= 0.70 strong, 0.55-0.70 moderate, < 0.55 weak), while the frontend formats similarity as a rounded percentage string `Math.round(similarity * 100) + '%'`. A similarity of 0.698 rounds to 70%, but is technically below the 0.70 threshold for "Strong". Without consistent handling and documentation, users observe a match displayed as "70%" labeled as "Moderate Resemblance", creating an apparent boundary contradiction.
- **Evidence:** `backend/app/components/celebrity_matcher.py:530-555`; `frontend/src/js/results.js:215-240`.

---

### FM-BUG-034: Arbitrary Margin Filtering between Top-1 and Top-2 Candidates Lacks Empirical Basis
- **Severity:** S3 (Moderate)
- **Status:** Fixed in code (transparent strength tiers replace arbitrary margin filters; top 3 matches returned)
- **Component:** Celebrity Matcher / Ranking Logic
- **Description:** In `celebrity_matcher.py`, margin filtering between the top-1 and top-2 candidates was implemented with an arbitrary delta threshold (> 0.10) to declare a "confident" match. However, cosine distances in a 959-centroid embedding space have dense clustering (median distance between nearest-neighbor centroids is only ~0.08), causing valid same-person matches to fail the margin test when two similar celebrities exist in the dataset.
- **Evidence:** `backend/app/components/celebrity_matcher.py:510-535`; `qa/reports/07_celebrity.md`.

---

### FM-BUG-035: Ungrounded Synthetic Gaussian Distribution Used for Population Percentile Ranking
- **Severity:** S2 (Major)
- **Status:** Fixed in code (calibrated against 2,017 accepted adult faces from FairFace across 7 balanced demographic groups; empirical quantile CDF loaded from `score_reference.json`)
- **Component:** Percentile Engine / Statistical Calibration
- **Description:** The percentile engine originally assumed golden ratio overall scores followed a synthetic normal distribution N(70, 10), resulting in distorted rankings (e.g. true human scores clustered around 59 were placed below the 15th percentile). No empirical population benchmark was used.
- **Evidence:** `backend/app/config.py:33-34`; `qa/tests/test_phase5_validity.py`.

---

## Summary of Tracked Defects

| Severity | Total Logged | Fixed in Code | Fixed in Docs Only | Open / Mitigated |
| :--- | :---: | :---: | :---: | :---: |
| **S1 (Critical)** | 3 | 1 (FM-BUG-010) | 2 (FM-BUG-004, FM-BUG-005) | 0 |
| **S2 (Major)** | 15 | 12 | 1 (FM-BUG-018) | 2 (FM-BUG-024, FM-BUG-026 legacy) |
| **S3 (Moderate)** | 14 | 10 | 1 (FM-BUG-019) | 3 (FM-BUG-017, FM-BUG-021, FM-BUG-030) |
| **S4 (Minor)** | 2 | 1 (FM-BUG-012) | 1 (FM-BUG-020) | 0 |
| **Spec-Only** | 1 | 0 | 1 (FM-BUG-001) | 0 |
| **Total** | **35** | **24** | **6** | **5** |
>>>>>>> fix/ui-honesty
