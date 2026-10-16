# FaceMetric — Test Suite Audit Report (Phase 0.5)

**Date:** 2026-10-14  
**Scope:** Line and branch coverage, test paradigm classification, tautology evaluation, mutation testing across 18 mutants in a throwaway worktree, and test smell inventory.  
**Execution Environment:** `qa/.venv` (Python 3.12.10, Pytest 7.4.3, Pytest-Cov 4.1.0). Main repository remained 100% clean and unmodified.

---

## 1. Code Coverage (Line & Branch)

The test suites were executed with `--cov=app --cov-branch` under `qa/.venv`. The two pre-existing baseline failures ([FM-BUG-002](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/BUGS.md#FM-BUG-002) in `test_rank_celebrities.py` and [FM-BUG-003](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/BUGS.md#FM-BUG-003) in `test_api_endpoint_integration.py`) are excluded as expected regressions.

### Module Coverage Breakdown

| Module | Statements | Missing | Branch | Partial Branch | Unit Cov | Int Cov | Combined Cov | Uncovered Branches & Gaps |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `app/components/golden_ratio_scorer.py` | 200 | 32 | 8 | 0 | 83% | 22% | **83%** | Duplicate shadowed methods (lines 276–385) completely unreached. |
| `app/components/symmetry_classifier.py` | 96 | 1 | 12 | 1 | 98% | 32% | **98%** | Default fallback branch on ambiguous ratios. |
| `app/components/percentile_engine.py` | 111 | 9 | 28 | 3 | 91% | 17% | **91%** | Edge cases in bootstrap sampling and empty empirical data fallbacks. |
| `app/components/face_detector.py` | 150 | 39 | 38 | 5 | 71% | 21% | **71%** | Unhandled detection errors and multi-face branching in real pipelines. |
| `app/components/image_processor.py` | 91 | 3 | 18 | 1 | 96% | 69% | **96%** | HEIC branch unreachable (missing `pillow-heif`). |
| `app/components/celebrity_matcher.py` | 196 | 61 | 58 | 3 | 68% | 31% | **70%** | DeepFace runtime init fallback branches and exception handling. |
| `app/components/overlay_generator.py` | 53 | 2 | 6 | 1 | 95% | 24% | **95%** | Guide line bounding limits. |
| `app/main.py` | 231 | 38 | 54 | 13 | 0% | 81% | **81%** | Celebrity match `None` branch, request timeout handling, error serializations. |
| `app/middleware/timeout_middleware.py` | 38 | 38 | 6 | 0 | 0% | 0% | **0%** | Middleware logic bypassed or never exercised during ASGI calls. |
| `app/models/internal.py` | 115 | 1 | 44 | 1 | 99% | 64% | **99%** | Full model validation covered. |
| `app/models/requests.py` | 29 | 0 | 8 | 0 | 100% | 35% | **100%** | Upload form validation. |
| `app/models/responses.py` | 75 | 0 | 0 | 0 | 100% | 100% | **100%** | Serialization models. |
| `app/utils/similarity.py` | 21 | 2 | 12 | 2 | 88% | 9% | **88%** | Zero-norm division branch. |
| **TOTAL** | **1552** | **237** | **310** | **31** | **71%** | **46%** | **84%** | **Combined line+branch coverage: 84%** |

---

## 2. Test Inventory & Paradigm Classification

Every test file was evaluated across three core testing paradigms:
1. **Real photographs through real MediaPipe:** Test reads a real photographic image file and runs real un-mocked MediaPipe detection.
2. **Synthetic landmarks:** Test uses hand-crafted 468 landmark coordinate arrays or synthetic facial dimension dictionaries.
3. **Mocks:** Test patches components, mocks MediaPipe/DeepFace, or exercises mocked API endpoints.

### File-by-File Classification (30 Files, 764 Tests)

| Test File | Package | Collected Tests | Category | Dominant Technique |
| :--- | :--- | :--- | :--- | :--- |
| `test_asymmetric_features.py` | Unit | 21 | Synthetic landmarks | Handcrafted asymmetric landmark coordinates |
| `test_extract_2d_landmarks.py` | Unit | 16 | Synthetic landmarks | Synthetic 3D-to-2D projection vectors |
| `test_facial_dimensions.py` | Unit | 13 | Synthetic landmarks | Handcrafted geometric coordinate dictionaries |
| `test_golden_ratio_guides.py` | Unit | 12 | Synthetic landmarks | Synthetic landmark coordinates |
| `test_golden_ratio_scorer.py` | Unit | 51 | Synthetic landmarks | Handcrafted symmetric face landmarks (no real images) |
| `test_overall_score_calculation.py` | Unit | 13 | Synthetic landmarks | Synthetic feature score arrays |
| `test_overlay_generator.py` | Unit | 26 | Synthetic landmarks | Synthetic 3D landmark arrays |
| `test_overlay_generator_integration.py` | Unit | 10 | Synthetic landmarks | Synthetic random 3D landmarks (`np.random.uniform`) |
| `test_response_models.py` | Unit | 44 | Synthetic landmarks | Synthetic landmark responses |
| `test_symmetry_classification.py` | Unit | 36 | Synthetic landmarks | Hand-crafted landmark dictionaries for 6 shapes |
| `test_celebrity_confidence_threshold.py` | Unit | 27 | Mocks | Mocked embeddings and similarity thresholds |
| `test_celebrity_matcher.py` | Unit | 56 | Mocks | Mocked DeepFace and mocked pickle dictionaries |
| `test_cosine_similarity.py` | Unit | 30 | Mocks | Synthetic float vectors |
| `test_face_detector.py` | Unit | 29 | Mocks | MediaPipe `FaceLandmarker.create_from_options` mocked via `@patch` |
| `test_image_processor.py` | Unit | 39 | Mocks | In-memory synthetic PIL images, EXIF tags |
| `test_image_resize.py` | Unit | 12 | Mocks | In-memory PIL images |
| `test_infrastructure.py` | Unit | 20 | Mocks | Mocked directory assertions |
| `test_internal_models.py` | Unit | 49 | Mocks | Mock model instantiations |
| `test_percentile_engine.py` | Unit | 50 | Mocks | Synthetic float scores against theoretical CDF |
| `test_rank_celebrities.py` | Unit | 15 | Mocks | Mock celebrity dictionary |
| `test_request_models.py` | Unit | 13 | Mocks | Mock request schemas |
| `test_resize_standalone.py` | Unit | 10 | Mocks | Synthetic PIL image buffers |
| `test_analyze_endpoint_validation.py` | Integration | 27 | Mocks | TestClient with mocked file uploads |
| `test_api_endpoint_integration.py` | Integration | 19 | Mocks | Mocked detector and mocked matcher |
| `test_error_handling_10_6.py` | Integration | 38 | Mocks | Mocked exception triggers |
| `test_fastapi_setup.py` | Integration | 42 | Mocks | TestClient routing checks |
| `test_mediapipe_initialization.py` | Integration | 3 | Mocks | Model file existence and empty object instantiation only |
| `test_orchestration_10_4.py` | Integration | 13 | Mocks | TestClient with solid-color synthetic boxes |
| `test_privacy_memory_cleanup_10_7.py`| Integration | 11 | Mocks | Mocked processing pipeline |
| `test_timeout_middleware.py` | Integration | 19 | Mocks | Mock async handlers |

### Distribution Summary

| Paradigm | File Count | File Share | Test Count | Test Share |
| :--- | :--- | :--- | :--- | :--- |
| **Real photographs through real MediaPipe** | **0** | **0.0%** | **0** | **0.0%** |
| **Synthetic landmarks** | 10 | 33.3% | 242 | 31.7% |
| **Mocks** | 20 | 66.7% | 522 | 68.3% |
| **Total** | **30** | **100.0%** | **764** | **100.0%** |

> [!CRITICAL]
> **Zero Real-Image Validation:** Across all 764 tests, **not a single test** passes a real photograph of a human face through the actual MediaPipe detector. Landmark indices cannot be verified by this test suite; if an index points to the wrong anatomical feature (e.g., [FM-BUG-007](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/BUGS.md#FM-BUG-007)), the entire test suite still passes.

---

## 3. Tautology Check (15 Sampled Tests)

Fifteen tests were sampled across the scorer, classifier, percentile, and matcher modules to evaluate whether assertions represent independent oracles or tautological self-affirmations.

| # | Test Identifier | Module | Assertion Pattern | Tautology Verdict | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `test_measure_face_length_to_width` | Scorer | `assert 0.5 < ratio < 3.0` | **Tautological** | Extremely loose bounding check. Any random positive number passes; does not verify exact geometric calculation. |
| 2 | `test_measure_face_to_jaw_width` | Scorer | `assert 0.5 < ratio < 3.0` | **Tautological** | Arbitrary bounds. Would pass even if jaw width indices were completely wrong. |
| 3 | `test_measure_eye_spacing_to_width` | Scorer | `assert 0.5 < ratio < 3.0` | **Tautological** | Identical copy-paste bound check. |
| 4 | `test_measure_vertical_thirds` | Scorer | `assert 0.5 < ratio < 3.0` | **Tautological** | Loose bounding check rather than mathematical ratio verification. |
| 5 | `test_classify_oval_shape` | Classifier | `assert result == SymmetryType.OVAL` | **Independent** | Fixture landmark points hand-crafted to yield 1.25 ratio + rounded jaw; verifies enum mapping. |
| 6 | `test_classify_round_shape` | Classifier | `assert result == SymmetryType.ROUND` | **Independent** | Landmark fixture explicitly creates 250px length $\times$ 250px width (1.0 ratio). |
| 7 | `test_classify_rectangle_shape` | Classifier | `assert result == SymmetryType.RECTANGLE` | **Independent** | Landmark fixture creates 400px length $\times$ 160px width (2.5 ratio). |
| 8 | `test_round_not_square` | Classifier | `assert result == SymmetryType.ROUND` | **Independent** | Verifies boundary distinction between round and square based on jaw curvature. |
| 9 | `test_initialize_with_default_parameters` | Percentile | `assert engine.mean == 70.0; assert engine.std == 10.0` | **Tautological** | Directly asserts default constructor parameters without testing distributional validity. |
| 10 | `test_score_to_percentile_uses_cdf` | Percentile | `assert 45 <= percentile <= 55` | **Independent** | Uses properties of normal median ($\mu=70 \implies 50\text{th percentile}$). |
| 11 | `test_percentile_bounds_are_enforced` | Percentile | `assert 0 <= percentile_low <= 100` | **Tautological** | Verifies basic clamping, but ignores mathematical correctness of extreme CDF tails. |
| 12 | `test_score_clamping_before_conversion` | Percentile | `assert percentile_clamped_low == percentile_zero` | **Independent** | Independent invariance check verifying that inputs $<0$ clamp identically to 0. |
| 13 | `test_embeddings_are_512_dimensional` | Matcher | `assert embedding.shape == (512,)` | **Tautological** | Asserts property of mock fixture dictionary generated by the test itself. |
| 14 | `test_rank_celebrities_uses_cosine_similarity` | Matcher | `assert len(rankings) == len(matcher.celebrity_embeddings)` | **Tautological** | Only checks list length; does not verify cosine similarity formula or ordering. |
| 15 | `test_identical_embedding_high_similarity` | Matcher | `assert best_score > 0.99` | **Independent** | Independent mathematical property of cosine similarity ($\cos(\mathbf{v}, \mathbf{v}) = 1.0$). |

**Tautology Summary:** 8 of 15 sampled tests (53.3%) are tautological or use loose bounding assertions (`0.5 < ratio < 3.0`) that mask underlying implementation bugs.

---

## 4. Mutation Testing (18 Mutants in Throwaway Worktree)

All 18 mutations (15 from `TESTING_PLAN.md` plus 3 requested additions) were executed in the throwaway worktree `..\facemetric_mut` using the `qa/.venv` interpreter. Baseline failures ([FM-BUG-002](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/BUGS.md#FM-BUG-002) and [FM-BUG-003](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/BUGS.md#FM-BUG-003)) were excluded from mutant kill determinations.

### Mutation Execution Log

| # | Mutation Description | Target File | Mutation Applied | Verdict | Killing Tests / Survival Analysis |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **01** | scorer: PHI 1.618 $\to$ 1.6 | `golden_ratio_scorer.py` | `PHI = 1.6` | **KILLED** | Killed by 12 tests (`test_score_feature_perfect_ratio`, etc.). |
| **02** | scorer: remove `abs(...)` | `golden_ratio_scorer.py` | `score = 100 * (1 - deviation / PHI)` | **KILLED** | Killed by 7 tests (`test_score_feature_half_ratio`, etc.). |
| **03** | scorer: remove 0–100 clamp | `golden_ratio_scorer.py` | Bypass `max(0, min(100, score))` | **KILLED** | Killed by 30 tests (`test_score_feature_bounds_lower`, etc.). |
| **04** | scorer: swap numerator/denominator | `golden_ratio_scorer.py` | Ratio 1 `width / length` | **KILLED** | Killed by 2 tests (`test_measure_face_length_to_width`, etc.). |
| **05** | scorer: landmark index 10 $\to$ 11 | `golden_ratio_scorer.py` | `FACE_TOP = 11` | **KILLED** | Killed by 3 tests (`test_measure_face_length_to_width`, etc.). |
| **06** | scorer: overall mean $\to$ median | `golden_ratio_scorer.py` | `overall_score = np.median(...)` | **KILLED** | Killed by 28 tests (`test_score_all_returns_features`, etc.). |
| **07** | scorer: drop one ratio from average | `golden_ratio_scorer.py` | Average first 7 ratios only | **KILLED** | Killed by 28 tests (`test_score_all_returns_features`, etc.). |
| **08** | percentile: mean 70 $\to$ 65 | `percentile_engine.py` | `mean: float = 65.0` | **KILLED** | Killed by 5 tests (`test_initialize_with_default_parameters`, etc.). |
| **09** | percentile: SD 10 $\to$ 5 | `percentile_engine.py` | `std: float = 5.0` | **KILLED** | Killed by 2 tests (`test_initialize_with_default_parameters`, etc.). |
| **10** | percentile: reverse CDF | `percentile_engine.py` | `(1.0 - cdf) * 100` | **KILLED** | Killed by 14 tests (`test_percentile_increases_with_score`, etc.). |
| **11** | classifier: ratio threshold 1.6 $\to$ 1.76 | `symmetry_classifier.py` | `length_width_ratio > 1.76` | **SURVIVED** | **SURVIVED.** Test fixtures only use ratio = 2.5 (`RECTANGLE`) or 1.25 (`OVAL`). No tests test boundary values between 1.60 and 1.76. |
| **12** | matcher: `>=` $\to$ `>` at threshold | `celebrity_matcher.py` | `best_score > threshold` | **KILLED** | Killed by 9 tests (`test_get_best_match_exactly_at_threshold`, etc.). |
| **13** | matcher: threshold 0.55 $\to$ 0.60 | `celebrity_matcher.py` | `threshold: float = 0.60` | **KILLED** | Killed by 3 tests (`test_default_threshold_0_55`, etc.). |
| **14** | image processor: ignore EXIF orientation | `image_processor.py` | Comment out `_apply_exif_rotation` | **SURVIVED** | **SURVIVED.** `TestEXIFRotation` only asserts `result.array is not None` and `width > 0`; never asserts rotated pixels/dimensions! |
| **15** | detector: skip multiple faces check | `face_detector.py` | Bypass `len(faces) > 1` check | **KILLED** | Killed by 2 tests (`test_multiple_faces_detected`, etc.). |
| **16** | scorer: bilateral split 0.10 $\to$ 0.20 | `golden_ratio_scorer.py` | `ASYMMETRY_THRESHOLD = 0.20` | **KILLED** | Killed by 3 tests (`test_asymmetry_above_threshold`, etc.). |
| **17** | matcher: shift 0.50 $\to$ 0.40 | `celebrity_matcher.py` | `c1 = max(0.60, min(0.96, 0.40 + sim))` | **SURVIVED** | **SURVIVED.** Existing matcher tests mock scores or assert ordering only; no test asserts exact output confidence calibration. |
| **18** | main.py: return empty celebrity object | `main.py` | Return empty object instead of `None` | **KILLED** | Killed by 11 tests (`test_sequential_image_and_face_detection`, etc.). |

### Module Survival Rates

| Module | Mutants Tested | Killed | Survived | Survival Rate | Missing Test Identified |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `golden_ratio_scorer` | 8 | 8 | 0 | **0.0%** | None in evaluated mutants. |
| `percentile_engine` | 3 | 3 | 0 | **0.0%** | None in evaluated mutants. |
| `symmetry_classifier` | 1 | 0 | 1 | **100.0%** | Boundary test for rectangle classification at ratio $1.65$. |
| `image_processor` | 1 | 0 | 1 | **100.0%** | EXIF rotation verification asserting pixel coordinates or aspect ratio swap. |
| `celebrity_matcher` | 3 | 2 | 1 | **33.3%** | Unmocked calibration test asserting exact confidence formula output. |
| `face_detector` | 1 | 1 | 0 | **0.0%** | None in evaluated mutants. |
| `main` | 1 | 1 | 0 | **0.0%** | None in evaluated mutants. |
| **OVERALL** | **18** | **15** | **3** | **16.7%** | **3 critical missing tests identified.** |

### Missing Tests for Surviving Mutants (To Be Implemented in Later Phases)
1. **Mutant 11 (`symmetry_classifier`):** *Add boundary test asserting that a face with length/width ratio 1.65 and sharp jaw angle is classified as `RECTANGLE`, not `OVAL`.*
2. **Mutant 14 (`image_processor`):** *Add test asserting that a $400\times 600$ image with EXIF orientation 6 (90° CW) emerges with dimensions $600\times 400$ and corner marker transposed.*
3. **Mutant 17 (`celebrity_matcher`):** *Add test asserting that a cosine similarity of $0.35$ produces a calibrated confidence score of exactly $0.85$ (or un-clamped value).*

---

## 5. Test Smells & Fragility Analysis

An automated AST scan of all 764 tests identified 30 hollow tests that cannot fail, 5 clock-dependent tests, and 24 path-dependent tests.

### 5.1. Tests with Zero Assertions (7 Tests)
These tests execute code but never assert results; if the target code returns corrupt data or fails silently, the test passes:
1. `tests/integration/test_analyze_endpoint_validation.py::test_both_file_and_base64_provided`
2. `tests/integration/test_fastapi_setup.py::test_startup_event_registered`
3. `tests/integration/test_timeout_middleware.py::test_timeout_response_returns_http_408`
4. `tests/integration/test_timeout_middleware.py::test_timeout_response_has_status_field`
5. `tests/unit/test_extract_2d_landmarks.py::test_no_mutation_of_input`
6. `tests/unit/test_extract_2d_landmarks.py::test_edge_case_all_zeros`
7. `tests/unit/test_extract_2d_landmarks.py::test_edge_case_all_ones`

### 5.2. Trivial Assertions that Cannot Fail (23 Tests)
These tests assert `assert response.status_code in [200, 400]` or `assert response.status_code in [200, 400, 408]`. Because the server returns 400 on invalid images and 200 on valid images, the test **structurally cannot fail** regardless of server behavior:
- 13 tests in `tests/integration/test_analyze_endpoint_validation.py` (e.g. `test_multipart_file_upload_only`, `test_base64_only`, `test_valid_extensions`, etc.)
- 5 tests in `tests/integration/test_api_endpoint_integration.py` (`test_multipart_jpeg_accepted`, `test_base64_upload_accepted`, etc.)
- 5 tests in `tests/integration/test_orchestration_10_4.py` (`test_sequential_image_and_face_detection`, `test_parallel_symmetry_scoring_overlay`, `test_optional_celebrity_matching_progressive_degradation`, etc.)

### 5.3. Skipped or xfail Tests
- **0 tests** are marked with `@pytest.mark.skip` or `@pytest.mark.xfail`.

### 5.4. Clock-Dependent Tests (5 Tests)
These tests rely on system clocks (`time.time()` or elapsed metadata) and can flake under high system load:
- `tests/integration/test_orchestration_10_4.py::test_processing_completes_within_30_seconds`
- `tests/unit/test_internal_models.py::test_valid_processing_metadata`
- `tests/unit/test_internal_models.py::test_add_component_time`
- `tests/unit/test_internal_models.py::test_get_total_time`
- `tests/unit/test_internal_models.py::test_get_component_summary`

### 5.5. File-Path & Disk Dependent Tests (24 Tests)
These tests assert specific local file existence or fail if run on systems with differing file structures:
- `tests/integration/test_mediapipe_initialization.py` (3 tests asserting `face_landmarker.task` file path)
- `tests/integration/test_privacy_memory_cleanup_10_7.py::test_no_tmp_files_created`
- `tests/unit/test_celebrity_matcher.py` (6 tests asserting local pickle database layout)
- `tests/unit/test_infrastructure.py` (7 tests asserting directory paths `backend/app/components`, etc.)
- `tests/unit/test_percentile_engine.py` (7 tests attempting to read local CSV empirical distribution files)

---

## Phase 0.5 Conclusion

The existing 764-test suite provides **84% line/branch coverage**, but is fundamentally brittle:
1. **Zero real photographs:** Not a single test verifies landmark indices against actual human faces.
2. **53% tautological sampling:** Over half of sampled algorithmic tests use bounds so loose (`0.5 < ratio < 3.0`) that wrong math passes.
3. **30 hollow tests:** 7 tests contain no assertions, and 23 integration tests assert `status_code in [200, 400]`, making failure impossible.
4. **Surviving mutants:** Identified 3 blind spots in EXIF rotation, symmetry classification boundaries, and celebrity confidence calibration.
