# FaceMetric — Phase 0 Reconnaissance Report

**Date:** 2026-10-14  
**Author:** Antigravity (QA Agent)  
**Execution Context:** Windows PowerShell, Python 3.12.10 virtual environment (`backend/venv`)  
**Scope:** Phase 0 Recon only. No production or test code modified.

---

## 1. Architectural Map of Core Components

| Component / Functionality | Implementation Files & Key Lines | Description |
|---|---|---|
| **Landmark Extraction** | [backend/app/components/face_detector.py](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/face_detector.py#L43-L156)<br>[backend/app/utils/landmark_extraction.py](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/utils/landmark_extraction.py#L11-L98)<br>[backend/app/models/internal.py](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/models/internal.py#L84-L150) | Uses MediaPipe `FaceLandmarker` with asset `backend/data/face_landmarker.task`. Extracts 468 landmarks, denormalizes $x, y$ coordinates to image pixel dimensions, and stores in `Landmarks.points` $(468, 3)$. |
| **Ratio Definitions & Scoring** | [backend/app/components/golden_ratio_scorer.py](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/golden_ratio_scorer.py#L31-L268)<br>[backend/app/components/golden_ratio_scorer.py](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/golden_ratio_scorer.py#L269-L577) | Implements Euclidean distance calculations, 8 facial proportion measurements, bilateral asymmetry handling (>10% deviation triggers split), and formula-based feature/overall scoring. |
| **Face-Shape Classification** | [backend/app/components/symmetry_classifier.py](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/symmetry_classifier.py#L22-L230) | Classifies face geometry into one of six `SymmetryType` categories: `oval`, `rectangle`, `round`, `square`, `heart`, `diamond`. Informational only; not factored into scores. |
| **Percentile Ranking** | [backend/app/components/percentile_engine.py](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/percentile_engine.py#L18-L82) | Computes percentile rank via `scipy.stats.norm(loc=70, scale=10).cdf(score) * 100`. |
| **Celebrity Matching & Look-Alikes** | [backend/app/components/celebrity_matcher.py](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/celebrity_matcher.py#L21-L542)<br>[backend/app/utils/similarity.py](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/utils/similarity.py#L10-L50)<br>[backend/scripts/preprocess_celebrity_database.py](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/scripts/preprocess_celebrity_database.py#L1-L198)<br>[backend/data/celebrity_embeddings.pkl](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/data/celebrity_embeddings.pkl) | Precomputed pickle database contains 205 celebrity embeddings (105 Western, 100 Bollywood). Matcher ranks by cosine similarity and scales top 5 look-alikes. *(Runtime embedding generation requires DeepFace, which is not installed in venv).* |
| **API Route `/api/analyze` & Orchestration** | [backend/app/main.py](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/main.py#L121-L276)<br>[backend/app/main.py](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/main.py#L417-L644)<br>[backend/app/middleware/timeout_middleware.py](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/middleware/timeout_middleware.py#L1-L80) | FastAPI route accepting multipart file or base64 JSON string. Orchestrates image processing, face detection, concurrent scoring/classification/overlay generation, and progressive degradation for celebrity look-alikes. |
| **Overlay Generation & Drawing** | [backend/app/components/overlay_generator.py](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/overlay_generator.py#L17-L100)<br>[frontend/src/js/results.js](file:///c:/Users/FayazHussain/Desktop/FaceMatric/frontend/src/js/results.js#L241-L380) | Backend calculates vertical midline, 4-point horizontal thirds, and phi spiral center. Frontend HTML5 canvas renders landmarks and guide overlays. |
| **Frontend Upload, Analyzing, & Results Screens** | [frontend/index.html](file:///c:/Users/FayazHussain/Desktop/FaceMatric/frontend/index.html)<br>[frontend/src/js/upload.js](file:///c:/Users/FayazHussain/Desktop/FaceMatric/frontend/src/js/upload.js)<br>[frontend/analyzing.html](file:///c:/Users/FayazHussain/Desktop/FaceMatric/frontend/analyzing.html)<br>[frontend/src/js/analyzing.js](file:///c:/Users/FayazHussain/Desktop/FaceMatric/frontend/src/js/analyzing.js)<br>[frontend/results.html](file:///c:/Users/FayazHussain/Desktop/FaceMatric/frontend/results.html)<br>[frontend/src/js/results.js](file:///c:/Users/FayazHussain/Desktop/FaceMatric/frontend/src/js/results.js) | Static client pages managing client-side image loading, multipart submission to `POST /api/analyze`, session persistence, and visualization of score circles, percentile badges, feature breakdowns, and celebrity look-alikes. |

---

## 2. Ratio Specification & Implementation Table

Distance metric: strictly **2D Euclidean distance** (`_euclidean_distance(p1, p2) = sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)`), ignoring the $z$-coordinate ([golden_ratio_scorer.py line 269](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/golden_ratio_scorer.py#L269-L272)).  
Coordinate space: **Pixel coordinates** (denormalized by `FaceDetector.detect_faces` via `x * width, y * height` before ratio calculation; [face_detector.py lines 110-117](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/face_detector.py#L110-L117)).  
Division by zero handling: Each measurement method returns `0.0` if denominator is $\le 0$. A ratio of `0.0` scores `0.0` via `max(0.0, min(100.0, 100 * (1 - abs(0.0 - 1.618)/1.618)))`.

> [!WARNING]
> In [backend/app/components/golden_ratio_scorer.py](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/golden_ratio_scorer.py), methods `_measure_cheekbone_to_jaw_bilateral` (lines 276 vs 486), `_measure_eye_spacing_to_width_bilateral` (lines 315 vs 520), and `_measure_eye_to_brow_distance_bilateral` (lines 352 vs 549) are declared twice in the class body. In Python, the latter definitions (lines 486–577) override the former. The table below documents the active, executing definitions.

| Ratio # | Feature Label | Landmark Endpoints Used | Numerator Distance | Denominator Distance | Left / Right Combination & Asymmetry Handling |
|---|---|---|---|---|---|
| **1** | `Face Length ÷ Face Width` | Top: 10<br>Bottom: 152<br>Left: 234<br>Right: 454 | `dist(10, 152)` | `dist(234, 454)` | Single pair (symmetric). |
| **2** | `Cheekbone to Jaw Width` | Left Cheek: 234<br>Right Cheek: 454<br>Left Jaw: 172<br>Right Jaw: 397 | Left: `dist(234, face_center)`<br>Right: `dist(face_center, 454)`<br>*(face_center = mid(234, 454))* | Left: `dist(172, jaw_center)`<br>Right: `dist(jaw_center, 397)`<br>*(jaw_center = mid(172, 397))* | Measured bilaterally. If `\|ratio_L - ratio_R\| / avg > 0.10` (10%), split into two breakdown entries: `... (Left)` and `... (Right)`. Otherwise, combined via arithmetic mean `(ratio_L + ratio_R) / 2`. |
| **3** | `Eye Spacing ÷ Eye Width` | Left Inner: 133<br>Left Outer: 33<br>Right Inner: 362<br>Right Outer: 263 | Left & Right share Numerator:<br>`dist(133, 362)` (interocular distance) | Left: `dist(33, 133)` (left eye width)<br>Right: `dist(263, 362)` (right eye width) | Measured bilaterally. If difference $> 10\%$, split into `... (Left)` and `... (Right)`. Otherwise, combined via arithmetic mean `(ratio_L + ratio_R) / 2`. |
| **4** | `Nose Width ÷ Mouth Width` | Nose Left: 219<br>Nose Right: 439<br>Mouth Left: 61<br>Mouth Right: 291 | `dist(219, 439)` | `dist(61, 291)` | Single pair (symmetric). |
| **5** | `Mouth Width ÷ Face Width` | Mouth Left: 61<br>Mouth Right: 291<br>Face Left: 234<br>Face Right: 454 | `dist(61, 291)` | `dist(234, 454)` | Single pair (symmetric). |
| **6** | `Nose-Chin ÷ Hairline-Nose (Vertical Thirds)` | Hairline proxy: 10<br>Nose Bridge: 6<br>Chin: 152 | `dist(6, 152)` (nose bridge to chin) | `dist(10, 6)` (hairline proxy to nose bridge) | Single pair (symmetric). **Note:** Uses landmark 10 (top of forehead mesh) as proxy for anatomical hairline. |
| **7** | `Eye-Eyebrow Distance ÷ Eye Height` | Left Eye Top: 159<br>Left Brow Bottom: 70<br>Left Eye Bottom: 145 | Left: `dist(159, 70)` | Left: `dist(159, 145)` | **Degenerate Bilateral:** Line 574 sets `ratio_right = ratio_left`. Right eye is not measured. Difference is always 0, so it always returns a single averaged feature based solely on the left eye. |
| **8** | `Chin Width ÷ Nose Width` | Jaw Left: 172<br>Jaw Right: 397<br>Nose Left: 219<br>Nose Right: 439 | `dist(172, 397)` (commented as chin width) | `dist(219, 439)` | Single pair (symmetric). |

---

## 3. System Constants & Parameters

All operational constants identified in codebase configuration and component implementations:

| Constant / Parameter | Value in Code | Source File & Location | PRD / Plan Alignment |
|---|---|---|---|
| **Golden Ratio ($\phi$)** | `1.618` (float literal) | [config.py:41](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/config.py#L41)<br>[golden_ratio_scorer.py:51](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/golden_ratio_scorer.py#L51) | Fixed float literal 1.618, not exact mathematical constant $(1+\sqrt{5})/2 \approx 1.6180339887$. |
| **Celebrity Confidence Threshold** | `0.35` in settings<br>`0.55` in matcher method<br>`0.55`–`0.96` in top-5 scaler | [config.py:44](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/config.py#L44)<br>[celebrity_matcher.py:288](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/celebrity_matcher.py#L288)<br>[celebrity_matcher.py:507-517](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/celebrity_matcher.py#L507-L517) | **Conflict**: PRD specifies `0.55`. `config.py` specifies `0.35`. Method default is `0.55`. Active request path bypasses threshold check and scales cosine similarity via `0.50 + sim`, clamping top match to $[0.60, 0.96]$ and lower matches to $\ge 0.55$. |
| **FaceNet Model Variant** | `Facenet512` (512-d) | [celebrity_matcher.py:231](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/celebrity_matcher.py#L231)<br>[preprocess_celebrity_database.py:35](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/scripts/preprocess_celebrity_database.py#L35) | Stored embeddings and model import use 512-dimensional `Facenet512`, despite docstring in `celebrity_matcher.py:25` stating 128-d. |
| **Face Detector Backend** | MediaPipe FaceLandmarker | [face_detector.py:46-57](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/face_detector.py#L46-L57) | Asset: `./data/face_landmarker.task`. `max_faces=1`, `min_detection_confidence=0.5`. Landmark confidence threshold: `0.6`. |
| **Image Resize Dimensions** | Min 300px, Max 640px | [config.py:36-37](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/config.py#L36-L37)<br>[image_processor.py:235-251](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/image_processor.py#L235-L251) | Resizes down if max dimension $> 640\text{px}$ using Lanczos filter. Minimum allowed upload dimension is $300\text{px}$. |
| **JPEG Compression Quality** | `85` | [config.py:38](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/config.py#L38)<br>[image_processor.py:270](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/image_processor.py#L270) | Images converted to JPEG quality 85 with buffer optimize=True. |
| **Processing Timeout** | `30` seconds | [config.py:17](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/config.py#L17)<br>[main.py:196](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/main.py#L196) | Hard timeout enforced via `asyncio.wait_for(..., timeout=30)` and `TimeoutMiddleware`. Returns HTTP 408. |
| **Percentile Distribution** | $\mathcal{N}(\mu=70.0, \sigma=10.0)$ | [percentile_engine.py:26-36](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/percentile_engine.py#L26-L36) | Bootstrap normal distribution initialized at engine startup. No empirical CSV dataset is loaded. |

---

## 4. Claims Reconciliation (AI Status Files vs. Code & Git Log)

Reconciliation of root status files against git history (`git log --stat`) and actual codebase state:

| Status File / Document | Key Claims in File | Actual State in Repository | Verdict | Reproducing Evidence |
|---|---|---|---|---|
| **QUICK_TEST_GUIDE.txt** & **TESTING_GUIDE.txt** (Oct 11 01:28, commit `48799ca`) | "399+ unit tests passing"<br>"API Gateway (/api/analyze endpoint)", "Celebrity database loading", "Celebrity matching" CANNOT be tested yet / blocked by API throttle. | API gateway `/api/analyze` was implemented in `d54c749`. Celebrity database was loaded in `d54c749`. Test count grew to 592. | **STALE** | Commits `d54c749`, `851c3cb`; [main.py:121](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/main.py#L121); [backend/data/celebrity_embeddings.pkl](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/data/celebrity_embeddings.pkl). |
| **PROGRESS_STATUS.txt** (Oct 11 01:24, commit `f3e91b1`) | "36 of 75 Tasks Complete (48%)"<br>"Task 8.3 & 8.5 IN-PROGRESS - API throttle" | Later commits completed Task 8, Task 9, Task 10, Task 11. | **STALE** | Commit `d54c749` marked Task 11 complete; `.kiro/specs/.../tasks.md`. |
| **PROJECT_STATUS.md** & **TEST_EXECUTION_SUMMARY.md** & **QUICK_TEST_REFERENCE.md** (Oct 11 01:47, commits `bd0f1ed`, `9669dfe`, `f2edfa5`) | "487/556 tests passing (87.1%)"<br>"18 blocked (celebrity confidence thresholding)" | Unit tests currently total 592 (591 pass, 1 fails). Integration tests were added later. | **STALE** | Git history; current unit test execution results (592 tests). |
| **BACKEND_IMPLEMENTATION_COMPLETE.md** & **BACKEND_SUMMARY.md** (Oct 11 01:14, commit `a6e357b`) | "Backend fully implemented with all 8 core components, FastAPI gateway... ready for testing and deployment." | Architecture exists, but status claims about test numbers are outdated, and integration tests fail to collect without `httpx`. | **PARTIALLY STALE** | Architecture matches; test suite claims are obsolete. |
| **TASK_6_4_COMPLETION_REPORT.md** (Oct 11 01:14, commit `a6e357b`) | "Asymmetric Feature Handling... 21 tests, 100% passing... 330 passed". | Bilateral asymmetric feature logic is active in `golden_ratio_scorer.py`. Test count is stale. | **CURRENT (Logic) / STALE (Counts)** | [golden_ratio_scorer.py:220-268](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/golden_ratio_scorer.py#L220-L268); `tests/unit/test_asymmetric_features.py`. |
| **INTEGRATION_COMPLETE.txt** (Oct 12 01:47, commit `d54c749`) | "Frontend-Backend Integration Complete - Frontend now sends images to backend API and displays real analysis results" | Verified: `analyzing.js` submits `FormData` to `POST /api/analyze`, stores JSON in `sessionStorage`, and `results.js` parses and renders it. | **CURRENT** | [frontend/src/js/analyzing.js:90](file:///c:/Users/FayazHussain/Desktop/FaceMatric/frontend/src/js/analyzing.js#L90); [frontend/src/js/results.js:43-115](file:///c:/Users/FayazHussain/Desktop/FaceMatric/frontend/src/js/results.js#L43-L115). |
| **OCCLUSION_FIX_SUMMARY.md** (Oct 12 01:47, commit `d54c749`) | Occlusion thresholds relaxed: `z_std_threshold = 0.12`, `eye aspect [0.08, 0.6]`, `z_depth_diff = 0.30`, `mouth z_std = 0.15`, `mouth_hw = 0.04/0.02`, `chin z_std = 0.16`. | Exact constants verified in `face_detector.py`. | **CURRENT** | [face_detector.py:333, 367, 392, 435, 463, 480](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/face_detector.py#L333). |
| **QUALITY_CHECK_FIX_SUMMARY.md** (Oct 12 01:47, commit `d54c749`) | "Changed landmark_confidence_threshold from 0.6 to 0.3 in config.py line 49"<br>"Floor 0.5". | `config.py:49` has `landmark_confidence_threshold: float = 0.6` (reverted in commit `d54c749`). Floor in `face_detector.py:273` is `0.65`. | **FALSE / STALE** | [config.py:49](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/config.py#L49); commit message `d54c749` explicitly states: *"restore landmark_confidence_threshold configuration"*. |
| **FIX_SUMMARY.md** (Oct 12 01:47, commit `d54c749`) | Fixed "No analysis results found" via `dataUrlToBlob` in `analyzing.js`. | Verified: `dataUrlToBlob` method exists and is invoked. | **CURRENT** | [frontend/src/js/analyzing.js:283-294](file:///c:/Users/FayazHussain/Desktop/FaceMatric/frontend/src/js/analyzing.js#L283-L294). |
| **Commit `d54c749` Log Message** | "feat: complete backend stabilization with 100% test pass rate (764/764)" | False. Integration tests cannot even collect without `httpx`, and current unit test suite has 1 failing test out of 592. | **FALSE** | Commit `d54c749` message; pytest collection failure on integration tests; pytest unit failure. |

---

## 5. Status of Repo Components Right Now

| Component | Status | Location & Details |
|---|---|---|
| **`/api/analyze` Route** | **EXISTS & ACTIVE** | Implemented at [backend/app/main.py:121](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/main.py#L121). Handles multipart and base64 payloads with validation, in-memory processing, and timeout protection. |
| **Celebrity Embeddings File** | **EXISTS & LOADED** | Located at [backend/data/celebrity_embeddings.pkl](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/data/celebrity_embeddings.pkl) (454,072 bytes). Contains 205 identities (105 Western, 100 Bollywood), all with shape `(512,)`. |
| **Celebrity Matcher** | **EXISTS (BLOCKED AT RUNTIME)** | Class `CelebrityMatcher` exists at [celebrity_matcher.py](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/celebrity_matcher.py). Database loads on startup (`is_available = True`). **However**, `_initialize_facenet()` attempts to import `deepface.basemodels.Facenet512`, which fails because `deepface` is not installed in the virtual environment. It degrades gracefully and returns `None` for all user requests. |
| **Frontend Results Screen** | **EXISTS & ACTIVE** | Implemented at [frontend/results.html](file:///c:/Users/FayazHussain/Desktop/FaceMatric/frontend/results.html) and [frontend/src/js/results.js](file:///c:/Users/FayazHussain/Desktop/FaceMatric/frontend/src/js/results.js). Renders score dials, percentile badges, feature bars, canvas overlay, and top look-alikes. |

---

## 6. Test Suite Execution Baseline

Execution command run in PowerShell:
```powershell
cd c:\Users\FayazHussain\Desktop\FaceMatric\backend
.\venv\Scripts\pytest.exe tests/unit -v
```

### Execution Metrics
- **Python Version:** 3.12.10 (tags/v3.12.10:04e6c98, Feb 10 2025, 14:48:47) [MSC v.1942 64 bit (AMD64)]
- **pytest Version:** 7.4.3
- **pluggy Version:** 1.6.0
- **Active Plugins:** `anyio-4.15.1`, `asyncio-0.21.1`, `cov-4.1.0`
- **Total Tests Collected:** 592
- **Passed:** 591
- **Failed:** 1
- **Skipped / xfailed:** 0
- **Warnings:** 253
- **Execution Duration:** 34.94 seconds

### Analysis of the Single Failure
- **Test:** `tests/unit/test_rank_celebrities.py::TestRankCelebritiesIntegration::test_rank_celebrities_finds_best_match`
- **Failure Detail:**
  ```python
  tests\unit\test_rank_celebrities.py:317: in test_rank_celebrities_finds_best_match
      assert match.confidence == round(best_score, 2)
  E   AssertionError: assert 0.96 == 1.0
  E    +  where 0.96 = CelebrityMatchInternal(name='Celebrity A', confidence=0.96, ...).confidence
  E    +  and   1.0 = round(1.0, 2)
  ```
- **Root Cause:** Commit `851c3cb` introduced a pseudo-calibrated lookup in `CelebrityMatcher._find_best_match` ([celebrity_matcher.py:507](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/celebrity_matcher.py#L507)):
  `c1 = max(0.60, min(0.96, 0.50 + float(best_similarity)))`.
  When synthetic unit test input has `best_similarity = 1.0`, `0.50 + 1.0 = 1.50`, which is clamped to `0.96`. The test asserted that `match.confidence == round(best_score, 2) == 1.0`. The commit modified production logic without running or updating the test.

### Key Installed Package Versions
- `fastapi`: 0.143.0
- `mediapipe`: 1.1.0
- `numpy`: 1.26.2
- `opencv-python`: 4.8.1.78
- `pillow`: 10.1.0
- `pydantic`: 2.14.0
- `scipy`: 1.11.4
- `starlette`: 1.7.0
- `uvicorn`: 0.24.0
- `onnxruntime`: 1.31.0
- **Missing Packages:** `deepface` (not installed), `httpx` (not installed, breaks integration tests), `pillow-heif` (not installed, breaks HEIC).

---

## 7. Hypotheses H1–H14 Verdicts

| ID | Hypothesis Summary | Verdict | Evidence (Files & Lines) |
|---|---|---|---|
| **H1** | PRD API example scores (1.52 → 82.3, 1.70 → 74.1) cannot be produced by formula $100 \times (1 - \|r - \phi\| / \phi)$ (gives 93.94, 94.93). | **CONFIRMED** | Mathematical proof: For $r=1.52$, $100 \times (1 - 0.098/1.618) = 93.943$. For $r=1.70$, $100 \times (1 - 0.082/1.618) = 94.932$. The PRD numbers 82.3 and 74.1 are mathematically impossible under the formula. Code in [golden_ratio_scorer.py:209](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/golden_ratio_scorer.py#L209) implements the exact formula and yields 93.94. [responses.py:295](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/models/responses.py#L295) amended the example to 93.9. |
| **H2** | MediaPipe landmarks are normalized; computing distances on raw $x, y$ distorts ratios on non-square images. | **CONFIRMED (Risk) / REFUTED (Implementation)** | [face_detector.py:112-113](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/face_detector.py#L112-L113) explicitly denormalizes landmarks to pixel coordinates (`lm.x * width`, `lm.y * height`). Distances are computed in pixel space, preventing aspect ratio distortion in production. |
| **H3** | Ratio 6 uses a "hairline", but MediaPipe has no hairline landmark. Code uses landmark 10 (upper forehead proxy). Status files still describe it as hairline. | **CONFIRMED** | [golden_ratio_scorer.py:55, 451](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/golden_ratio_scorer.py#L55): `FACE_TOP = 10`, `hairline = points[self.FACE_TOP]`. Landmark 10 is the apex of the forehead mesh contour, not the anatomical hairline. Feature label and documentation remain "Hairline-Nose". |
| **H4** | Ratio 4 (nose $\div$ mouth) and Ratio 5 (mouth $\div$ face width) are structurally far from $\phi$ in real faces, capping overall scores well below 100. | **CONFIRMED** | Human anatomy dictates normal nose/mouth ratio $\approx 0.65\text{--}0.75$ (score $\approx 40\text{--}46$) and mouth/face ratio $\approx 0.35\text{--}0.40$ (score $\approx 22\text{--}25$). With equal weights across 8 ratios, pulling ~16 points off the mean makes overall scores $>85$ structurally impossible for real human faces. |
| **H5** | Feature scores are not clamped, so ratios $> 2\phi$ give negative scores. | **REFUTED** | [golden_ratio_scorer.py:210](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/golden_ratio_scorer.py#L210) explicitly clamps: `score = max(0.0, min(100.0, score))`. Line 178 also clamps overall score. |
| **H6** | Reference celebrity embeddings built offline with different preprocessing than production request path. | **CONFIRMED** | Offline script [preprocess_celebrity_database.py:95-100](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/scripts/preprocess_celebrity_database.py#L95-L100) reads raw BGR images via OpenCV, converts to RGB, forces resize to $(160, 160)$, and feeds to FaceNet. Production path ([main.py:459](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/main.py#L459), [image_processor.py:98-113](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/image_processor.py#L98-L113)) applies EXIF rotation, aspect-ratio-preserving downsampling to 640px, JPEG 85 compression, and MediaPipe face detection. Parity is broken. |
| **H7** | Per-celebrity embeddings averaged without L2-normalizing first; 0.55 threshold uncalibrated; "confidence" is just cosine similarity; embeddings are 512-d Facenet512. | **PARTIALLY REFUTED, PARTIALLY CONFIRMED** | **Refuted**: [preprocess_celebrity_database.py:128-136](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/scripts/preprocess_celebrity_database.py#L128-L136) L2-normalizes each raw embedding before averaging, then L2-normalizes the average.<br>**Confirmed**: Uses 512-d `Facenet512`. Threshold 0.55 is uncalibrated. Furthermore, [celebrity_matcher.py:507](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/celebrity_matcher.py#L507) replaced raw cosine similarity with an arbitrary ad-hoc linear shift `0.50 + sim` clamped to $[0.60, 0.96]$. |
| **H8** | Occlusion detection has no real mechanism: Face Mesh returns full mesh on masked/sunglassed faces. | **CONFIRMED (Fundamental) / PARTIALLY REFUTED (Heuristic)** | MediaPipe returns 468 landmarks regardless of occlusion. However, [face_detector.py:277-491](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/face_detector.py#L277-L491) implements heuristic checks on $z$-depth variance and 2D bounding ratios. These are uncalibrated heuristics prone to false positives/negatives. |
| **H9** | Celebrity result shape is inconsistent: PRD Step 5 says `null` below threshold, API example returns object with `name: null`. | **CONFIRMED** | PRD Step 5 (line 168) specifies `null`. PRD Contract (line 236) specifies `{ name: null, confidence: 0.42, message: ... }`. Code in [main.py:564](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/main.py#L564) returns `null` if matcher returns `None`, but returns `{ name: null, ... }` if matcher returns an object below threshold. Frontend [results.js:216](file:///c:/Users/FayazHussain/Desktop/FaceMatric/frontend/src/js/results.js#L216) handles `{ message: ... }`, but ignores `null` (leaving static placeholder). |
| **H10** | EXIF orientation is ignored and/or HEIC is unsupported despite claims. | **EXIF REFUTED; HEIC CONFIRMED** | EXIF orientation is handled via `ImageOps.exif_transpose(image)` in [image_processor.py:145](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/image_processor.py#L145). HEIC is claimed in `SUPPORTED_FORMATS`, but `pillow-heif` is not installed; Pillow throws `UnidentifiedImageError` on HEIC files. |
| **H11** | The 0.55 comparison is made on a rounded value, so the UI can show "0.55" next to "No match". | **REFUTED (Backend) / CONFIRMED (Frontend)** | Backend [celebrity_matcher.py:335](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/celebrity_matcher.py#L335) compares unrounded `best_score >= threshold`. However, frontend [results.js:170](file:///c:/Users/FayazHussain/Desktop/FaceMatric/frontend/src/js/results.js#L170) formats confidence via `Math.round(confidence * 100) + '%'`. A confidence of 0.546 rounds to 55%. |
| **H12** | Percentile engine uses assumed Normal(70, 10), not real data, so "score higher than X% of people" is unsupported. | **CONFIRMED** | [percentile_engine.py:26-36](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/percentile_engine.py#L26-L36) hardcodes `mean=70.0, std=10.0`. No empirical data or calibration CSV is loaded. Claims of comparison against "population distribution" are theoretical assertions. |
| **H13** | Status guides are stale: claim API gateway, celebrity matching and frontend are not implemented, but later files show they were completed. | **CONFIRMED** | Git history confirms guides were created at commit `48799ca` (Oct 11 01:28). Subsequent commits (`14637f8`, `931551b`, `d54c749`, `851c3cb`) implemented `/api/analyze`, database loading, timeout middleware, and frontend integration without updating the status files. |
| **H14** | Existing unit tests are component-level with synthetic/mocked inputs, allowing mathematical conflicts (H1) or bad percentile assumptions (H12) to pass undetected. | **CONFIRMED** | Review of `backend/tests/unit/` shows tests use synthetic landmark arrays (`np.zeros((468, 3))`) or mock data. No unit test verifies end-to-end geometry on real photographs through MediaPipe. Tautological assertions in `test_rank_celebrities.py` verify re-implemented mock math rather than ground truth. |

---

## 8. Missing Elements & Plan Assumptions

Elements assumed by `TESTING_PLAN.md` that do not exist in the repository:

1. **Missing Python Dependencies in Virtual Environment:**
   - `httpx` is missing: Starlette's `TestClient` cannot be imported, causing all 8 integration test suites (`tests/integration/test_*.py`, ~172 tests) to fail during collection.
   - `deepface` is missing: `CelebrityMatcher._initialize_facenet()` and `preprocess_celebrity_database.py` cannot load `Facenet512` at runtime.
   - `pillow-heif` is missing: Required for HEIC image support claimed in the PRD.
2. **Missing QA Datasets (`qa/data/`):**
   - `qa/data/frontal_set/` (20–30 clean frontal photos) does not exist.
   - `qa/data/same_person/` (volunteer multi-photo sets) does not exist.
   - `qa/data/unknown/` (non-celebrity face set) does not exist.
   - `qa/data/faceshape_labels.csv` (ground-truth shape labels) does not exist.
3. **Missing Empirical Percentile Reference:**
   - No historical score distribution or empirical calibration dataset exists; the system is running solely on theoretical bootstrap $\mathcal{N}(70, 10)$.
4. **Discrepancy in Disclaimer Text:**
   - PRD Section 15 mandates: *"This tool analyzes geometric facial proportions for educational purposes. Scores do not reflect beauty, worth, or health. Results are influenced by photo angle and quality."*
   - Active UI in [results.html:496](file:///c:/Users/FayazHussain/Desktop/FaceMatric/frontend/results.html#L496) displays: *"This analysis is based on facial geometry and proportions, not skin color, ethnicity, or beauty standards."*

---

## Conclusion & Next Steps

Phase 0 Recon is complete. The system architecture, ratio mathematical definitions, constants, status claims, and test suite baselines have been fully audited and verified against code and git evidence.

**Current Test Suite Status:**
- `tests/unit/`: 591 passed, 1 failed (total 592 tests in 34.94s).
- `tests/integration/`: Blocked during collection due to missing `httpx` dependency.

Stopping execution here per instructions. Awaiting confirmation to proceed to Phase 0.5 (Audit of Existing Test Suite).
