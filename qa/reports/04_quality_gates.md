# FaceMetric — Phase 4 Quality Gates and Error Handling Report

**Date:** 2026-10-15  
**Environment:** `qa/.venv` (Python 3.12.10, FastAPI 0.104.1, Starlette 0.27.0, Pydantic 2.5.2, Pillow 10.1.0, pillow-heif 1.8.1, MediaPipe 0.10.8)  
**Test Suite:** [qa/tests/test_phase4_quality_gates.py](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/tests/test_phase4_quality_gates.py)  
**Execution Results:** 13 Passed, 4 Expected Strict Failures (`XFAIL(strict=True)`) in 37.56s  
**Summary Artifact:** [qa/reports/04_quality_gates_summary.json](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/04_quality_gates_summary.json)  

---

## 1. Executive Summary & Quality Gate Parameters

Phase 4 evaluated the input validation, format decoding, quality thresholds, occlusion filters, and error contract compliance of the FaceMetric backend API route (`/api/analyze`).

### 1.1 Quality Gate Thresholds: Proposal vs Active Code

Per Phase 4 instructions, the tolerances and thresholds proposed in Phase 3 were evaluated against the active constants in the codebase:

| Parameter | Proposed Threshold (Phase 3) | Active Code Constant | Code Location | Status / Discrepancy |
| :--- | :--- | :--- | :--- | :--- |
| **Minimum Dimension** | $250 \times 250$ px | $300 \times 300$ px | [backend/app/components/image_processor.py:28](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/image_processor.py#L28) | Code is more restrictive (rejects $64\times 64$ Bollywood and $250-299$ px crops; see `FM-BUG-024`) |
| **Max Aspect Ratio** | 3.0 : 1 | 3.0 : 1 | [backend/app/components/image_processor.py:29](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/image_processor.py#L29) | Matches proposal |
| **Max File Size** | 10 MB | 10 MB (`10 * 1024 * 1024`) | [backend/app/components/image_processor.py:27](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/image_processor.py#L27) | Matches proposal |
| **Landmark Confidence** | $\ge 0.60$ | 0.60 (`landmark_confidence_threshold`) | [backend/app/config.py:37](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/config.py#L37) | **Dead Code**: Floored at 0.65 in [face_detector.py:273](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/face_detector.py#L273) |
| **Quality Score Floor** | None (unclamped) | $0.65$ (`quality_score = max(0.65, ...)`) | [backend/app/components/face_detector.py:273](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/face_detector.py#L273) | Prevents landmark confidence rejection |
| **Face Quality Threshold** | $\ge 0.50$ | 0.60 (`quality_threshold`) | [backend/app/config.py:38](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/config.py#L38) | Contradicts `QUALITY_CHECK_FIX_SUMMARY.md` (`0.30`) |
| **Pose Yaw Gate** | $[-10.0^\circ, +10.0^\circ]$ | None (unconstrained) | N/A | No yaw check implemented in API gate |
| **Pose Pitch Gate** | $[-12.0^\circ, +12.0^\circ]$ | None (unconstrained) | N/A | No pitch check implemented in API gate |
| **Pose Roll Gate** | $[-8.0^\circ, +8.0^\circ]$ | None (unconstrained) | N/A | No roll check implemented in API gate |

---

## 2. Clean Photo Acceptance on Frontal Benchmark

All 30 frontal images in `qa/data/frontal_set/` (10 original frontal portraits + 20 top-up celebrity portraits with estimated yaw $\le 5^\circ$) were submitted to `/api/analyze`.

### 2.1 Acceptance Results
- **Total Images Submitted:** 30
- **Total Accepted (HTTP 200):** 30 (100.0%)
- **Total Rejected (HTTP 400):** 0 (0.0%)
- **Status:** **PASS**

### 2.2 Analysis of Quality Gate Code: The 0.65 Hardcoded Floor
In `QUALITY_CHECK_FIX_SUMMARY.md`, the documentation claimed that the face quality threshold was reduced to `0.3` because high thresholds were aggressively rejecting valid frontal portraits. However, inspection of the active codebase revealed the real mechanism ensuring 100% acceptance:

1. In `config.py:37-38`, `landmark_confidence_threshold = 0.6` and `quality_threshold = 0.6`.
2. In `face_detector.py:273`:
   ```python
   # Line 273 in FaceDetector._extract_detection_result:
   quality_score = max(0.65, min(1.0, quality_score))
   ```
3. In `face_detector.py:136`:
   ```python
   # Line 136 in FaceDetector.detect_face:
   if avg_confidence < settings.landmark_confidence_threshold:
       logger.warning(...)
       return None
   ```
Because line 273 artificially clamps `quality_score` to a floor of `0.65`, `avg_confidence < 0.60` can **never evaluate to True** for any detected face. The landmark confidence gate is mathematically unreachable dead code.

---

## 3. Occlusion Filter Evaluation

### 3.1 Dataset Directory Status
Per `AGENTS.md` and `TESTING_PLAN.md`, the designated occlusion directories under `qa/data/occlusion/` were audited:

| Directory | File Count | Status | Reason |
| :--- | :---: | :---: | :--- |
| `qa/data/occlusion/sunglasses/` | 0 | **BLOCKED** | Directory empty; no curated real sunglasses dataset present. |
| `qa/data/occlusion/glasses/` | 0 | **BLOCKED** | Directory empty; no curated clear glasses dataset present. |
| `qa/data/occlusion/mask/` | 0 | **BLOCKED** | Directory empty; no curated surgical mask dataset present. |
| `qa/data/occlusion/hand/` | 0 | **BLOCKED** | Directory empty; no curated hand occlusion dataset present. |
| `qa/data/occlusion/hair/` | 0 | **BLOCKED** | Directory empty; no curated hair occlusion dataset present. |

### 3.2 Spot-Check Evaluation on Clear Glasses and Synthetic Occlusions
To evaluate whether the active occlusion detection heuristics (`_detect_glasses`, `_detect_mask`) function as specified, spot-check probes were executed:

```
+-----------------------------------------------------------------------------+
| OCCLUSION SPOT CHECKS                                                       |
+------------------------------+------------+------------------+--------------+
| Probe Case                   | HTTP Code  | API Result       | Gate Outcome |
+------------------------------+------------+------------------+--------------+
| Clear Glasses (Bill Gates)   | HTTP 200   | Success (Score)  | Valid Accept |
| Synthetic Dark Sunglasses    | HTTP 200   | Success (Score)  | FALSE ACCEPT |
| Synthetic Surgical Mask      | HTTP 200   | Success (Score)  | FALSE ACCEPT |
+------------------------------+------------+------------------+--------------+
```

#### Detailed Findings on Occlusion Failure (`FM-BUG-017`):
1. **Clear Glasses (Bill Gates 1):** Accepted with HTTP 200. In `OCCLUSION_FIX_SUMMARY.md`, the eye z-depth variance threshold was relaxed from `0.05` to `0.12`. For Bill Gates, the eye region z-depth variance was $\approx 0.08 < 0.12$, preventing false rejection on standard prescription glasses.
2. **Synthetic Sunglasses (Opaque Black Overlay over Eye Box):**
   - **Expected:** Rejected with HTTP 400 `OCCLUSION_DETECTED` ("Eyes appear to be covered by sunglasses or dark glasses").
   - **Observed:** Accepted with HTTP 200 (`FM-BUG-017`). MediaPipe Face Mesh infers landmark depth from the surrounding visible facial perimeter, predicting a smooth 3D depth plane across the occluded eye region. Measured eye z-variance was $0.091 < 0.12$, so `_detect_glasses` failed to trigger.
3. **Synthetic Mask (Opaque Polygon Covering Nose, Mouth, and Chin):**
   - **Expected:** Rejected with HTTP 400 `OCCLUSION_DETECTED` ("Lower face appear to be covered by a mask").
   - **Observed:** Accepted with HTTP 200 (`FM-BUG-017`). The landmark depth variance across the lower face remained $0.114 < 0.15$. `_detect_mask` failed to trigger.
4. **Conclusion:** The heuristic z-depth variance method fails to detect opaque obstructions because MediaPipe's regression model regularizes occluded landmarks into plausible 3D facial planes. The occlusion rejection filter exhibits a **100% False Acceptance Rate** on these obstructions.

---

## 4. Input Handling and Edge Case Matrix

Seventeen distinct input scenarios were evaluated against `/api/analyze`.

```
+----------------------------------------------------------------------------------------------------+
| INPUT HANDLING EVALUATION MATRIX                                                                   |
+----+----------------------------+-----------------------+--------+------------------+--------------+
| #  | Scenario                   | Input Payload         | Status | Error Code       | Outcome      |
+----+----------------------------+-----------------------+--------+------------------+--------------+
| 1  | RGBA PNG                   | multipart/form-data   | 200    | None (Success)   | PASS         |
| 2  | Grayscale JPEG             | multipart/form-data   | 200    | None (Success)   | PASS         |
| 3  | WebP Format                | multipart/form-data   | 200    | None (Success)   | PASS         |
| 4  | BMP Format                 | multipart/form-data   | 200    | None (Success)   | PASS         |
| 5  | Animated GIF (Multi-frame) | multipart/form-data   | 200    | None (Success)   | PASS         |
| 6  | EXIF Rotated (Orient=6)    | multipart/form-data   | 200    | None (Success)   | PASS         |
| 7  | HEIC Format (iPhone)       | multipart/form-data   | 400    | INVALID_IMAGE    | XFAIL (BUG)  |
| 8  | Corrupt Image Bytes        | multipart/form-data   | 400    | INVALID_IMAGE    | PASS (Logic) |
| 9  | Empty File (0 bytes)       | multipart/form-data   | 400    | INVALID_IMAGE    | PASS         |
| 10 | File Exceeding 10MB (20MB) | multipart/form-data   | 400    | FILE_TOO_LARGE   | PASS         |
| 11 | Dimensions < 300x300       | 250x250 pixels        | 400    | POOR_QUALITY     | PASS         |
| 12 | Aspect Ratio > 3:1 (4:1)   | 1200x300 pixels       | 400    | POOR_QUALITY     | PASS         |
| 13 | Dual Inputs (File + Base64)| Both fields provided  | 400    | INVALID_REQUEST  | PASS         |
| 14 | Missing Inputs (Neither)   | No body/query params  | 400    | NO_IMAGE         | PASS         |
| 15 | Data-URI Prefix in Query   | "data:image/jpeg;..." | 400    | INVALID_IMAGE    | PASS         |
| 16 | Raw Base64 in URL Query    | Query string parameter| 400    | INVALID_IMAGE    | PASS         |
| 17 | Base64 in POST Body        | Form-data / JSON body | 400    | NO_IMAGE         | XFAIL (BUG)  |
+----+----------------------------+-----------------------+--------+------------------+--------------+
```

### 4.1 Detailed Format and Input Observations

1. **Alpha Channel & Color Space Support (RGBA, Grayscale, WebP, BMP):**
   `ImageProcessor._convert_to_rgb` properly converts 4-channel RGBA and single-channel 'L' grayscale images to standard 3-channel RGB before NumPy array conversion and MediaPipe ingestion.
2. **Animated GIF Handling:**
   `ImageProcessor.load_image` successfully seeks frame 0 (`image.seek(0)`) and converts it to RGB without throwing multi-frame sequence errors.
3. **EXIF Orientation Auto-Correction:**
   Pillow's `ImageOps.exif_transpose` correctly normalizes orientation tag 6 (90° clockwise phone orientation), returning a correctly oriented portrait that MediaPipe detects without rotation penalty.
4. **HEIC Decoding Failure (`FM-BUG-006`):**
   Although PRD Section 8 and `ImageProcessor.SUPPORTED_FORMATS` advertise HEIC support, attempting to decode a genuine iPhone HEIC file fails with `PIL.UnidentifiedImageError: cannot identify image file`. The API returns HTTP 400 `INVALID_IMAGE`.
5. **Base64 in POST Request Body Broken (`FM-BUG-028`):**
   In `main.py:128-129`:
   ```python
   async def analyze_face(
       file: Optional[UploadFile] = File(None),
       image_base64: Optional[str] = None
   )
   ```
   Because `image_base64` is declared without `Form(None)` or `Body(None)`, FastAPI assigns it to query parameter parsing. When a client sends a standard HTTP POST request with base64 data in the form body or JSON payload, `image_base64` evaluates to `None`, resulting in an immediate `NO_IMAGE` rejection. Attempting to supply an actual image via URL query string triggers `httpx.InvalidURL: URL too long` (HTTP 414). Base64 upload via POST is non-functional.

---

## 5. API Error Contract Verification

PRD Section 9 and `backend/app/schemas/responses.py` (lines 166–173) define the canonical error response model:

```python
class ErrorResponse(BaseModel):
    status: str = Field("error", description="Always 'error'")
    error_code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    retry: bool = Field(True, description="Whether the request can be retried")
```

### 5.1 Contract Breach Discovered: Dual-Schema & Nested `"detail"` (`FM-BUG-027`)

When auditing error payloads across all 400-series rejections, two completely conflicting response structures were returned depending on where the error originated in the pipeline:

```
Contract Inconsistency Flow:

1. Early Validation (Outside _process_analysis):
   ValidationError raised -> @app.exception_handler(ValidationError) catches it
   Response Payload:
   {
       "status": "error",
       "error_code": "NO_IMAGE",
       "message": "No image provided...",
       "retry": true
   }  <-- Matches responses.py ErrorResponse (Root Level)

2. Late Validation (Inside _process_analysis):
   ValidationError raised -> main.py:233 catches it:
   raise HTTPException(status_code=400, detail=e.to_dict())
   Response Payload:
   {
       "detail": {
           "status": "error",
           "error_code": "INVALID_IMAGE",
           "message": "Corrupt or unsupported file format...",
           "retry": true
       }
   }  <-- VIOLATES CONTRACT: Nested under 'detail' key!
```

### 5.2 Error Contract Summary by Error Code

| Error Code | Trigger Condition | Status Code | Nested under `"detail"`? | Complies with `ErrorResponse`? |
| :--- | :--- | :---: | :---: | :---: |
| `NO_IMAGE` | Neither file nor query provided | 400 | No (Root level) | **YES** |
| `INVALID_REQUEST` | Both file and base64 provided | 400 | No (Root level) | **YES** |
| `FILE_TOO_LARGE` | File size > 10MB | 400 | No (Root level) | **YES** |
| `INVALID_IMAGE` | Corrupt bytes, empty file, HEIC | 400 | **Yes** (`FM-BUG-027`) | **NO** |
| `POOR_QUALITY` | Under 300x300, aspect ratio > 3:1 | 400 | **Yes** (`FM-BUG-027`) | **NO** |
| `NO_FACE_DETECTED` | No face found in image | 400 | **Yes** (`FM-BUG-027`) | **NO** |
| `OCCLUSION_DETECTED`| Detected sunglasses / mask | 400 | **Yes** (`FM-BUG-027`) | **NO** |

**Impact:** Frontend client code written as `if (data.error_code === 'POOR_QUALITY')` will evaluate to `undefined` on all image processing and face detection errors, breaking user-facing error dialogs and retry prompts.

---

## 6. Bugs Logged & Updated in `qa/reports/BUGS.md`

During Phase 4, two new major defects were identified, verified, and logged in [qa/reports/BUGS.md](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/BUGS.md):

1. **`FM-BUG-027` (Severity S2): Error Response Schema Inconsistency (Nested "detail" vs Root-Level Contract)**
   - In `main.py:233`, re-raising `HTTPException(detail=e.to_dict())` causes FastAPI to nest the dictionary under `"detail"`, while `main.py:107` emits root-level JSON. Breaks API schema guarantees.
2. **`FM-BUG-028` (Severity S2): Base64 Image Input Broken for POST Request Bodies**
   - In `main.py:128-129`, `image_base64` is declared without `Form(None)` or `Body(None)`, treating it as a URL query parameter. Standard base64 payloads in POST bodies are ignored, throwing `NO_IMAGE`.

### Existing Bugs Verified in Phase 4:
- **`FM-BUG-006`:** HEIC decoding fails (`UnidentifiedImageError`). Verified via strict xfail in `test_heic_acceptance`.
- **`FM-BUG-017`:** Occlusion heuristics fail to detect sunglasses and masks. Verified via strict xfail in `test_sunglasses_rejected_with_occlusion_detected`.
- **`FM-BUG-019`:** `QUALITY_CHECK_FIX_SUMMARY.md` claimed quality threshold was lowered to 0.3; verified that `config.py` still has 0.6 and line 273 floors scores at 0.65.

---

## 7. Reproduction Instructions

To reproduce all findings and verify the automated Phase 4 test suite:

```powershell
# In PowerShell from project root, using qa/.venv:
& .\qa\.venv\Scripts\pytest.exe qa/tests/test_phase4_quality_gates.py -v
```

**Expected Test Output:**
```
================ 13 passed, 4 xfailed, 491 warnings in 37.56s =================
```
*(The 4 xfailed tests assert strict failures for `FM-BUG-006`, `FM-BUG-017`, `FM-BUG-027`, and `FM-BUG-028` without modifying any production code).*
