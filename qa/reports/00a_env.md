# FaceMetric — Environment & Dependency Investigation (Phase 0a)

**Date:** 2026-10-14  
**Author:** Antigravity (QA Agent)  
**Target Environment:** Isolated virtual environment at `qa/.venv` (Python 3.12.10)  
**Reference Specification:** [TESTING_PLAN.md](file:///c:/Users/FayazHussain/Desktop/FaceMatric/TESTING_PLAN.md), [FaceMatric-prd.md](file:///c:/Users/FayazHussain/Desktop/FaceMatric/FaceMatric-prd.md)

---

## 1. Inventory of `backend/requirements.txt` for Key Packages

Inspection of [backend/requirements.txt](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/requirements.txt):

| Package | Status in `backend/requirements.txt` | Line Number | Specified Version | Notes |
|---|---|---|---|---|
| **`deepface`** | **Listed** | Line 18 | `deepface==0.0.79` | Pinned with `tensorflow==2.18.0` (Line 19). |
| **`httpx`** | **Listed** | Line 29 | `httpx==0.25.2` | Listed under testing dependencies, but was never installed in `backend/venv`. |
| **`pillow-heif`** | **NOT LISTED** | N/A | *Absent* | Absent from `requirements.txt` entirely, causing runtime `UnidentifiedImageError` for HEIC files. |

---

## 2. Fresh Installation in `qa/.venv`

A clean virtual environment was created at `qa/.venv` using Python 3.12.10 (`python -m venv qa/.venv`) without modifying `backend/venv`.

**Installation Command:**
```powershell
.\qa\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

### Installation Outcome
- **Result:** **Success (Exit Code 0)**.
- **Packages Installed (76 total):**
  - `deepface-0.0.79`
  - `tensorflow-2.18.0` & `tensorflow-intel-2.18.0`
  - `keras-3.15.1`
  - `fastapi-0.104.1`
  - `uvicorn-0.24.0`
  - `httpx-0.25.2` & `httpcore-1.0.9`
  - `mediapipe-0.10.35`
  - `opencv-python-4.8.1.78`
  - `Pillow-10.1.0`
  - `numpy-1.26.2`
  - `scipy-1.11.4`
  - `pytest-7.4.3`
  - `pytest-asyncio-0.21.1`
  - `pytest-cov-4.1.0`
  - *(plus supporting wheels including retina-face, mtcnn, gdown, etc.)*
- **What Failed:** Nothing failed to install from `backend/requirements.txt`.
- **Finding:** `httpx` and `deepface` installed without issue in `qa/.venv`. Their absence in `backend/venv` was due to an incomplete environment setup in `backend/venv`, rather than an unresolvable dependency conflict.

---

## 3. DeepFace Import & `Facenet512` Verification

Verification was executed inside `qa/.venv`:
```powershell
.\qa\.venv\Scripts\python.exe -c "from deepface.basemodels import Facenet512; print('Import SUCCESS')"
```

### Results
- **Module Import:** `import deepface` succeeded.
- **Class Import:** `from deepface.basemodels import Facenet512` **SUCCEEDED**.
- **Module Location:** `deepface.basemodels.Facenet512` is the exact module path where `Facenet512` resides in `deepface==0.0.79`.
- **Available Base Models:**
  `['ArcFace', 'DeepID', 'DlibResNet', 'DlibWrapper', 'Facenet', 'Facenet512', 'FbDeepFace', 'OpenFace', 'SFace', 'VGGFace']`.
- **Verdict:** The import path used in `CelebrityMatcher._initialize_facenet()` ([celebrity_matcher.py line 231](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/celebrity_matcher.py#L231)) is **correct** for `deepface==0.0.79`. In `backend/venv`, it only failed because `deepface` was never installed.

---

## 4. Integration Test Suite Execution in `qa/.venv`

With `httpx==0.25.2` installed in `qa/.venv`, Starlette's `TestClient` can now be initialized. The complete integration test suite was executed:

**Command:**
```powershell
cd c:\Users\FayazHussain\Desktop\FaceMatric\backend
..\qa\.venv\Scripts\pytest.exe tests/integration -v
```

### Execution Summary
- **Total Tests:** 172
- **Passed:** 171
- **Failed:** 1
- **Errors during collection:** 0
- **Duration:** 86.33 seconds
- **Warnings:** 553

### Per-File Breakdown

| Test File | Total | Passed | Failed | Errors |
|---|---|---|---|---|
| `tests/integration/test_analyze_endpoint_validation.py` | 27 | 27 | 0 | 0 |
| `tests/integration/test_api_endpoint_integration.py` | 19 | 18 | 1 | 0 |
| `tests/integration/test_error_handling_10_6.py` | 38 | 38 | 0 | 0 |
| `tests/integration/test_fastapi_setup.py` | 42 | 42 | 0 | 0 |
| `tests/integration/test_mediapipe_initialization.py` | 3 | 3 | 0 | 0 |
| `tests/integration/test_orchestration_10_4.py` | 13 | 13 | 0 | 0 |
| `tests/integration/test_privacy_memory_cleanup_10_7.py` | 11 | 11 | 0 | 0 |
| `tests/integration/test_timeout_middleware.py` | 19 | 19 | 0 | 0 |
| **Total** | **172** | **171** | **1** | **0** |

### Failure Analysis
- **Failing Test:** `test_api_endpoint_integration.py::TestResponseStructure::test_successful_response_structure`
- **Error:**
  ```python
  TypeError: 'Mock' object is not iterable
  File "backend/app/main.py", line 581, in _process_analysis
      for item in celebrity_match_internal.top_matches:
  ```
- **Root Cause:** Commit `851c3cb` introduced the top-5 celebrity look-alike list and iterated over `celebrity_match_internal.top_matches`. The test mocked `celebrity_match_internal` with `MagicMock()`. Because `.top_matches` was not defined on the mock, accessing it returned a child `MagicMock` instance, which evaluates to truthy (`bool(mock) == True`) but raises `TypeError` when iterated over. This caused `/api/analyze` to return HTTP 500 instead of HTTP 200.

---

## 5. Origin and Provenance of `celebrity_embeddings.pkl`

Detailed tracking using `git log` (`git log -p backend/scripts/preprocess_celebrity_database.py` and commit history):

1. **Originating Script:**
   - [backend/scripts/preprocess_celebrity_database.py](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/scripts/preprocess_celebrity_database.py).
   - Originally created in commit `a6e357b` (Oct 11 01:14), then completely rewritten in commit `851c3cb` (Oct 13 03:13:29).
2. **DeepFace & Model Version:**
   - In commit `a6e357b`, it called `DeepFace.represent(..., model_name="Facenet")` (128-d).
   - In commit `851c3cb`, it was rewritten to load `from deepface.basemodels import Facenet512` directly, generating 512-dimensional embeddings via `model.predict(batch)`.
   - The deepface version pinned in `requirements.txt` during both commits is `deepface==0.0.79`.
3. **Images Sampled Per Identity:**
   - Lines 84–89 of [preprocess_celebrity_database.py](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/scripts/preprocess_celebrity_database.py#L84-L89):
     ```python
     def extract_face_batch(image_paths, max_images=10):
         selected_paths = image_paths[:max_images]
     ```
   - Exactly **up to 10 images** per celebrity identity (`max_images=10`), sorted alphabetically.
4. **Dataset Sources & Selection:**
   - Lines 63–64: `selected_western = western_folders[:105]`, `selected_bollywood = bollywood_folders[:100]`.
   - Exactly 205 identities selected: 105 from `pins_*` folders (Western) and 100 non-pins folders (Bollywood).
   - Preprocessing applied: OpenCV `cv2.imread()`, resize to `(160, 160)` without facial landmark alignment, normalize to $[-1, 1]$.
   - Each individual embedding was L2-normalized, averaged across the $\le 10$ photos, and the average was L2-normalized again.
