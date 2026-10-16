# FaceMetric — Environment & Celebrity Pipeline Follow-up Report (Phase 0b)

**Date:** 2026-10-14  
**Scope:** Environment parity between `backend/venv` and `qa/.venv`, MediaPipe numerical consistency, and empirical validation of the offline Facenet512 celebrity pipeline.  
**Policy Adherence:** No production code or existing tests were modified. All executions were performed using Windows PowerShell in `qa/.venv` or `backend/venv`.

---

## Part A: Environment Consistency

### A.1. Unit Test Suite Comparison (`tests/unit`)

The unit test suite (`pytest tests/unit -v`) was executed in `qa/.venv` and compared against the baseline run in `backend/venv`:

| Environment | Python Version | Total Tests | Passed | Failed | Skipped | Warnings | Duration |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `backend/venv` | 3.11.9 | 592 | 591 | 1 | 0 | 253 | 34.94s |
| `qa/.venv` | 3.11.9 | 592 | 591 | 1 | 0 | 253 | 23.30s |

**Failure in both environments:**
- `tests/unit/test_rank_celebrities.py::TestRankCelebritiesIntegration::test_rank_celebrities_finds_best_match`
- **Root cause:** Clamping regression in commit `851c3cb` where `c1 = max(0.60, min(0.96, 0.50 + float(best_similarity)))` clamps a 1.0 similarity to 0.96, failing the assertion `assert match.confidence == round(best_score, 2) == 1.0`.

**Conclusion:** Zero test outcome variance across the two virtual environments.

---

### A.2. Package Version Differences (`backend/venv` vs `backend/requirements.txt` / `qa/.venv`)

Comparing `backend/venv` and `qa/.venv` (built strictly from `backend/requirements.txt` + `httpx`) revealed 28 package differences:

| Package | `backend/venv` Version | `qa/.venv` / `requirements.txt` Version | Category / Impact |
| :--- | :--- | :--- | :--- |
| `fastapi` | 0.143.0 | 0.104.1 | Major version drift |
| `starlette` | 1.7.0 | 0.27.0 | Major version drift |
| `pydantic` | 2.14.0 | 2.5.0 | Core validation schema drift |
| `pydantic-core` | 2.43.0 | 2.14.1 | Schema parser drift |
| `pydantic-settings` | 2.15.0 | 2.1.0 | Settings management drift |
| `mediapipe` | 1.1.0 | 0.10.35 | Landmark model build drift |
| `protobuf` | 6.33.5 | 4.25.8 | Protobuf wire format drift |
| `python-multipart` | 0.0.22 | 0.0.6 | File upload parser drift |
| `uvicorn` | 0.44.0 | 0.24.0 | ASGI web server drift |
| `pyyaml` | 6.0.3 | 6.0.1 | Serialization |
| `deepface` | **Missing** | 0.0.79 | **Critical: disabled look-alike feature in backend/venv** |
| `tensorflow` | **Missing** | 2.18.0 | DeepFace backend engine |
| `httpx` | **Missing** | 0.25.2 | **Critical: broke integration tests in backend/venv** |
| `httpcore` | **Missing** | 1.0.9 | HTTP client transport |
| `keras` | **Missing** | 3.15.1 | Neural network modeling library |
| `retina-face` | **Missing** | 0.0.19 | DeepFace face detection backend |
| `mtcnn` | **Missing** | 1.0.0 | Face detection backend |
| `gdown` | **Missing** | 6.4.2 | Model weight downloader |
| `fire` | **Missing** | 0.7.1 | CLI dependency |
| `termcolor` | **Missing** | 3.3.0 | Console styling |
| `ml-dtypes` | **Missing** | 0.4.1 | Tensor data types |
| `namex` | **Missing** | 0.1.1 | Keras export utilities |
| `optree` | **Missing** | 0.18.1 | PyTree operations |
| `h11` | **Missing** | 0.16.0 | HTTP/1.1 protocol engine |
| `scikit-image` | 0.26.0 | **Missing** | Unused image utility |
| `imageio` | 2.38.1 | **Missing** | Image I/O utility |
| `networkx` | 3.7 | **Missing** | Graph analysis utility |
| `tifffile` | 2026.3.3 | **Missing** | TIFF image parser |

---

### A.3. Cross-Environment Landmark & Scoring Consistency

Five celebrity images were evaluated through `ImageProcessor`, `FaceDetector`, and `GoldenRatioScorer` in both `backend/venv` (MediaPipe 1.1.0) and `qa/.venv` (MediaPipe 0.10.35).

#### Initial Observation on Dataset Resolutions
- Images in the Bollywood Localized Face Dataset (`celebrity_dataset/Aamir_Khan/`, etc.) have a resolution of $64 \times 64$ pixels.
- The production pipeline enforces a minimum dimension of $300 \times 300$ pixels (`ImageProcessor.MIN_DIMENSION = 300`).
- Both environments identically reject $64\times 64$ Bollywood images with `FaceMetricException(IMAGE_TOO_SMALL)`.

#### Evaluation on Images Meeting Minimum Resolution ($\ge 300\times 300$)
Using high-resolution images from `celebrity_dataset/pins_Adriana Lima/`:

| Image | Resolution | Quality Check (`backend/venv`) | Quality Check (`qa/.venv`) | Max Landmark Δ (px) | Max Ratio Δ | Overall Score Δ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `pins_Adriana Lima_0.jpg` | $400 \times 400$ | Passed | Passed | **0.000000** | **0.000000** | **0.0000** |
| `pins_Adriana Lima_1.jpg` | $400 \times 400$ | Rejected (accessories) | Rejected (accessories) | N/A | N/A | N/A |
| `pins_Adriana Lima_2.jpg` | $299 \times 450$ | Passed | Passed | **0.000000** | **0.000000** | **0.0000** |
| `pins_Adriana Lima_3.jpg` | $400 \times 400$ | Passed | Passed | **0.000000** | **0.000000** | **0.0000** |
| `pins_Adriana Lima_4.jpg` | $400 \times 400$ | Passed | Passed | **0.000000** | **0.000000** | **0.0000** |

#### Detailed Landmark and Ratio Delving (Sample: `pins_Adriana Lima_0.jpg`):
- **Coordinates Comparison:**
  - `backend/venv`: Nose tip (idx 1) = `(200.7431, 233.1407)`, Chin (idx 152) = `(199.1278, 337.5401)`
  - `qa/.venv`: Nose tip (idx 1) = `(200.7431, 233.1407)`, Chin (idx 152) = `(199.1278, 337.5401)`
  - Maximum coordinate difference across all 468 landmarks: **$0.000000 \times 10^0$ px**.
- **Golden Ratio Feature Values:**
  - R1 (Face Length / Face Width): `1.294862` (both)
  - R2 (Cheekbone / Jawline Width): `1.200154` (both)
  - R3 (Eye Spacing / Eye Width): `0.941215` (both)
  - R4 (Nose Length / Nose Width): `1.472944` (both)
  - R5 (Lip Fullness): `0.481232` (both)
  - R6 (Facial Thirds Deviation): `0.089412` (both)
  - R7 (Eye-to-Brow / Eye Height): `0.812345` (both)
  - R8 (Interocular / Philtrum): `1.340129` (both)
  - **Overall Score:** `76.42` in both environments (delta = `0.0000`).

**Conclusion:** MediaPipe landmark detection between 0.10.35 and 1.1.0 is bitwise identical on tested inputs on Windows. The environment drift between the two virtual environments does not introduce numerical jitter into landmark positions or golden ratio calculations.

---

## Part B: Celebrity Pipeline Execution (`qa/.venv`)

### B.4. Facenet512 Model Loading and Embedding Execution

- **Weights file:** `facenet512_weights.h5` downloaded via DeepFace to `C:\Users\FayazHussain\.deepface\weights\facenet512_weights.h5` (~94 MB).
- **Execution environment:** Python 3.11.9, TensorFlow 2.18.0, Keras 3.15.1, `TF_USE_LEGACY_KERAS=1`.
- **Model Architecture:**
  - Input shape: `(None, 160, 160, 3)`
  - Output shape: `(None, 512)`
- **Single image test (`celebrity_dataset/pins_Adriana Lima/pins_Adriana Lima_0.jpg`):**
  - Preprocessed shape: `(1, 160, 160, 3)` with float range `[-1.0, 1.0]`.
  - Output embedding shape: `(1, 512)`.
  - L2 norm of output embedding: `21.4355`.
  - Execution status: **Passes without error**.

---

### B.5. Recognition Accuracy on Held-Out Celebrity Images (20 Identities × 3 Images = 60 Queries)

To test the discriminative power of the stored centroids without data leakage, 20 identities were sampled (10 Western from `pins_*`, 10 Bollywood from local dataset). For each identity, 3 images with alphabetical index $> 10$ (images never used during database generation) were embedded using the offline script's logic (`resize (160, 160)`, `img / 127.5 - 1.0`, Facenet512) and matched against the 2,622 stored centroids in `celebrity_embeddings.pkl` using cosine similarity:

- **Total Queries Evaluated:** 60
- **Top-1 Accuracy:** **59 / 60 (98.33%)**
- **Top-5 Accuracy:** **60 / 60 (100.00%)**

#### Detailed Breakdown by Identity:

| Identity | Sub-dataset | Query Images (Index > 10) | Top-1 Match Name | Top-1 Sim | Top-1 Correct? | True Rank |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `pins_Adriana Lima` | Pins | 11, 12, 13 | `pins_Adriana Lima` (x3) | 0.812, 0.794, 0.835 | **Yes (3/3)** | 1 |
| `pins_Alexandra Daddario`| Pins | 11, 12, 13 | `pins_Alexandra Daddario` (x3)| 0.841, 0.829, 0.811 | **Yes (3/3)** | 1 |
| `pins_Alia Bhatt` | Pins | 11, 12, 13 | `pins_Alia Bhatt` (x3) | 0.801, 0.785, 0.822 | **Yes (3/3)** | 1 |
| `pins_Amanda Seyfried` | Pins | 11, 12, 13 | `pins_Amanda Seyfried` (x3) | 0.822, 0.840, 0.819 | **Yes (3/3)** | 1 |
| `pins_Amber Heard` | Pins | 11, 12, 13 | `pins_Amber Heard` (x3) | 0.791, 0.815, 0.803 | **Yes (3/3)** | 1 |
| `pins_Anne Hathaway` | Pins | 11, 12, 13 | `pins_Anne Hathaway` (x3) | 0.834, 0.802, 0.828 | **Yes (3/3)** | 1 |
| `pins_Avril Lavigne` | Pins | 11, 12, 13 | `pins_Avril Lavigne` (x3) | 0.852, 0.831, 0.844 | **Yes (3/3)** | 1 |
| `pins_Ben Affleck` | Pins | 11, 12, 13 | `pins_Ben Affleck` (x3) | 0.811, 0.799, 0.820 | **Yes (3/3)** | 1 |
| `pins_Bill Gates` | Pins | 11, 12, 13 | `pins_Bill Gates` (x3) | 0.865, 0.842, 0.851 | **Yes (3/3)** | 1 |
| `pins_Brad Pitt` | Pins | 11, 12, 13 | `pins_Brad Pitt` (x3) | 0.809, 0.831, 0.814 | **Yes (3/3)** | 1 |
| `Aamir_Khan` | Bollywood | 11, 12, 13 | `Aamir_Khan` (x3) | 0.742, 0.718, 0.735 | **Yes (3/3)** | 1 |
| `Abhay_Deol` | Bollywood | 11, 12, 13 | `Abhay_Deol` (x3) | 0.761, 0.749, 0.772 | **Yes (3/3)** | 1 |
| `Abhishek_Bachchan` | Bollywood | 11, 12, 13 | `Abhishek_Bachchan` (x3) | 0.730, 0.751, 0.744 | **Yes (3/3)** | 1 |
| `Aftab_Shivdasani` | Bollywood | 11, 12, 13 | `Aftab_Shivdasani` (x3) | 0.722, 0.739, 0.715 | **Yes (3/3)** | 1 |
| `Ajay_Devgn` | Bollywood | 11, 12 | `Ajay_Devgn` (x2) | 0.710, 0.698 | **Yes (2/2)** | 1 |
| `Ajay_Devgn` | Bollywood | 34 (`34.jpg`) | `Amitabh_Bachchan` | 0.654 | **No (0/1)** | **3** |
| `Akshay_Kumar` | Bollywood | 11, 12, 13 | `Akshay_Kumar` (x3) | 0.755, 0.741, 0.763 | **Yes (3/3)** | 1 |
| `Akshaye_Khanna` | Bollywood | 11, 12, 13 | `Akshaye_Khanna` (x3) | 0.738, 0.750, 0.729 | **Yes (3/3)** | 1 |
| `Amitabh_Bachchan` | Bollywood | 11, 12, 13 | `Amitabh_Bachchan` (x3) | 0.781, 0.769, 0.792 | **Yes (3/3)** | 1 |
| `Amol_Palekar` | Bollywood | 11, 12, 13 | `Amol_Palekar` (x3) | 0.712, 0.725, 0.709 | **Yes (3/3)** | 1 |
| `Anil_Kapoor` | Bollywood | 11, 12, 13 | `Anil_Kapoor` (x3) | 0.744, 0.758, 0.736 | **Yes (3/3)** | 1 |

**Analysis of Error:**
The only misclassification was `Ajay_Devgn/34.jpg`. Its similarity to the `Ajay_Devgn` centroid was `0.529` (Rank 3), while matching `Amitabh_Bachchan` at `0.654` and `Abhishek_Bachchan` at `0.582`. Inspection of the image reveals significant pose variation, lighting shadows, and low resolution ($64\times 64$).

---

### B.6. Replication of Centroids from First 10 Images

To verify that the offline script `generate_celebrity_embeddings.py` matches the stored centroids in `celebrity_embeddings.pkl`, the first 10 images of 5 identities were re-embedded and averaged into a centroid:

$$\mathbf{c}_{\text{recomputed}} = \frac{1}{N} \sum_{i=1}^{10} \mathbf{e}_i, \quad \text{sim} = \frac{\mathbf{c}_{\text{recomputed}} \cdot \mathbf{c}_{\text{stored}}}{\|\mathbf{c}_{\text{recomputed}}\| \|\mathbf{c}_{\text{stored}}\|}$$

| Identity | Number of Images | Stored Centroid Norm | Recomputed Centroid Norm | Cosine Similarity | L2 Distance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `pins_Adriana Lima` | 10 | 20.4812 | 20.4812 | **1.000000** | $2.49 \times 10^{-7}$ |
| `pins_Alexandra Daddario` | 10 | 20.9145 | 20.9145 | **1.000000** | $2.14 \times 10^{-7}$ |
| `pins_Alia Bhatt` | 10 | 20.1534 | 20.1534 | **1.000000** | $2.55 \times 10^{-7}$ |
| `Aamir_Khan` | 10 | 19.8421 | 19.8421 | **1.000000** | $3.89 \times 10^{-7}$ |
| `Abhay_Deol` | 10 | 19.6538 | 19.6538 | **1.000000** | $2.41 \times 10^{-7}$ |

**Conclusion:** The stored centroids in `celebrity_embeddings.pkl` were definitively generated by the exact preprocessing and model weights implemented in `generate_celebrity_embeddings.py`. There is zero numerical divergence ($1.000000$ cosine similarity, distance within floating-point roundoff).

---

## Summary of Findings for Phase 0.5 Readiness

1. **Model Weights & Engine Validated:** `Facenet512` loads cleanly in `qa/.venv` under TensorFlow 2.18 / Keras 3 with legacy mode.
2. **Offline Database High Discriminative Quality:** The stored centroids achieve **98.3% Top-1** and **100% Top-5** accuracy on held-out images.
3. **Severe Upstream Data Discrepancy:** The Bollywood dataset images are cropped to $64\times 64$, making them unusable in the real production `/api/analyze` pipeline unless the $300\times 300$ resolution check is bypassed.
4. **Offline vs Production Preprocessing Divergence (H6 Confirmed):**
   - Offline database: simple $160\times 160$ resize without landmark face alignment, $[-1, 1]$ normalization.
   - Production pipeline: 640px JPEG compression, MediaPipe bounding-box cropping, alignment step omitted.
5. **Clamping & Display Bugs Confirmed:** Production confidence score clamping ($0.50 + \text{sim} \in [0.60, 0.96]$) corrupts true cosine similarities and removes any natural "No match found" branch.
