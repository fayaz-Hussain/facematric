# FaceMetric — Quality Assurance Reconciliation Report (Phase 9)

**Date:** 2026-10-18  
**Scope:** Forensic reconciliation of metrics, experimental setups, environment versions, and data discrepancies between `FINAL_REPORT.md` and the underlying phase reports (`00` through `08`).  
**Governing Policy:** [AGENTS.md](file:///c:/Users/FayazHussain/Desktop/FaceMatric/AGENTS.md). Production code and existing tests are unmodified.

---

## 1. Item-by-Item Reconciliation Matrix

| # | Item Description | Claim in Initial `FINAL_REPORT.md` | Verified Value from Raw Artifacts | Source Artifact(s) | Status | Root Cause & Explanation |
| :---: | :--- | :--- | :--- | :--- | :---: | :--- |
| **1** | Celebrity Top-1, Coverage, Precision & FMR | 54.9% top-1 on 2,131 test images across 187 identities; 76.2% precision; 34.5% FMR | **99.25% Top-1** (2,115/2,131), **97.47% Coverage**, **99.66% Precision**, **0.42% FMR** (1/240) on 40 validation identities; **Test split SEALED** | [celebrity_predictions.csv](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/celebrity_predictions.csv), [splits.csv](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/splits.csv), [07_celebrity_eval.md](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/07_celebrity_eval.md) | **CORRECTED** | Draft report contained fabricated/hallucinated test numbers. The test split was explicitly sealed; only the validation pilot (2,131 queries across 40 identities) and unknown LFW faces (240 images) were evaluated. |
| **2** | Same-Person Repeatability & ICC(1) | 12 identities, Mean SD = 1.81, ICC(1) = 0.854 | **15 identities** (72 photos), **Mean SD = 2.97**, **Median SD = 2.72**, **Max SD = 4.92**, **ICC(1) = 0.235** (Poor Reliability) | [05_same_person_stability.csv](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/05_same_person_stability.csv), [05_validity_summary.json](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/05_validity_summary.json) | **CORRECTED** | Draft report misreported sample size and variance decomposition. 9 of 15 identities (60.0%) met $\text{SD}\le 3.0$, and within-person variance ($76.5\%$) dominated between-person variance ($23.5\%$). |
| **3** | Mutation Testing Survival | 7 of 18 survived (38.9%) | **3 of 18 survived** (16.7%), **15 of 18 killed** (83.3%) | [00c_test_audit.md](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/00c_test_audit.md) Section 4 | **CORRECTED** | Draft report misstated mutant counts. The 3 surviving mutants were Mutant 11 (classifier ratio), Mutant 14 (EXIF rotation), and Mutant 17 (matcher shift). |
| **4** | Yaw Degradation Curve | Overall scores drop by $>12$ points beyond $15^\circ$ | Overall scores show a **gradual rise of $+5.87$ points** from $49.65$ ($0^\circ\text{--}5^\circ$) to $55.52$ ($>30^\circ$). Bins hold **DIFFERENT PEOPLE**. | [03_invariance.md](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/03_invariance.md) Section 2, [03_invariance_summary.json](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/03_invariance_summary.json) | **CORRECTED** | Draft report claimed an erroneous drop. Perspective foreshortening compresses eye width, artificially inflating Ratio 4 closer to $\phi$. The 282 photos are a cross-sectional sample of distinct individuals, not a single person rotated. |
| **5a**| Mirror-Flip Drift Median | Median $\Delta = 0.52$ pt | **Median $\Delta = 0.90$ pt** (Mean = 1.32 pt, P95 = 3.98 pt, Max = 4.30 pt) | [03_invariance.csv](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/03_invariance.csv), [03_invariance.md](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/03_invariance.md) line 36 | **CORRECTED** | Draft report used an unverified 0.52 figure. The verified median across 29 detected flipped frontal photos is 0.90 points. |
| **5b**| Empirical Population Mean Score | $\mu = 56.4 \pm 5.1$ | **$\mu = 52.41, \sigma = 4.62$** (Sample: **959 real photographs** from Phase 2 benchmark) | [05_validity_summary.json](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/05_validity_summary.json), [05_validity.md](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/05_validity.md) line 159 | **CORRECTED** | Draft report cited an unverified 56.4 figure. Baseline frontal set ($N=30$) mean was 49.62, and the large population benchmark ($N=959$) mean is 52.41. |
| **6** | `FM-BUG-017` Occlusion Findings | Claimed false rejections of clean portraits | Clean photos accepted **30 / 30 (100.0%)**. Synthetic dark sunglasses and masks **falsely accepted (HTTP 200)**. Real occlusion folders empty (**BLOCKED**). | [04_quality_gates.md](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/04_quality_gates.md) Section 3 | **CORRECTED** | The active defect is **100% false acceptance of occlusions** (heuristics fail to trigger on synthetic opaque overlays), not false rejection of clean photos. Results on occlusions stem from synthetic overlays. |
| **7** | Celebrity Input Path & Crop Comparison | Unverified claim of face cropping | Production path passes the **WHOLE UPLOAD RESIZED TO (160, 160)** without face crop. Mean cosine similarity between whole image and face crop is **0.9411** across 10 frontal images. | [celebrity_matcher.py:384-390](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/components/celebrity_matcher.py#L384-L390), [main.py:528](file:///c:/Users/FayazHussain/Desktop/FaceMatric/backend/app/main.py#L528), Empirical Evaluation | **VERIFIED** | Code inspection confirmed `_generate_embedding` resizes full frame. Across 10 frontal portraits, whole vs. MediaPipe-crop cosine similarity ranges from 0.8607 to 0.9763. |
| **8** | Environment Package Versions | Quoted conflicting versions (e.g. TF 2.15 vs 2.18, Python 3.10 vs 3.12) | **`qa/.venv`:** Python 3.12.10, TF 2.18.0, MediaPipe 0.10.35, DeepFace 0.0.79.<br>**`backend/venv`:** Python 3.12.10, TF Not Installed, MediaPipe 1.1.0, DeepFace Not Installed. | System Inspection (`pip list`, Python `-V`) | **CORRECTED** | Both environments run Python 3.12.10. `backend/venv` completely lacks TensorFlow and DeepFace. `qa/.venv` has TensorFlow 2.18.0 and MediaPipe 0.10.35. |

---

## 2. Detailed Technical Reconciliations

### 2.1 Celebrity Matching (Item 1)
- **Evaluation Command:**
  ```powershell
  .\qa\.venv\Scripts\python.exe -c "
  import pandas as pd
  df = pd.read_csv('qa/reports/celebrity_predictions.csv')
  val = df[df['split'] == 'validation']
  unk = df[df['split'] == 'unknown']
  top1 = (val['top1_identity'] == val['true_identity']).mean()
  val_ret = val[val['top1_similarity'] >= 0.55]
  cov = len(val_ret) / len(val)
  prec = (val_ret['top1_identity'] == val_ret['true_identity']).mean()
  fmr = (unk['top1_similarity'] >= 0.55).mean()
  print(f'Top-1: {top1:.4f}, Cov: {cov:.4f}, Prec: {prec:.4f}, FMR: {fmr:.4f}')
  "
  ```
- **Execution Result:**
  - `Top-1: 0.9925 (99.25%)` (2,115 / 2,131 correct)
  - `Coverage at 0.55: 0.9747 (97.47%)` (2,077 / 2,131 returned)
  - `Precision at 0.55: 0.9966 (99.66%)` (2,070 / 2,077 correct among returned)
  - `Unknown LFW Count: 240`
  - `False-Match Rate at 0.55: 0.0042 (0.42%)` (1 / 240 returned; `max similarity = 0.6074`, `mean = 0.3041`)
- **Status of Test Split:** In `splits.csv`, there are 11,456 images assigned to `test`. The test split was **NEVER evaluated**; it remains sealed per Phase 7.5 of `TESTING_PLAN.md` awaiting human approval of operating thresholds.

---

### 2.2 Same-Person Repeatability & ICC(1) (Item 2)
- **Evaluation Command:**
  ```powershell
  .\qa\.venv\Scripts\python.exe -c "
  import pandas as pd
  df = pd.read_csv('qa/reports/05_same_person_stability.csv')
  print('Identities:', len(df), 'Photos:', df['n_photos'].sum())
  print('Mean SD:', df['sd_score'].mean(), 'Median SD:', df['sd_score'].median())
  print('SD <= 3.0:', (df['sd_score'] <= 3.0).sum(), '/', len(df))
  "
  ```
- **Execution Result:**
  - Number of Identities: **15** (Adriana Lima, Alex Lawther, Alexandra Daddario, Alvaro Morte, Amanda Crew, Andy Samberg, Anne Hathaway, Anthony Mackie, Avril Lavigne, Ben Affleck, Bill Gates, Bobby Morley, Brenton Thwaites, Brian J. Smith, Brie Larson).
  - Total Photos: **72**.
  - Mean Within-Person SD: **2.97 points**.
  - Median Within-Person SD: **2.72 points**.
  - Target Compliance ($\text{SD} \le 3.0$): **9 of 15 identities (60.0%)**.
  - Variance Decomposition:
    - Between-Identity Variance ($\sigma_b^2$): **3.093** (23.5% of total variance)
    - Within-Identity Variance ($\sigma_w^2$): **10.084** (76.5% of total variance)
    - **$\text{ICC}(1) = 0.2347$** ($\implies$ Poor Reliability per biostatistical standards).

---

### 2.3 Mutation Testing (Item 3)
- **Source:** [qa/reports/00c_test_audit.md](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/00c_test_audit.md) Section 4.
- **Results:**
  - Total Mutants Hand-Applied: **18**
  - Mutants Killed: **15 (83.3%)**
  - Mutants Survived: **3 (16.7%)**
    - Mutant 11 (`symmetry_classifier.py`): Ratio threshold 1.6 $\to$ 1.76 survived (missing boundary test at 1.65).
    - Mutant 14 (`image_processor.py`): Commenting out `_apply_exif_rotation` survived (test only checked dimensions $>0$, not rotated pixels).
    - Mutant 17 (`celebrity_matcher.py`): Linear shift 0.50 $\to$ 0.40 survived (tests mocked scores; none asserted exact calibration output).

---

### 2.4 Yaw Degradation Curve (Item 4)
- **Evaluation Command:**
  ```powershell
  .\qa\.venv\Scripts\python.exe -c "
  import json
  with open('qa/reports/03_invariance_summary.json') as f:
      d = json.load(f)['yaw_degradation']
  for b, data in d.items():
      print(f'{b:12s}: N={data[\"count\"]:2d}, MeanYaw={data[\"mean_yaw\"]:5.2f} deg, Score={data[\"mean_score\"]:5.2f}')
  "
  ```
- **Execution Result:**
  - `0-5 deg`: $N=84$, Mean Yaw = $2.30^\circ$, Mean Overall Score = **49.65**
  - `5-10 deg`: $N=50$, Mean Yaw = $7.10^\circ$, Mean Overall Score = **50.03** ($+0.38$)
  - `10-15 deg`: $N=44$, Mean Yaw = $12.33^\circ$, Mean Overall Score = **51.63** ($+1.98$)
  - `15-20 deg`: $N=34$, Mean Yaw = $16.72^\circ$, Mean Overall Score = **53.35** ($+3.70$)
  - `20-25 deg`: $N=15$, Mean Yaw = $22.44^\circ$, Mean Overall Score = **55.34** ($+5.69$)
  - `25-30 deg`: $N=12$, Mean Yaw = $26.83^\circ$, Mean Overall Score = **54.30** ($+4.65$)
  - `>30 deg`: $N=33$, Mean Yaw = $37.11^\circ$, Mean Overall Score = **55.52** ($+5.87$)
- **Population Nature:** The 282 evaluated images are **DIFFERENT PEOPLE** (sampled from `celebrity_dataset/pins_*/*.jpg`). Scores **rise gradually by ~6 points** due to perspective foreshortening; they do **not** drop by 12 points.

---

### 2.5 Mirror-Flip & Population Mean Scores (Item 5)
- **Mirror-Flip Median:**
  - Command: `python -c "import pandas as pd; df = pd.read_csv('qa/reports/03_invariance.csv'); m = df[df['transform'] == 'mirror']; print('Median:', m['delta_score'].median(), 'Mean:', m['delta_score'].mean(), 'P95:', m['delta_score'].quantile(0.95))"`
  - Result: Evaluated across 29 detected flipped frontal photos from `qa/data/frontal_set/`. **Median $\Delta = 0.90$ pt**, Mean $\Delta = 1.32$ pt, P95 = 3.98 pt, Max = 4.30 pt.
- **Empirical Population Mean Score:**
  - Sourced from `05_validity_summary.json` and `05_validity.md` line 159.
  - Sample: **959 real photographs** from the Phase 2 dataset (`celebrity_dataset/pins_*`).
  - Empirical Distribution: **$\mu = 52.41, \sigma = 4.62$** (Max Observed = 68.1).
  - Baseline Frontal Set ($N=30$) mean is **49.62**, and 72-photo repeatability mean is **49.97**.

---

### 2.6 FM-BUG-017: Occlusion Filter Analysis (Item 6)
- **Curated Dataset Status:** The real-photo occlusion directories (`qa/data/occlusion/{sunglasses,glasses,mask,hand,hair}`) were empty ($N=0$) and marked **BLOCKED**.
- **Clean Photo Verification:** All 30 clean frontal portraits (100%) were accepted with HTTP 200 without false rejection.
- **Occlusion Testing Probes:** Evaluated via **synthetic black box overlays** (opaque black rectangle over eyes; opaque black polygon over mouth/nose/chin).
- **Result:** Both synthetic occlusions were **falsely accepted (HTTP 200)**. MediaPipe regularizes occluded landmarks into smooth 3D depth planes, keeping eye $z$-variance at $0.091 < 0.12$ and lower-face $z$-variance at $0.114 < 0.15$.
- **Bug Nature:** FM-BUG-017 is a failure to reject occluded photos (**100% false acceptance of synthetic occlusions**), not an aggressive false rejection of clean photos.

---

### 2.7 Production Celebrity Input & Face-Crop Embedding Comparison (Item 7)
- **Code Inspection:**
  - `backend/app/main.py:528` calls `celebrity_matcher.find_match(image_data=image_data)`.
  - `image_data` is the processed full-frame upload.
  - `backend/app/components/celebrity_matcher.py:384-390` executes:
    ```python
    rgb_img = image_data.array
    resized = cv2.resize(rgb_img, (160, 160))
    normalized = (resized.astype(np.float32) / 127.5) - 1.0
    batch = np.expand_dims(normalized, axis=0)
    ```
  - **Verdict:** The production path passes the **WHOLE UPLOAD RESIZED TO (160, 160)** to FaceNet. It does **NOT** crop to the MediaPipe face bounding box.
- **Empirical Comparison on 10 Frontal Images (`qa/data/frontal_set/*.jpg`):**
  - Model: DeepFace `Facenet512` (weights: `facenet512_weights.h5`)
  - Method A (Production): Whole image resized to $(160, 160) \to [-1, 1] \to \text{Facenet512} \to \text{L2 Normalize}$.
  - Method B (MediaPipe Crop): 468 landmarks extracted via `FaceDetector` $\to$ bounding box with 10% context margin $\to$ crop $\to$ resize to $(160, 160) \to [-1, 1] \to \text{Facenet512} \to \text{L2 Normalize}$.

| Image File | Original Dimensions | MediaPipe Crop Dimensions | Cosine Similarity (Whole vs Crop) |
| :--- | :---: | :---: | :---: |
| `frontal_00_pins_Adriana Lima_Adriana Lima0_0.jpg` | $550 \times 668$ | $541 \times 618$ | **0.9740** |
| `frontal_01_pins_Alex Lawther_Alex Lawther102_3.jpg` | $353 \times 463$ | $316 \times 357$ | **0.9420** |
| `frontal_02_pins_Alexandra Daddario_Alexandra Daddario107_224.jpg` | $302 \times 320$ | $237 \times 265$ | **0.9451** |
| `frontal_03_pins_Alvaro Morte_Alvaro Morte155_193.jpg` | $302 \times 320$ | $264 \times 303$ | **0.9575** |
| `frontal_04_pins_Amanda Crew_Amanda Crew0_0.jpg` | $365 \times 387$ | $252 \times 314$ | **0.8607** |
| `frontal_05_pins_Andy Samberg_Andy Samberg105_438.jpg` | $365 \times 387$ | $266 \times 326$ | **0.9470** |
| `frontal_06_pins_Anne Hathaway_Anne Hathaway100_296.jpg` | $302 \times 320$ | $259 \times 279$ | **0.9272** |
| `frontal_07_pins_Anthony Mackie_Anthony Mackie152_468.jpg` | $365 \times 387$ | $327 \times 361$ | **0.9616** |
| `frontal_08_pins_Avril Lavigne_Avril Lavigne150_601.jpg` | $510 \times 554$ | $458 \times 542$ | **0.9763** |
| `frontal_09_pins_Ben Affleck_Ben Affleck12_133.jpg` | $302 \times 320$ | $231 \times 281$ | **0.9199** |

- **Summary Statistics:**
  - Mean Cosine Similarity: **0.9411**
  - Standard Deviation: **0.0319**
  - Minimum Cosine Similarity: **0.8607** (Amanda Crew)
  - Maximum Cosine Similarity: **0.9763** (Avril Lavigne)
  *(Note: If tight landmark crops without 10% margin are used, mean similarity drops to 0.5140, min 0.4431).*

---

### 2.8 Virtual Environment Package Versions (Item 8)
Inspected via `pip list` and `python -c "import ..."`:

| Package / Runtime | `qa/.venv` Installed Version | `backend/venv` Installed Version | Discrepancy Impact |
| :--- | :---: | :---: | :--- |
| **Python** | **3.12.10** | **3.12.10** | Identical Python runtime |
| **TensorFlow** | **2.18.0** | **Not Installed** | Backend crashes or disables FaceNet at runtime |
| **MediaPipe** | **0.10.35** | **1.1.0** | Different landmark extraction internals |
| **DeepFace** | **0.0.79** | **Not Installed** | Silently disables celebrity matching in `backend/venv` (`FM-BUG-004`) |
| **FastAPI** | 0.104.1 | 0.143.0 | API framework version drift |
| **Starlette** | 0.27.0 | 1.7.0 | TestClient / ASGI version drift |
| **Pydantic** | 2.5.0 | 2.14.0 | Schema validation version drift |
| **Pillow** | 12.3.0 | 10.1.0 | Image decoding version drift |
| **Pillow-Heif** | 1.8.1 | **Not Installed** | Breaks HEIC support in `backend/venv` (`FM-BUG-006`) |
| **HTTPX** | 0.25.2 | **Not Installed** | Breaks integration tests in `backend/venv` (`FM-BUG-005`) |

---

## 3. Conclusion & Next Steps

All discrepancies identified by the user have been forensicly recomputed and verified from raw repository artifacts (`03_invariance.csv`, `05_same_person_stability.csv`, `05_validity_summary.json`, `celebrity_predictions.csv`, `splits.csv`, `00c_test_audit.md`, and direct python scripts). 

The corrected findings are now embedded into the updated [qa/reports/FINAL_REPORT.md](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/FINAL_REPORT.md).
