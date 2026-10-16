# FaceMetric — Celebrity Dataset Audit Report (Phase 6)

**Date:** 2026-10-15  
**Scope:** Comprehensive audit of `celebrity_dataset/`, evaluation of the shipped 205-identity database (`backend/data/celebrity_embeddings.pkl`), perceptual hashing (`imagehash`), MediaPipe face detection, name hygiene, and contact sheet generation.  
**Execution Environment:** `qa/.venv` on Windows. Production code and existing tests were not modified. In compliance with `AGENTS.md`, no raw face images or embeddings are committed to git.

---

## 1. Executive Summary

1. **Dataset Size:** `celebrity_dataset/` contains **275 identity folders** and **29,791 images** (17,534 Western, 12,257 Bollywood).
2. **Arbitrary Truncation:** The shipped database contains only **205 identities** (105 Western, 100 Bollywood). **70 Bollywood identity folders (41.2% of the Bollywood collection) were silently omitted** during database generation.
3. **No-Face Ingestion into Shipped Centroids:** In the 2,050 images selected to generate the database (the first 10 images alphabetically per identity), **14 images had zero detected faces** and **5 images had multiple faces**. The offline script averaged raw image vectors regardless of whether a face was present.
4. **Near-Duplicates & Burst Capture:** **12 identities** have near-duplicate pairs (perceptual Hamming distance $\le 6$, dhash $\le 2$). Taking the first 10 images alphabetically selected sequential video frames and burst shots from single events rather than varied real-world face distributions.
5. **Name Hygiene & Formatting Flaws:**
   - **15 Western celebrities are stored in all-lowercase** (`alycia_dabnem_carey`, `barack_obama`, `elon_musk`, `margot_robbie`, etc.), which display in lowercase on the frontend.
   - Multiple misspellings: `alycia_dabnem_carey` (Debnam), `kiernen_shipka` (Kiernan), `Henry_Cavil` (Cavill).
   - Deadname retained: `ellen_page` (now Elliot Page).
   - Hardcoded local paths: All thumbnail paths in `celebrity_embeddings.pkl` contain absolute Windows paths specific to the developer machine (`C:/Users/FayazHussain/Desktop/...`).

---

## 2. Dataset Inventory & Identity Counts

### 2.1. Source Breakdown
| Source | Directory Prefix | Folder Count | Total Images | Mean Imgs / Identity | Min Imgs | Max Imgs |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Western (Pins)** | `pins_[Name]` | 105 | 17,534 | 167.0 | 86 | 237 |
| **Bollywood** | `[Name]` | 170 | 12,257 | 72.1 | 42 | 123 |
| **Combined** | — | **275** | **29,791** | **108.3** | **42** | **237** |

### 2.2. Usable Images per Identity
- **Identities with fewer than 20 images:** **0** (All 275 folders contain $\ge 42$ images).
- **Identities with fewer than 50 images:** **23** (all from Bollywood).
- **Identities with 100+ images:** **139** (all 105 Pins folders, plus 34 Bollywood folders).

---

## 3. Shipped Database Audit (`celebrity_embeddings.pkl`)

The shipped file `backend/data/celebrity_embeddings.pkl` was produced by `backend/scripts/preprocess_celebrity_database.py`.

### 3.1. Vector Integrity
- **Total Registered Identities:** 205 (105 Western, 100 Bollywood).
- **Embedding Dimensions:** All 205 embeddings are strictly `(512,)` float32 numpy arrays matching `Facenet512`.
- **NaN / Inf / Zero Checks:** **0 NaN vectors, 0 Inf vectors, 0 all-zero vectors**. All embeddings are normalized unit vectors ($\|\mathbf{e}\|_2 = 1.000000$).

### 3.2. Unused Identities (70 Bollywood Folders Dropped)
The preprocessing script lines 63–64 hardcodes:
```python
selected_western = western_folders[:105]    # Uses all 105 Pins folders
selected_bollywood = bollywood_folders[:100]  # Drops remaining 70 folders!
```
The remaining 70 Bollywood folders (e.g. `Priyanka_Chopra`, `Ranbir_Kapoor`, `Rani_Mukerji`, `Rishi_Kapoor`, `Saif_Ali_Khan`, `Salman_Khan`, `Shah_Rukh_Khan`, `Shraddha_Kapoor`, `Sunil_Shetty`, `Tabu`, `Vicky_Kaushal`, etc.) were completely excluded from the look-alike matcher despite being fully available in the local dataset.

### 3.3. Thumbnail File Path Fragility
In all 205 database entries, `thumbnail_path` stores:
```python
"C:/Users/FayazHussain/Desktop/FaceMatric/celebrity_dataset/pins_Adriana Lima/Adriana Lima0_0.jpg"
```
- **Portability defect:** If the project directory is cloned to a different path or deployed to Linux/Docker, every single thumbnail fails to resolve.
- While `main.py:566-575` contains a URL normalizer that attempts to extract the relative path after `celebrity_dataset/`, storing absolute host paths directly inside the serialized database violates deployment isolation.

---

## 4. Shipped Selection Audit (205 Identities × First 10 Images = 2,050 Images)

The preprocessing script built each centroid by taking `images[:10]` alphabetically, resizing to $160\times 160$, and running batch inference without face detection or alignment.

Every image of this 2,050-image set was audited using PIL, perceptual hashing (`imagehash`), and MediaPipe `FaceLandmarker`:

| Metric | Western (1,050 Imgs) | Bollywood (1,000 Imgs) | Total (2,050 Imgs) | Share |
| :--- | :---: | :---: | :---: | :---: |
| **Corrupt Files** | 0 | 0 | **0** | 0.0% |
| **No Face Detected (0 faces)** | 3 | 11 | **14** | 0.7% |
| **Multiple Faces (>1 faces)** | 2 | 3 | **5** | 0.2% |
| **Clean Single Face (1 face)** | 1,045 | 986 | **2,031** | 99.1% |
| **Exact File Duplicates (SHA256)** | 0 | 0 | **0** | 0.0% |
| **Identities with Near-Duplicates** | 1 | 11 | **12** | 5.9% |

### 4.1. Images with No Face Detected (Ingested into Centroids)
The following 14 images contained no face detectable by MediaPipe ($min\_confidence = 0.50$), yet were averaged into the celebrity's stored facial centroid:
1. `pins_Anne Hathaway`: 1 image
2. `pins_Dwayne Johnson`: 1 image
3. `pins_Morgan Freeman`: 1 image
4. `Aamir_Khan`: 1 image (`Aamir.106.jpg`)
5. `Adah_Sharma`: 1 image
6. `Alia_Bhatt`: 1 image
7. `Dulquer_Salmaan`: 1 image
8. `Ileana_DCruz`: 1 image
9. `Karisma_Kapoor`: 1 image
10. `Kiara_Advani`: 1 image
11. `N.T.Rama_Rao_Jr`: 1 image
12. `Nagarjuna`: 1 image
13. `Nana_Patekar`: 1 image
14. `Pankaj_Kapur`: 1 image

### 4.2. Images with Multiple Faces Detected
The following 5 images contained group photos or bystanders:
1. `pins_Christian Bale`: 1 image
2. `pins_Zac Efron`: 1 image
3. `Arjun_Rampal`: 1 image
4. `Kay_Kay_Menon`: 1 image
5. `Pankaj_Kapur`: 1 image

### 4.3. Near-Duplicate Images & Photo Set Homogeneity
Because the first 10 images were selected alphabetically, sequential event naming resulted in multiple near-identical frames being averaged:

| Identity | Image 1 | Image 2 | pHash Distance | dHash Distance | Note |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `Prabhas` | `Prabhas.126.jpg` | `Prabhas.127.jpg` | **2** | **1** | Consecutive video frames |
| `Alia_Bhatt` | `Alia_Bhatt.104.jpg` | `Alia_Bhatt.115.jpg` | **4** | **1** | Same event photoshoot |
| `Akshaye_Khanna` | `Akshaye_Khanna.100.jpg` | `Akshaye_Khanna.108.jpg` | **2** | **2** | Near-identical crop |
| `Abhay_Deol` | `Abhay_Deol.113.jpg` | `Abhay_Deol.124.jpg` | **4** | **3** | Repeated press photo |
| `Boman_Irani` | `Boman_Irani_15.JPG` | `Boman_Irani_17.jpg` | **4** | **2** | Consecutive pose |
| `Nushrat_Bharucha` | `Nushrat_Bharucha.124.jpg` | `Nushrat_Bharucha.125.jpg` | **4** | **1** | Consecutive burst |
| `pins_Robert Downey Jr`| `Robert Downey Jr112_4091.jpg`| `Robert Downey Jr116_4094.jpg`| **6** | **7** | Red carpet burst |

**Photo Set Effect:** In Bollywood identities (e.g. `Nushrat_Bharucha`, `Nayanthara`, `Kriti_Kharbanda`), average pairwise perceptual hash distance dropped below 22.0. Rather than representing 10 diverse captures across varying ages, angles, and lighting, the centroid is heavily overfitted to a single scene or press release.

---

## 5. Name Hygiene, Formatting & Classification Issues

### 5.1. Lowercase Naming Defect (15 Western Identities)
15 Western identities were stored in all-lowercase in `celebrity_embeddings.pkl`:
- `alycia_dabnem_carey`, `amber_heard`, `barack_obama`, `barbara_palvin`, `camila_mendes`, `elizabeth_olsen`, `ellen_page`, `elon_musk`, `gal_gadot`, `grant_gustin`, `jeff_bezos`, `kiernen_shipka`, `margot_robbie`, `melissa_fumero`, `scarlett_johansson`, `tom_ellis`.

When matched in production, the frontend receives `"name": "barack_obama"` instead of `"Barack Obama"`.

### 5.2. Misspellings and Deadnames
- **`alycia_dabnem_carey`:** Folder and key misspelled ("dabnem" instead of "Debnam").
- **`kiernen_shipka`:** Misspelled ("kiernen" instead of "Kiernan").
- **`Henry_Cavil`:** Misspelled (missing second "l").
- **`ellen_page`:** Deadname for Elliot Page (Pins dataset predates 2020 transition).

### 5.3. Source Category Inconsistencies
- **Non-Actors in "Western":** Politicians (`barack_obama`), tech founders (`elon_musk`, `jeff_bezos`, `bill_gates`, `mark_zuckerberg`), and athletes (`cristiano_ronaldo`, `lionel_messi`) are classified under the generic "Western" label without role tagging.
- **Foreign / Regional Actors in "Bollywood":**
  - British actress `Amy_Jackson` and French-descended actress `Kalki_Koechlin` are classified under "Bollywood".
  - Pakistani actress `Mahira_Khan` is classified under "Bollywood".
  - Over a dozen South Indian (Tollywood, Kollywood, Mollywood) stars (`Chiranjeevi`, `Nagarjuna`, `Mohanlal`, `Allu_Arjun`, `Mahesh_Babu`, `Prabhas`, `Nayanthara`) are grouped as "Bollywood" (Hindi cinema).

---

## 6. Visual Contact Sheets

Two contact sheet grids ($6 \times 5 = 30$ random identities each) were generated and saved to `qa/reports/overlays/`:
- **Pins (Western):** [qa/reports/overlays/pins_contact_sheet.jpg](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/overlays/pins_contact_sheet.jpg)
- **Bollywood:** [qa/reports/overlays/bollywood_contact_sheet.jpg](file:///c:/Users/FayazHussain/Desktop/FaceMatric/qa/reports/overlays/bollywood_contact_sheet.jpg)

### Human Review Observations:
1. **Pins Images:** High visual quality, natural lighting, diverse angles, resolution ranging from $300\times 300$ to $800\times 800$. All 30 sampled faces are clearly centered portraits.
2. **Bollywood Images:** Extremely tight facial crops at very low resolution ($64 \times 64$). Visual inspection shows heavy JPEG compression artifacts, pixelation, and motion blur from movie screengrabs, explaining why MediaPipe landmark detection fails on some frames.

---

## 7. Reproduction Command

To reproduce all dataset metrics, hashes, face detections, and contact sheets:
```powershell
.\qa\.venv\Scripts\python.exe qa/eval/dataset_audit.py
```
Outputs:
- Audit Data: `qa/eval/dataset_audit_results.pkl`
- Contact Sheets: `qa/reports/overlays/pins_contact_sheet.jpg`, `qa/reports/overlays/bollywood_contact_sheet.jpg`
