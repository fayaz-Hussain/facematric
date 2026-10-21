# FaceMatric

**Facial Symmetry Type & Golden Ratio Scoring App**

An AI-powered application that evaluates facial symmetry types and golden ratio (φ = 1.618) proportions using computer vision, landmark extraction, and deep learning.

---

## 📌 Master Blueprint & Specifications

The complete project vision, methodology, architecture, API contract, and testing specifications are maintained in:
- 📄 [**FaceMatric-prd.md**](FaceMatric-prd.md) — Master Product Requirements Document & API Contract
- 🧪 [**TESTING_PLAN.md**](TESTING_PLAN.md) — Source of truth for testing, validation, and QA verification

Please refer to `FaceMatric-prd.md` when implementing features, landmark extraction, scoring formulas, and celebrity look-alike matching, and `TESTING_PLAN.md` for quality criteria and test plans.

---

## 📁 Cleaned Project Structure

```text
FaceMatric/
├── FaceMatric-prd.md            # Master PRD & Task Breakdown (CRITICAL)
├── README.md                    # Project Documentation
├── vite.config.js               # Dev server configuration
├── package.json                 # Frontend dev dependencies
├── celebrity_dataset/           # Merged celebrity image dataset (Pins + Bollywood)
│   ├── Aamir_Khan/
│   ├── pins_Adriana Lima/
│   └── ...
└── frontend/                    # Web Client (Vanilla HTML, CSS, JavaScript)
    ├── index.html               # Upload & Landing Page
    ├── analyzing.html           # Processing & Scan State Page
    ├── results.html             # Detailed Results & Visual Overlay Page
    └── src/
        ├── assets/              # Logos and media assets
        ├── css/                 # Modular design system (variables, components)
        └── js/                  # Frontend logic (theme, upload, camera, results)
```

---

## 🚀 Getting Started

### Prerequisites
- **Python**: 3.12
- **Node.js**: v18 or higher

### Setup & Run Commands

```bash
cd backend
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
cd ../frontend && npm install && npm run dev
```

> [!NOTE]
> The first run downloads Facenet512 model weights (~94 MB). Note that TensorFlow has a large disk footprint.

Open `http://localhost:3000` in your browser.

---

## 🛠️ Planned Backend Architecture (from PRD)

- **Framework**: Python FastAPI
- **Landmark Detection**: MediaPipe Face Mesh (468 3D landmarks)
- **Golden Ratio Scoring**: Math formulation calculating divergence from $\phi = 1.618$ across 8 key ratios
- **Face Identity Matcher**: FaceNet embeddings evaluated against `celebrity_dataset/` via Cosine Similarity

---

## 🔒 Privacy First

User uploaded images are processed transiently in memory and discarded immediately after scoring. No user photo is saved or stored server-side.


---

## 📄 Data and Licensing

The celebrity look-alike matching feature uses face reference images from:
- **Pins Face Recognition Dataset** (scraped from Pinterest)
- **Bollywood Celeb Localized Face Dataset**

These datasets are intended strictly for educational and portfolio demonstration purposes. Their original licenses, copyright restrictions, and terms of use must be thoroughly reviewed and verified prior to any public or commercial deployment.

### Population Percentile Reference & FairFace Attribution

The percentile ranking compares each analyzed photograph with a public benchmark dataset, not with everyone. The population distribution is calibrated empirically on adult faces from the **FairFace** dataset:
- **Dataset Attribution:** Karkkainen, K., & Joo, J. (2021). *FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation*. IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2021).
- **License:** Licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- **Stratification:** Reference quantiles are derived from 2,017 accepted adult faces (ages 20-59) passing production frontal pose gates, sampled evenly across 7 demographic groups (East Asian, Indian, Black, White, Middle Eastern, Latino/Hispanic, Southeast Asian) and balanced by gender.
