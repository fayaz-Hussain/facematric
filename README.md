# FaceMatric

**Facial Symmetry Type & Golden Ratio Scoring App**

An AI-powered application that evaluates facial symmetry types and golden ratio (φ = 1.618) proportions using computer vision, landmark extraction, and deep learning.

---

## 📌 Master Blueprint & PRD

The complete project vision, methodology, architecture, API contract, and task breakdown are maintained in:
- 📄 **`FaceMatric-prd.md`** *(Master Product Requirements Document & Task Breakdown)*

Please refer to `FaceMatric-prd.md` when implementing backend features, landmark extraction, scoring formulas, and celebrity look-alike matching.

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
- **Node.js**: v18 or higher

### Frontend Development Server
To launch the frontend preview with live reload:
```bash
npm run dev
```
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
