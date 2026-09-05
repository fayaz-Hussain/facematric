# PRD: FaceSymmetry (working title)
**Facial Symmetry & Golden Ratio Scoring App**

Author: Fayaz Hussain
Date: August 2026
Status: Draft v3 — stack, methodology, and celebrity dataset finalized

---

## 1. Problem / Idea

Give users an objective, color-blind score (0–100) of facial symmetry and golden-ratio proportion, based purely on geometric landmark analysis — not skin tone, lighting, or ethnicity.

## 2. Goals

- Upload a photo or take one with the camera → detect face → return a 0–100 score with a breakdown.
- Works regardless of skin color (pure geometry, no color analysis).
- Visual feedback: symmetry line overlay, mirrored comparison, ratio breakdown.
- Fast enough to feel "instant" (under ~2–3 seconds per image).

## 3. Non-Goals (v1)

- No "beauty personality" / face-reading gimmicks.
- No storing user photos server-side (privacy-first, process and discard).

## 4. Target Users

Students/portfolio reviewers, casual users curious about the golden ratio, and — since this is a build-to-learn project for you — a strong CV/ML portfolio piece.

## 5. Core User Flow

1. User uploads or takes a photo.
2. App detects a face; if none/multiple, prompts to retry.
3. App extracts landmarks → computes symmetry score + golden-ratio score, and separately runs a celebrity look-alike match.
4. App shows: overall score (big number), sub-scores (symmetry, proportion), a visual overlay (midline + mirrored ghost image), and the closest celebrity match.

## 6. Functional Requirements

| Feature | Priority |
|---|---|
| Face detection (single face) | P0 |
| Live camera scanning (real-time capture, not just static upload) | P1 |
| Landmark extraction (468-pt) | P0 |
| Symmetry score algorithm | P0 |
| Golden ratio proportion score | P0 |
| Combined 0–100 score | P0 |
| Visual overlay (midline, mirror) | P1 |
| Per-feature breakdown (eyes/nose/mouth ratios) | P1 |
| Celebrity look-alike match (FaceNet embedding vs. merged Pins + Bollywood celebrity dataset) | P2 |

## 7. Finalized Tech Stack

Decision made: **Python/FastAPI backend**, chosen to showcase backend + ML pipeline skills for your final-semester portfolio, and because celebrity matching needs a server-side embedding database anyway.

- **Backend:** Python + FastAPI
- **Landmark detection:** MediaPipe Face Mesh (Python SDK)
- **Face identity / embeddings:** FaceNet (via the `deepface` wrapper, or a direct pretrained FaceNet model)
- **Celebrity reference dataset:** Two datasets merged into one shared folder structure —
  - [Pins Face Recognition](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition) — 105 celebrities, 17,534 images, Pinterest-sourced, pre-cropped and labeled by folder
  - [Bollywood Celeb Localized Face Dataset](https://www.kaggle.com/datasets/sushilyadav1998/bollywood-celeb-localized-face-dataset) — 100 Bollywood celebrities, pre-localized/cropped faces, folder-per-actor
- **Frontend:** React client calling the FastAPI backend
- **Deployment:** Render/Railway (backend) + Vercel (frontend)

## 8. Methodology — How It Actually Works

This is the core of the system: **one uploaded photo feeds two independent processing branches**, both running inside the FastAPI backend. Neither branch depends on the other — they just both read the same image and produce two separate outputs that the frontend displays together.

**Step 1 — Landmark detection (shared starting point)**
The photo goes into **MediaPipe Face Mesh**, which detects the face and returns 468 (x, y, z) landmark coordinates covering the eyes, nose, mouth, jawline, and more. MediaPipe does nothing beyond this — no scoring, no identity, no color analysis. It hands off pure geometry, which is exactly why the whole system is color-blind by design.

**Step 2a — Symmetry & golden ratio branch (pure math, no model)**
Using only the landmark coordinates from Step 1:
- *Symmetry score:* mirror every landmark across the face's vertical midline (through the nose bridge), measure the average normalized distance between each real landmark and its mirrored counterpart, and convert that distance into a 0–100 score (smaller distance = higher score).
- *Golden ratio score:* compute 5–8 key facial ratios (face length ÷ width, eye spacing ÷ nose width, mouth width ÷ nose width, etc.), compare each to φ (1.618), and score by closeness.
- *Combined score:* a weighted average (e.g. 60% symmetry + 40% golden ratio, tuned after testing) becomes the headline 0–100 number.
This branch is plain NumPy — no additional ML model is involved at all.

**Step 2b — Celebrity look-alike branch (a second, separate model)**
Run in parallel, using the same uploaded photo:
- **FaceNet** generates a face embedding — a numeric vector that captures facial identity (not geometry, not landmarks — this is a different kind of representation entirely).
- That embedding is compared against a **precomputed database of celebrity embeddings**, built ahead of time by running FaceNet once over the merged celebrity dataset (Pins Face Recognition + Bollywood Celeb Localized Face Dataset — folders combined into one root directory before embedding generation).
- The comparison uses cosine or Euclidean distance; the closest match(es) are returned as the look-alike result.
- Where a celebrity has multiple photos, their embeddings are averaged into a single representative vector per person, reducing noise from lighting/angle differences before comparison.

**Step 3 — Combine and display**
The backend returns both results in one response: the symmetry/golden-ratio score with its breakdown and overlay data, and the celebrity match. The frontend renders them together as a single result screen.

**Why two separate models matter here:** MediaPipe answers "where are this face's features" (geometry), FaceNet answers "whose face is this most similar to" (identity). They're solving different problems, so there was never a way to get celebrity matching "for free" out of MediaPipe — it always required bringing in a second, purpose-built model.

## 9. Visual Design Direction

**Suggested palette — "clinical but warm" (trust + aesthetics, not garish beauty-app pink):**

| Role | Color | Hex |
|---|---|---|
| Background | Soft off-white / near-black (light & dark mode) | `#FAFAF8` / `#0F0F12` |
| Primary accent | Deep indigo-violet (ties to "golden ratio/math" feel without being cliché gold) | `#5B4FE9` |
| Secondary accent | Warm gold (sparingly — for the phi symbol, score highlight) | `#D4A93B` |
| Success/high score | Soft emerald | `#3FAE7A` |
| Neutral text | Charcoal | `#1E1E24` |
| Overlay/mirror line | Coral (high contrast against skin tones for the midline overlay) | `#FF6B5E` |

Reasoning: avoid the typical "beauty app" hot-pink/gold gradient look (feels gimmicky, low-trust) — lean into a data-viz / precision-tool aesthetic (think Notion x Figma) with gold used only as a small accent tied to the phi concept, not the whole theme.

**Typography:** A clean geometric sans (Inter, or Space Grotesk if you want a slightly more "mathematical" feel) for a precise, modern tone.

## 10. Success Metrics (portfolio framing)

- Working end-to-end demo (upload → score → visual overlay → celebrity match)
- Landmark accuracy visibly correct on test photos
- Clear, explainable scoring methodology (good for viva/interview discussion)
- Clean UI suitable for a portfolio/LinkedIn demo video

## 11. Risks / Open Questions

- Sensitivity around "beauty scoring" apps — consider framing as "symmetry & proportion analysis" rather than "beauty score" to keep it feeling like a geometry/CV tool, not a judgment tool.
- Non-frontal photos will skew ratios — v1 should nudge users toward front-facing, neutral-expression photos.
- Celebrity dataset licensing — both selected datasets (Pins Face Recognition, Bollywood Celeb Localized Face Dataset) are public Kaggle datasets intended for research/educational face-recognition projects; confirm their listed terms before any public/commercial deployment, and keep usage framed as a portfolio/educational project.
- No Pakistani celebrity coverage yet — neither dataset includes Pakistani actors; a hand-curated folder (15–20 photos each for ~20–50 Pakistani celebrities) would need to be added separately and merged in the same way if that coverage matters for the final demo.
