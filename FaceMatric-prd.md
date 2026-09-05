# PRD: FaceMatric
**Facial Symmetry Type & Golden Ratio Scoring App**

Author: Fayaz Hussain
Date: January 2026
Status: Draft v4 — Updated with technical specifications and scoring methodology

---

## 1. Problem / Idea

Give users an objective, color-blind score (0–100) based on golden-ratio facial proportions, with face symmetry type classification (oval, rectangle, etc.) for context — not based on skin tone, lighting, or ethnicity.

## 2. Goals

- Upload a photo or take one with the camera → detect face → return a 0–100 golden ratio score with percentile ranking.
- Classify and display face symmetry type (oval, rectangle, etc.) without scoring it.
- Works regardless of skin color (pure geometry, no color analysis).
- Visual feedback: golden ratio overlay, landmark visualization, symmetry type classification.
- Fast enough to feel "instant" with timeout handling for slow processing.

## 3. Non-Goals (v1)

- No "beauty personality" / face-reading gimmicks.
- No storing user photos server-side (privacy-first, process and discard immediately).
- No synthetic "perfect face" comparisons (no human has a perfect 100/100 score).
- No scalability optimization for large user bases (optimized for small user groups; scale later if needed).

## 4. Target Users

Students/portfolio reviewers, casual users curious about the golden ratio, and — since this is a build-to-learn project for you — a strong CV/ML portfolio piece.

## 5. Core User Flow

1. User uploads or takes a photo (any format, any size).
2. Backend normalizes, compresses, and resizes image to optimal format.
3. App detects a face; if none/multiple/poor quality/occluded, prompts user to retake with specific guidance.
4. App extracts landmarks → computes:
   - **Face symmetry type** (oval, rectangle, etc.) based on geometric classification
   - **Golden ratio score** (0–100) based on facial proportions
   - **Percentile ranking** (how user compares to general population)
   - **Celebrity look-alike match** (with confidence threshold)
5. App shows:
   - Overall golden ratio score (big number)
   - Percentile ranking ("You score higher than 73% of people")
   - Face symmetry type classification (informational, not scored)
   - Per-feature breakdown with individual scores
   - Visual overlay with golden ratio proportions
   - Closest celebrity match (if confidence ≥ 0.55)

## 6. Functional Requirements

| Feature | Priority | Details |
|---|---|---|
| Face detection (single face) | P0 | Reject multiple faces or no faces with user-friendly prompt |
| Image normalization & compression | P0 | Accept any format/size, convert to optimal format for processing |
| Landmark extraction (468-pt) | P0 | MediaPipe Face Mesh |
| Face symmetry type classification | P0 | Classify as oval, rectangle, etc. (informational only, NOT scored) |
| Golden ratio proportion score | P0 | 0–100 score based on facial proportions vs. golden ratio |
| Per-feature scoring | P0 | Individual scores for each facial ratio with breakdown |
| Percentile ranking | P0 | Compare score to population distribution |
| Visual overlay (golden ratio guides) | P1 | Show golden ratio proportions on face |
| Live camera scanning | P1 | Real-time capture, not just static upload |
| Occlusion detection | P1 | Detect glasses/masks and prompt for clear photo |
| Timeout handling | P1 | If processing exceeds threshold, return timeout error with retry option |
| Celebrity look-alike match | P2 | FaceNet embedding vs. merged celebrity dataset (Pins + Bollywood) |
| Confidence thresholding | P2 | Only show match if confidence ≥ 0.55; otherwise "No match found in dataset" |

## 7. Finalized Tech Stack

- **Backend:** Python + FastAPI
- **Landmark detection:** MediaPipe Face Mesh (Python SDK)
- **Face identity / embeddings:** FaceNet (via `deepface` wrapper or direct pretrained model)
- **Celebrity reference dataset:** Combined from two sources:
  - [Pins Face Recognition](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition) — 105 celebrities, 17,534 images (note: "pins" prefix in some folder names to be removed during preprocessing)
  - [Bollywood Celeb Localized Face Dataset](https://www.kaggle.com/datasets/sushilyadav1998/bollywood-celeb-localized-face-dataset) — 100 Bollywood celebrities
  - **Dataset status:** Merged, deduplicated, no future additions planned
- **Frontend:** React client calling the FastAPI backend
- **Deployment:** Render/Railway (backend) + Vercel (frontend)

## 8. Methodology — How It Actually Works

**Step 1 — Image Preprocessing**
- Accept any image format (JPEG, PNG, HEIC, WebP, etc.) and any size
- Backend normalizes, compresses, and resizes to optimal dimensions for MediaPipe (e.g., 640x480)
- Maintain aspect ratio during resize
- Convert to RGB color space if needed

**Step 2 — Face Detection & Quality Validation**
- Run MediaPipe Face Mesh to detect faces
- **Error cases with user-friendly prompts:**
  - **No face detected:** "We couldn't detect a face in your photo. Please take a clear, front-facing photo with good lighting."
  - **Multiple faces detected:** "Multiple faces detected. Please upload a photo with only one person."
  - **Poor lighting/angle:** "Photo quality is too low. Please ensure your face is well-lit and facing the camera directly."
  - **Occlusion detected (glasses/mask):** "We detected accessories covering your face. Please upload a photo without glasses, masks, or other obstructions for accurate analysis."
- If validation passes, extract 468 (x, y, z) landmark coordinates

**Step 3a — Face Symmetry Type Classification (Informational Only)**
Using landmark coordinates, classify face geometry into symmetry types based on established facial morphology research:
- **Oval:** Face length ~1.5x width, gently rounded features
- **Rectangle:** Face length significantly greater than width, angular jawline
- **Round:** Face length ≈ width, soft angles
- **Square:** Face length ≈ width, strong angular jawline
- **Heart:** Wider forehead, narrow chin
- **Diamond:** Widest at cheekbones, narrow forehead and chin

This classification is **displayed to the user for context** but **NOT used in scoring**. Users simply learn their face shape type.

**Step 3b — Golden Ratio Scoring (The Only Score)**
All scoring is based purely on **golden ratio (φ = 1.618) comparisons** of facial proportions:

**Key facial ratios measured (8 primary ratios):**
1. Face length ÷ face width
2. Face width ÷ jaw width
3. Eye spacing ÷ eye width
4. Nose width ÷ mouth width
5. Mouth width ÷ face width
6. Nose bridge to chin ÷ hairline to nose bridge
7. Eye-to-eyebrow distance ÷ eye height
8. Chin width ÷ nose width

**Per-feature scoring:**
- For each ratio, calculate: `score_i = 100 × (1 - |ratio_i - φ| / φ)`
- This gives a 0–100 score for each individual feature
- Features closer to φ score higher

**Handling outliers (asymmetric features):**
- Calculate individual scores for symmetrical feature pairs (e.g., left eye vs. right eye spacing)
- If one feature is asymmetric (scores low), it pulls down only its component score
- Overall score is weighted average of all 8 feature scores
- Display per-feature breakdown so users see which features scored high/low

**Overall golden ratio score:**
- Weighted average of all 8 feature scores
- Weights are equal by default (can be tuned based on research)
- Formula: `Overall Score = (Σ score_i) / 8`
- **No synthetic "perfect" reference:** A score of 100 is theoretically perfect φ alignment but unattainable by any human face

**Percentile ranking:**
- Maintain a reference distribution of scores from a calibration dataset (or bootstrap from initial users)
- Convert user's score to percentile: "Your score is higher than X% of people"
- Provides context: 75/100 might sound average, but "higher than 82% of people" frames it better

**Step 3c — Celebrity Look-Alike Branch (Separate, Confidence-Thresholded)**
Run in parallel using the same uploaded photo:
- **FaceNet** generates a face embedding vector
- Compare against precomputed celebrity embedding database
- Calculate cosine similarity scores
- **Confidence threshold: 0.55**
  - If best match has confidence ≥ 0.55 → display match with confidence percentage
  - If best match has confidence < 0.55 → display "No close match found in our dataset"
- Where a celebrity has multiple photos, embeddings are averaged into a single representative vector

**Step 4 — Timeout Handling**
- Set backend processing timeout threshold (e.g., 30 seconds)
- If processing (landmark detection + scoring + celebrity matching) exceeds timeout:
  - Return HTTP 408 (Request Timeout) with error message
  - Frontend displays: "Processing took too long. Please try again with a clearer photo."
- User can retry with same or different photo

**Step 5 — Combine and Display**
Backend returns structured response (see API Contract below) containing:
- Overall golden ratio score (0–100)
- Percentile ranking
- Face symmetry type (classification label)
- Per-feature breakdown with individual scores
- Visual overlay data (landmarks, golden ratio guides)
- Celebrity match (if confidence ≥ 0.55, else null)

## 9. API Contract

**Endpoint Design:** Follow modern REST practices with single endpoint for simplicity.

### `POST /api/analyze`

**Request:**
```json
{
  "image": "base64_encoded_image_string",
  "include_celebrity_match": true  // optional, default true
}
```
- Accepts multipart/form-data as alternative: `image` file field

**Response (Success - 200 OK):**
```json
{
  "status": "success",
  "processing_time_ms": 1847,
  "results": {
    "golden_ratio_score": 78.4,
    "percentile_rank": 73,
    "face_symmetry_type": "oval",
    "feature_breakdown": [
      {
        "feature": "Face Length ÷ Face Width",
        "measured_ratio": 1.52,
        "ideal_ratio": 1.618,
        "score": 82.3,
        "deviation": -0.098
      },
      {
        "feature": "Eye Spacing ÷ Eye Width",
        "measured_ratio": 1.70,
        "ideal_ratio": 1.618,
        "score": 74.1,
        "deviation": 0.082
      }
      // ... 6 more features
    ],
    "overlay_data": {
      "landmarks": [[x1, y1], [x2, y2], ...],  // 468 points
      "golden_ratio_guides": {
        "horizontal_thirds": [y1, y2, y3],
        "vertical_midline": x_center,
        "phi_spiral_center": [x, y]
      }
    },
    "celebrity_match": {
      "name": "Shahrukh Khan",
      "confidence": 0.67,
      "dataset_source": "Bollywood Celeb Dataset",
      "thumbnail_url": "/celebrities/shahrukh_khan.jpg"
    }
  }
}
```

**Response (No Celebrity Match - 200 OK):**
```json
{
  "status": "success",
  "results": {
    // ... (same as above)
    "celebrity_match": {
      "name": null,
      "confidence": 0.42,
      "message": "No close match found in our dataset"
    }
  }
}
```

**Response (Error - 400 Bad Request):**
```json
{
  "status": "error",
  "error_code": "NO_FACE_DETECTED",
  "message": "We couldn't detect a face in your photo. Please take a clear, front-facing photo with good lighting.",
  "retry": true
}
```

**Error Codes:**
- `NO_FACE_DETECTED`: No face found in image
- `MULTIPLE_FACES`: More than one face detected
- `POOR_QUALITY`: Low lighting or extreme angle
- `OCCLUSION_DETECTED`: Glasses, mask, or other obstruction
- `INVALID_IMAGE`: Corrupt or unsupported file format
- `IMAGE_TOO_LARGE`: File size exceeds limits (soft validation, backend will compress)

**Response (Timeout - 408 Request Timeout):**
```json
{
  "status": "error",
  "error_code": "PROCESSING_TIMEOUT",
  "message": "Processing took too long. Please try again with a clearer photo.",
  "retry": true
}
```

## 10. Non-Functional Requirements

### Performance
- **Target processing time:** < 3 seconds for P0 features (landmark detection + golden ratio scoring)
- **Timeout threshold:** 30 seconds hard limit
  - If exceeded, return 408 timeout error
  - User prompted to retry
- **Celebrity matching:** Can take additional 1-2 seconds (runs asynchronously, optional)
- **Optimization deferred:** Performance tuning happens after MVP validation

### Image Handling
- **Upload size:** No hard limit initially (backend compresses as needed)
- **Supported formats:** All common formats (JPEG, PNG, HEIC, WebP, BMP, GIF)
  - Backend converts to normalized format (JPEG @ 85% quality, 640px max dimension)
- **Minimum resolution:** 300x300px for accurate landmark detection
- **Aspect ratio:** Preserved during resize

### Scalability
- **Target load:** Small user base (< 100 concurrent users)
- **Deployment:** Single-instance backend initially
- **Scale later:** If traffic grows, implement load balancing + caching (out of scope for v1)

### Data & Privacy
- **Image retention:** Zero. Images deleted immediately after processing.
- **Logging:** Only metadata (timestamp, processing time, error codes) — no image data or embeddings.
- **Celebrity embeddings:** Precomputed and stored; never regenerated from user photos.
- **Compliance:** No PII stored; no GDPR concerns for v1 (educational project).

### Error Handling Strategy
- **Progressive degradation:** If celebrity matching fails, still return golden ratio score
- **User-friendly prompts:** All error messages include actionable guidance
- **Retry flow:** Frontend offers "Try Again" button on all recoverable errors
- **Non-recoverable errors:** Logged for debugging (corrupt files, server errors)

## 11. Celebrity Dataset Implementation

### Preprocessing Pipeline
1. **Merge datasets:** Combine Pins Face Recognition + Bollywood Celeb folders into single directory structure
2. **Rename folders:** Remove "pins_" prefix from celebrity names for consistency
3. **Deduplication:** Manually verified — no overlapping celebrities between datasets (Pins = Western, Bollywood = Indian)
4. **Structure:** `celebrity_dataset/{Celebrity_Name}/*.jpg`

### Embedding Generation (One-Time Setup)
1. Run FaceNet on all images in `celebrity_dataset/`
2. Generate embedding vector per image
3. Average embeddings per celebrity (handle lighting/angle variance)
4. Store in `celebrity_embeddings.pkl` (pickle file with `{name: vector}` mapping)

### Matching Process
1. User photo → FaceNet → embedding vector
2. Cosine similarity against all celebrity vectors
3. Rank by similarity score
4. Apply 0.55 confidence threshold
5. Return top match or null

### Update Mechanism
- **No future updates planned** for v1
- If needed later: add new folders to `celebrity_dataset/`, rerun embedding generation script

## 12. Visual Design Direction

**Design System: "Clinical Precision + Warmth"**

| Role | Color | Hex |
|---|---|---|
| Background | Soft off-white / near-black (light & dark mode) | `#FAFAF8` / `#0F0F12` |
| Primary accent | Deep indigo-violet (math/geometry aesthetic) | `#5B4FE9` |
| Secondary accent | Warm gold (phi symbol, score highlight) | `#D4A93B` |
| Success/high score | Soft emerald | `#3FAE7A` |
| Neutral text | Charcoal | `#1E1E24` |
| Overlay lines | Coral (high contrast for golden ratio guides) | `#FF6B5E` |

**Typography:** Inter or Space Grotesk (geometric, modern, precise)

**Key UI Components:**

1. **Upload/Camera Screen**
   - Large dropzone or camera viewfinder
   - Real-time guidance if live camera enabled ("Face the camera", "Move closer")

2. **Processing State**
   - Animated golden ratio spiral
   - Progress text: "Analyzing your face..."

3. **Results Screen**
   - **Hero:** Large golden ratio score (78.4/100) with percentile badge
   - **Face Type Card:** "Your face symmetry type: Oval" (informational callout)
   - **Breakdown Section:** Expandable accordion showing 8 feature scores with bars
   - **Visual Overlay:** Canvas with face photo + golden ratio guides + landmarks
   - **Celebrity Match:** Card with photo, name, confidence percentage (or "No match found")

4. **Error Screens**
   - Icon + friendly message + "Try Again" button
   - Specific guidance based on error type

## 13. Testing & Validation Strategy

### Test Dataset Requirements
- **Diversity:** Different ages, ethnicities, genders, facial structures
- **Angles:** Front-facing (ideal), slight angles (acceptable), profile (should reject)
- **Lighting:** Good, moderate, poor (should detect poor quality)
- **Occlusion:** Clear faces, glasses, sunglasses, masks

### Edge Cases to Test
1. **Multiple faces:** Group photo → should reject
2. **No face:** Landscape/object photo → should reject
3. **Extreme angle:** Side profile → should reject with quality warning
4. **Partial occlusion:** Sunglasses (minor) vs. mask (major) → detect and prompt
5. **Non-human:** Pet faces, cartoon drawings → should fail detection gracefully
6. **Celebrity match edge cases:**
   - Photo of actual celebrity from dataset → should match with high confidence
   - Generic face → may or may not match, threshold handles it
   - Confidence exactly 0.55 → should display match (≥ threshold)

### Baseline Metrics
- **No perfect 100 score:** Even synthetic faces won't score 100 (slight measurement variance)
- **Expected range:** Most users 60-85 range
- **High scores:** 85+ should be rare (top 5-10% percentile)

### Quality Assurance
- **Manual review:** Test with 20-30 diverse photos, verify scores feel "reasonable"
- **Landmark accuracy:** Visual inspection of overlay to ensure landmarks are correct
- **Celebrity matching:** Test with known celebrities to verify confidence scores
- **Error handling:** Trigger all error types and verify prompts are clear

## 14. Success Metrics (Portfolio Framing)

- ✅ Working end-to-end demo (upload → score → visual overlay → celebrity match)
- ✅ Accurate landmark detection on diverse test photos
- ✅ Clear, explainable golden ratio scoring methodology
- ✅ Intuitive error handling with actionable prompts
- ✅ Clean UI suitable for portfolio/LinkedIn demo video
- ✅ API documentation ready for technical interviews

## 15. Risks / Open Questions

### Sensitivity & Framing
- **Risk:** "Beauty scoring" apps can be controversial
- **Mitigation:** Frame as **"geometric proportion analysis"**, not beauty judgment
- **Disclaimer:** Add prominent disclaimer: *"This tool analyzes geometric facial proportions for educational purposes. Scores do not reflect beauty, worth, or health. Results are influenced by photo angle and quality."*

### Technical Risks
- **Non-frontal photos:** Will skew ratios; v1 relies on user following guidance (acceptable tradeoff)
- **MediaPipe accuracy:** Occasional landmark errors on difficult photos (handle via quality validation)
- **Celebrity dataset licensing:** Both datasets are public Kaggle datasets for educational/research use; keep project framed as portfolio/educational (not commercial)

### Open Questions
- **Percentile calibration:** Need initial dataset to establish distribution (bootstrap from first 100-200 users?)
- **Golden ratio weights:** Are all 8 ratios equally important, or should some be weighted higher?
- **Timeout threshold tuning:** 30 seconds reasonable, or too long for UX?

## 16. Future Roadmap (Post-v1)

- **v1.1:** Video mode (real-time scoring while moving face around)
- **v1.2:** Age/gender analysis (separate from scoring, informational only)
- **v2.0:** Historical tracking (requires storing data, conflicts with privacy — only if users opt in)

---

**Document Status:** Ready for technical design phase
**Next Steps:** Create API implementation spec and frontend component design

