# FaceMatric Backend

Python FastAPI backend for facial geometry analysis using the Golden Ratio (φ = 1.618).

## Architecture Overview

The backend processes facial images through 8 core components to deliver:
- **Golden Ratio Score** (0-100): Measures how closely facial proportions match φ
- **Face Symmetry Classification**: Categorizes face shape (oval, rectangle, round, square, heart, diamond)
- **Percentile Ranking**: Compares score against population distribution
- **Celebrity Matching**: Optional FaceNet-based look-alike detection (205 celebrities)
- **Visual Overlays**: Golden ratio guides for frontend rendering

### Core Principles
- ✅ **Privacy-First**: All processing in-memory, no persistent storage
- ✅ **Progressive Degradation**: Celebrity matching failures don't block core scoring
- ✅ **30-Second Timeout**: Fast processing with explicit timeout handling
- ✅ **Explicit Error Messages**: User-friendly guidance for all error scenarios

## Project Structure

```
backend/
├── app/
│   ├── components/           # 8 core processing components
│   │   ├── image_processor.py          # Image normalization & validation
│   │   ├── face_detector.py            # MediaPipe Face Mesh integration
│   │   ├── symmetry_classifier.py      # Face shape classification
│   │   ├── golden_ratio_scorer.py      # 8 facial ratio measurements
│   │   ├── percentile_engine.py        # Score-to-percentile conversion
│   │   ├── celebrity_matcher.py        # FaceNet embedding comparison
│   │   ├── overlay_generator.py        # Visual guide data generation
│   │   └── __init__.py
│   ├── models/               # Pydantic schemas & data structures
│   │   ├── requests.py                 # API request models
│   │   ├── responses.py                # API response models
│   │   ├── internal.py                 # Internal data structures
│   │   └── __init__.py
│   ├── utils/                # Shared utilities
│   │   ├── errors.py                   # Custom exceptions & error codes
│   │   ├── logger.py                   # Structured logging setup
│   │   └── __init__.py
│   ├── main.py               # FastAPI app & API gateway
│   └── config.py             # Environment configuration
├── data/                     # Model files & embeddings
│   ├── face_landmarker.task            # MediaPipe Face Mesh model
│   ├── celebrity_embeddings.pkl        # Precomputed FaceNet embeddings
│   └── README.md
├── tests/
│   ├── unit/                 # Component unit tests
│   └── integration/          # End-to-end API tests
├── requirements.txt          # Python dependencies
├── Dockerfile               # Container configuration
├── .env.example             # Environment variables template
└── README.md                # This file
```

## 8 Core Components

### 1. **Image Processor** (`image_processor.py`)
**Responsibility**: Accept, normalize, and validate uploaded images

**Features**:
- Supports JPEG, PNG, HEIC, WebP, BMP, GIF
- EXIF rotation handling
- Color space normalization (CMYK→RGB, RGBA→RGB)
- Aspect ratio-preserving resize (max 640px)
- JPEG compression at 85% quality

**Error Handling**:
- `INVALID_IMAGE`: Corrupt or unsupported format
- `POOR_QUALITY`: Resolution < 300x300px or aspect ratio > 3:1

---

### 2. **Face Detector** (`face_detector.py`)
**Responsibility**: Detect faces and extract 468 3D facial landmarks

**Technology**: MediaPipe Face Mesh

**Features**:
- Single face detection validation
- 468 landmark extraction with (x, y, z) coordinates
- Confidence score validation (threshold: 0.6)
- Occlusion detection (glasses, masks)

**Error Handling**:
- `NO_FACE_DETECTED`: No face found in image
- `MULTIPLE_FACES`: More than one face detected
- `POOR_QUALITY`: Low landmark confidence
- `OCCLUSION_DETECTED`: Face accessories blocking analysis

---

### 3. **Symmetry Classifier** (`symmetry_classifier.py`)
**Responsibility**: Classify face shape based on geometric proportions

**Algorithm**: Decision tree analyzing face dimensions

**Classifications**:
- **Oval**: Length/width ≈ 1.5, rounded jawline
- **Rectangle**: Length > width, angular jawline
- **Round**: Length ≈ width, soft angles
- **Square**: Length ≈ width, angular jawline
- **Heart**: Forehead > chin width significantly
- **Diamond**: Cheekbone width > forehead & chin

**Output**: Classification label only (no numeric score)

---

### 4. **Golden Ratio Scorer** (`golden_ratio_scorer.py`)
**Responsibility**: Compute 8 facial ratios and score against φ = 1.618

**The 8 Measurements**:
1. Face Length ÷ Face Width
2. Face Width ÷ Jaw Width
3. Eye Spacing ÷ Eye Width
4. Nose Width ÷ Mouth Width
5. Mouth Width ÷ Face Width
6. Nose-Chin Distance ÷ Hairline-Nose Distance (Vertical Thirds)
7. Eye-Eyebrow Distance ÷ Eye Height
8. Chin Width ÷ Nose Width

**Scoring Formula**:
```
score = 100 × (1 - |measured_ratio - φ| / φ)
overall_score = mean(8 feature scores)
```

**Output**: Overall score (0-100) + per-feature breakdown

---

### 5. **Percentile Engine** (`percentile_engine.py`)
**Responsibility**: Convert golden ratio score to population percentile

**Algorithm**: 
- Bootstrap distribution: Normal(μ=70, σ=10)
- CDF-based percentile calculation
- Future: Replace with empirical distribution from real user data

**Output**: "You score higher than X% of people"

---

### 6. **Celebrity Matcher** (`celebrity_matcher.py`)
**Responsibility**: Find celebrity look-alikes using face embeddings

**Technology**: FaceNet (128-dimensional embeddings)

**Features**:
- Generate user face embedding
- Compute cosine similarity with 205 celebrity embeddings
- Confidence threshold: 0.55 minimum
- Graceful degradation on failure

**Celebrity Database**:
- 105 Western celebrities
- 100 Bollywood celebrities
- Precomputed embeddings in `celebrity_embeddings.pkl`

**Output**: Celebrity name, confidence, dataset source, thumbnail URL (or null if confidence < 0.55)

---

### 7. **Overlay Generator** (`overlay_generator.py`)
**Responsibility**: Generate visual guide data for frontend rendering

**Features**:
- 468 2D landmark coordinates
- Vertical midline calculation
- Horizontal golden ratio thirds
- Phi spiral center coordinates

**Output**: JSON structure for client-side canvas rendering

---

### 8. **API Gateway** (`main.py`)
**Responsibility**: Orchestrate components and handle HTTP requests

**Endpoint**: `POST /api/analyze`

**Request**:
```json
{
  "image_base64": "base64_encoded_image_string",
  "include_celebrity_match": true
}
```

**Response** (Success - HTTP 200):
```json
{
  "status": "success",
  "processing_time_ms": 2847,
  "results": {
    "golden_ratio_score": 78.3,
    "percentile_rank": 82,
    "face_symmetry_type": "oval",
    "feature_breakdown": [
      {
        "feature": "Face Length ÷ Face Width",
        "measured_ratio": 1.625,
        "ideal_ratio": 1.618,
        "score": 99.6,
        "deviation": 0.007
      }
      // ... 7 more features
    ],
    "overlay_data": {
      "landmarks": [[x1, y1], [x2, y2], ...],
      "golden_ratio_guides": {
        "midline": 320,
        "horizontal_thirds": [120, 240, 360],
        "phi_spiral_center": [320, 180]
      }
    },
    "celebrity_match": {
      "name": "Ryan Gosling",
      "confidence": 0.72,
      "dataset_source": "Western",
      "thumbnail_url": "/celebrities/ryan_gosling.jpg"
    }
  }
}
```

**Response** (Error - HTTP 400):
```json
{
  "status": "error",
  "error_code": "NO_FACE_DETECTED",
  "message": "We couldn't detect a face in your photo. Please take a clear, front-facing photo with good lighting.",
  "retry": true
}
```

**Features**:
- 30-second timeout enforcement
- CORS configuration
- Privacy-preserving in-memory processing
- Progressive degradation for celebrity matching
- Comprehensive error handling

---

## Implementation Tasks

Based on the spec document, here are the **15 major task groups**:

### ✅ **Task 1: Backend Infrastructure Setup**
- [x] Create Python FastAPI project structure
- [ ] Install dependencies (FastAPI, MediaPipe, DeepFace, etc.)
- [ ] Configure logging infrastructure
- [ ] Create base error classes
- [ ] Environment variable management
- [ ] Dockerfile

### **Task 2: Data Models and Schemas**
- [ ] 2.1 Request models (AnalyzeRequest)
- [ ] 2.2 Response models (AnalysisResponse, etc.)
- [ ] 2.3 Internal data structures (ImageData, Landmarks, etc.)
- [ ] 2.4 Unit tests for data models

### **Task 3: Image Processor Component**
- [ ] 3.1 Format detection & validation
- [ ] 3.2 EXIF rotation handling
- [ ] 3.3 Color space normalization
- [ ] 3.4 Aspect ratio-preserving resize
- [ ] 3.5 JPEG compression & validation
- [ ] 3.6 Unit tests

### **Task 4: Face Detector & Landmark Extractor**
- [ ] 4.1 Initialize MediaPipe Face Mesh
- [ ] 4.2 Face detection with count validation
- [ ] 4.3 Extract 468 landmarks with confidence validation
- [ ] 4.4 Occlusion detection
- [ ] 4.5 Unit tests

### **Task 5: Symmetry Classifier**
- [ ] 5.1 Facial dimension measurements
- [ ] 5.2 Classification logic with decision tree
- [ ] 5.3 Unit tests

### **Task 6: Golden Ratio Scorer**
- [ ] 6.1 8 facial ratio measurement functions
- [ ] 6.2 Per-feature scoring with golden ratio formula
- [ ] 6.3 Overall score calculation
- [ ] 6.4 Asymmetric feature handling
- [ ] 6.5 Unit tests

### **Task 7: Percentile Engine**
- [ ] 7.1 Bootstrap normal distribution
- [ ] 7.2 Score-to-percentile conversion
- [ ] 7.3 Empirical distribution loading (future)
- [ ] 7.4 Unit tests

### **Task 8: Celebrity Matcher**
- [ ] 8.1 FaceNet embedding generation
- [ ] 8.2 Cosine similarity calculation
- [ ] 8.3 Celebrity database comparison
- [ ] 8.4 Confidence thresholding (0.55)
- [ ] 8.5 Celebrity database loading
- [ ] 8.6 Unit tests

### **Task 9: Overlay Generator**
- [ ] 9.1 2D landmark extraction
- [ ] 9.2 Golden ratio guide calculations
- [ ] 9.3 Unit tests

### **Task 10: API Gateway & Orchestration**
- [ ] 10.1 FastAPI app with CORS
- [ ] 10.2 /api/analyze endpoint
- [ ] 10.3 Timeout middleware (30s)
- [ ] 10.4 Component orchestration
- [ ] 10.5 Progressive degradation
- [ ] 10.6 Error handling
- [ ] 10.7 Privacy-preserving cleanup
- [ ] 10.8 Integration tests

### **Task 11: Checkpoint - Core Backend Complete**
- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] End-to-end test with Postman/curl

### **Task 12: Celebrity Database Preprocessing**
- [ ] 12.1 Merge celebrity datasets
- [ ] 12.2 Generate FaceNet embeddings
- [ ] 12.3 Average embeddings & save to pickle

### **Task 13: Frontend JavaScript Integration**
- [ ] 13.1 Create api-client.js
- [ ] 13.2 Create upload-handler.js
- [ ] 13.3 Create analyzing-handler.js
- [ ] 13.4 Create results-handler.js
- [ ] 13.5 Add script tags to HTML
- [ ] 13.6 Test end-to-end

### **Task 14: Deployment Configuration**
- [ ] 14.1 Backend deployment (Render/Railway)
- [ ] 14.2 Frontend deployment (Vercel/Netlify)
- [ ] 14.3 Test production deployment

### **Task 15: Final Testing & Validation**
- [ ] 15.1 Create diverse test dataset
- [ ] 15.2 End-to-end validation
- [ ] 15.3 Performance testing
- [ ] 15.4 Security & privacy validation

---

## Dependencies

### Core Dependencies
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6
```

### Image Processing
```
Pillow==10.1.0
opencv-python==4.8.1
numpy==1.26.2
```

### Face Analysis
```
mediapipe==0.10.8
scipy==1.11.4
```

### Celebrity Matching
```
deepface==0.0.79
tensorflow==2.15.0
```

### Utilities
```
python-dotenv==1.0.0
aiofiles==23.2.1
```

---

## Environment Variables

Create a `.env` file:

```bash
# API Configuration
ALLOWED_ORIGINS=http://localhost:5173,https://facematric.vercel.app
PROCESSING_TIMEOUT=30

# Paths
CELEBRITY_DB_PATH=./data/celebrity_embeddings.pkl
MEDIAPIPE_MODEL_PATH=./data/face_landmarker.task

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Feature Flags
ENABLE_CELEBRITY_MATCHING=true
```

---

## Getting Started

### Installation
```bash
cd backend
pip install -r requirements.txt
```

### Download Model Files
1. **MediaPipe Face Mesh**: Download `face_landmarker.task` from [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
2. Place in `backend/data/` directory

### Run Development Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Run Tests
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# All tests with coverage
pytest --cov=app tests/
```

---

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Next Steps

Ready to start building? Here's the recommended order:

1. **Start with Task 2**: Create data models (foundation for everything)
2. **Then Task 3**: Build Image Processor (first component in pipeline)
3. **Follow Task 4-9**: Build remaining components in order
4. **Task 10**: Wire everything together in API Gateway
5. **Task 12**: Preprocess celebrity database
6. **Task 13**: Integrate with frontend
7. **Task 14-15**: Deploy and validate

Let's begin! 🚀
