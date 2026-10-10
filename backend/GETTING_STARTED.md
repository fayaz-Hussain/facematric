# Getting Started with FaceMatric Backend

Welcome! This guide will help you start building the FaceMatric backend step by step.

## 📋 Prerequisites

- Python 3.11+
- pip (Python package manager)
- Git

## 🚀 Quick Start

### 1. Set Up Environment

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Required Model Files

#### MediaPipe Face Mesh Model
1. Visit https://developers.google.com/mediapipe/solutions/vision/face_landmarker#models
2. Download `face_landmarker.task` (~26 MB)
3. Place it in `backend/data/face_landmarker.task`

### 3. Create Environment File

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings (optional for development)
```

### 4. Verify Setup

```bash
# Test Python imports
python -c "import fastapi, mediapipe, cv2, PIL; print('✅ All dependencies installed!')"
```

---

## 📚 Implementation Roadmap

Follow these tasks in order for systematic development:

### **Phase 1: Foundation (Tasks 2-3)**
**Goal**: Set up data models and image processing

#### Task 2: Data Models (2-3 hours)
✅ **Start here!** Create Pydantic models for API communication.

Files to create:
- `app/models/internal.py` - Internal data structures
- `app/models/requests.py` - API request schemas
- `app/models/responses.py` - API response schemas

**Why start here?** 
Models define the contract between components. Build these first so all components know what data structures to expect.

**Key deliverables**:
- `ImageData` dataclass
- `Landmarks` dataclass
- `AnalyzeRequest` Pydantic model
- `AnalysisResponse` Pydantic model
- Request/response validation

#### Task 3: Image Processor (3-4 hours)
Build the first component in the processing pipeline.

File to create:
- `app/components/image_processor.py`

**Functionality**:
- Accept JPEG, PNG, HEIC, WebP, BMP, GIF
- Handle EXIF rotation
- Normalize color spaces (CMYK→RGB, RGBA→RGB)
- Resize with aspect ratio preservation (max 640px)
- JPEG compression at 85% quality
- Validation (min 300x300px, aspect ratio checks)

**Test file**: `tests/unit/test_image_processor.py`

---

### **Phase 2: Core Analysis (Tasks 4-6)**
**Goal**: Implement face detection and golden ratio scoring

#### Task 4: Face Detector (4-5 hours)
Integrate MediaPipe Face Mesh for landmark extraction.

File to create:
- `app/components/face_detector.py`

**Functionality**:
- Initialize MediaPipe with correct settings
- Detect single face (reject multiple/no faces)
- Extract 468 3D landmarks
- Validate landmark confidence
- Detect occlusions (glasses, masks)

**Test file**: `tests/unit/test_face_detector.py`

#### Task 5: Symmetry Classifier (2-3 hours)
Classify face shape based on geometric ratios.

File to create:
- `app/components/symmetry_classifier.py`

**Functionality**:
- Measure face dimensions (length, width, jaw, etc.)
- Compute jawline angle
- Decision tree classification
- Return one of 6 types: oval, rectangle, round, square, heart, diamond

**Test file**: `tests/unit/test_symmetry_classifier.py`

#### Task 6: Golden Ratio Scorer (5-6 hours)
**This is the heart of FaceMatric!**

File to create:
- `app/components/golden_ratio_scorer.py`

**Functionality**:
- Measure 8 facial ratios using specific landmark indices
- Score each ratio against φ = 1.618
- Compute overall score (mean of 8 features)
- Handle asymmetric features
- Return detailed feature breakdown

**Test file**: `tests/unit/test_golden_ratio_scorer.py`

---

### **Phase 3: Auxiliary Components (Tasks 7-9)**
**Goal**: Add percentile ranking, celebrity matching, and visual overlays

#### Task 7: Percentile Engine (1-2 hours)
Convert scores to population percentiles.

File to create:
- `app/components/percentile_engine.py`

**Functionality**:
- Bootstrap normal distribution (μ=70, σ=10)
- CDF-based percentile calculation
- Format output: "You score higher than X% of people"

**Test file**: `tests/unit/test_percentile_engine.py`

#### Task 8: Celebrity Matcher (4-5 hours)
FaceNet-based celebrity look-alike matching.

File to create:
- `app/components/celebrity_matcher.py`

**Functionality**:
- Generate 128-dimensional FaceNet embeddings
- Compute cosine similarity with celebrity database
- Apply confidence threshold (0.55)
- Graceful degradation on failure

**Test file**: `tests/unit/test_celebrity_matcher.py`

**Note**: Requires celebrity embeddings (Task 12). You can stub this for now.

#### Task 9: Overlay Generator (1-2 hours)
Generate visual guide data for frontend.

File to create:
- `app/components/overlay_generator.py`

**Functionality**:
- Convert 468 3D landmarks to 2D
- Calculate vertical midline
- Calculate horizontal golden ratio thirds
- Compute phi spiral center

**Test file**: `tests/unit/test_overlay_generator.py`

---

### **Phase 4: API Integration (Task 10)**
**Goal**: Wire all components together into FastAPI endpoint

#### Task 10: API Gateway (6-8 hours)
**The orchestration layer that connects everything.**

File to create:
- `app/main.py`

**Functionality**:
- FastAPI app with CORS
- `POST /api/analyze` endpoint
- 30-second timeout middleware
- Component orchestration (parallel processing where possible)
- Progressive degradation (celebrity matching can fail)
- Comprehensive error handling
- Privacy-preserving memory cleanup
- Health check endpoint

**Test file**: `tests/integration/test_api.py`

**Key considerations**:
- Use `asyncio.wait_for` for timeout
- Use `asyncio.create_task` for parallel processing
- Catch exceptions from celebrity matcher without blocking
- Log only metadata (no image data)

---

### **Phase 5: Celebrity Database (Task 12)**
**Goal**: Preprocess celebrity images into embeddings

#### Task 12: Celebrity Database Preprocessing (3-4 hours)

Script to create:
- `scripts/preprocess_celebrity_database.py`

**Functionality**:
1. Scan `../celebrity_dataset/` folders (275 celebrities)
2. Generate FaceNet embeddings for each image
3. Average embeddings per celebrity
4. Save to `data/celebrity_embeddings.pkl`

**Output format**:
```python
{
    "Ryan_Gosling": {
        "embedding": np.array([...]),  # 128-dim vector
        "dataset_source": "Western",
        "thumbnail_path": "celebrity_dataset/Ryan_Gosling/1.jpg",
        "image_count": 8
    },
    ...
}
```

---

### **Phase 6: Testing & Deployment (Tasks 11, 14, 15)**

#### Task 11: Checkpoint Testing
- Run all unit tests: `pytest tests/unit/`
- Run integration tests: `pytest tests/integration/`
- Manual testing with Postman/curl

#### Task 14: Deployment
- Build Docker image: `docker build -t facematric-backend .`
- Deploy to Render/Railway
- Configure environment variables
- Upload model files

#### Task 15: Final Validation
- Performance testing (<3s for core scoring)
- Security validation (no image persistence)
- Error scenario testing
- Production smoke tests

---

## 🧪 Testing Strategy

### Unit Tests
Test each component in isolation:
```bash
# Test specific component
pytest tests/unit/test_image_processor.py -v

# Test all components
pytest tests/unit/ -v

# With coverage
pytest tests/unit/ --cov=app/components
```

### Integration Tests
Test full API workflow:
```bash
# Test API endpoint
pytest tests/integration/test_api.py -v

# Test with real images
pytest tests/integration/ --use-real-images
```

### Manual Testing
```bash
# Start development server
uvicorn app.main:app --reload

# Test with curl
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "..."}'
```

---

## 📊 Development Workflow

### Recommended Daily Flow
1. Pick a task from the roadmap
2. Create the component file
3. Write the component logic
4. Write unit tests
5. Run tests: `pytest tests/unit/test_<component>.py -v`
6. Fix failing tests
7. Commit when tests pass
8. Move to next task

### Debugging Tips
- Use `pytest -v -s` to see print statements
- Use `pytest --pdb` to drop into debugger on failure
- Check logs in JSON format for structured debugging
- Test with real images early and often

---

## 🎯 Success Criteria

By the end of implementation, you should have:

✅ **8 fully functional components**:
- ImageProcessor
- FaceDetector
- SymmetryClassifier
- GoldenRatioScorer
- PercentileEngine
- CelebrityMatcher
- OverlayGenerator
- API Gateway (main.py)

✅ **Complete API**:
- `POST /api/analyze` endpoint
- CORS configured
- Timeout handling
- Error responses
- Health check

✅ **Celebrity Database**:
- 205 celebrity embeddings
- Generated from dataset
- Pickle file ready

✅ **Tests**:
- Unit tests for each component
- Integration tests for API
- Coverage > 80%

✅ **Deployment Ready**:
- Docker image builds
- Environment variables configured
- Model files included

---

## 🆘 Troubleshooting

### Common Issues

**1. MediaPipe import fails**
```bash
pip install --upgrade mediapipe
```

**2. TensorFlow/DeepFace issues on Windows**
```bash
pip install tensorflow==2.15.0 --no-cache-dir
```

**3. OpenCV missing system libraries**
```bash
# Windows: Install Visual C++ Redistributable
# Linux: sudo apt-get install libgl1-mesa-glx
```

**4. Out of memory during celebrity preprocessing**
- Process celebrities in batches
- Use `gc.collect()` after each celebrity
- Consider using GPU if available

---

## 📖 Additional Resources

### Documentation
- **FastAPI**: https://fastapi.tiangolo.com/
- **MediaPipe Face Mesh**: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
- **DeepFace**: https://github.com/serengil/deepface
- **Pydantic**: https://docs.pydantic.dev/

### Spec Documents
- `../.kiro/specs/facematric-golden-ratio-scoring/requirements.md`
- `../.kiro/specs/facematric-golden-ratio-scoring/design.md`
- `../.kiro/specs/facematric-golden-ratio-scoring/tasks.md`

---

## 🎉 Ready to Start?

Begin with **Task 2: Data Models**!

```bash
# Create your first file
touch app/models/internal.py

# Open in editor and start coding
code app/models/internal.py
```

Good luck building FaceMatric! 🚀
