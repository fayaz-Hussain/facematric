# Implementation Plan: FaceMatric Golden Ratio Scoring

## Overview

This task list implements FaceMatric - a facial geometry analysis application that evaluates facial proportions against the golden ratio (φ = 1.618). The system consists of a Python FastAPI backend with 8 core processing components and JavaScript integration for 3 pre-built static HTML frontend screens. Implementation follows a bottom-up approach: infrastructure → core components → API integration → frontend wiring → testing → deployment.

## Tasks

- [-] 1. Backend Infrastructure Setup
  - Create Python FastAPI project structure with standard directories (app/, tests/, data/)
  - Install core dependencies: FastAPI, Uvicorn, Pydantic, NumPy, Pillow, MediaPipe, SciPy, DeepFace, OpenCV
  - Configure logging infrastructure with structured logging (JSON format, appropriate log levels)
  - Create base error classes (ValidationError, ErrorCode enum) for standardized error handling
  - Set up environment variable management (.env file support with python-dotenv)
  - Create Dockerfile for containerized deployment
  - _Requirements: 1.1, 1.2, 1.5, 1.6, 2.1, 2.3, 2.4, 2.5, 3.2, 12.1, 12.2, 12.3, 12.4, 13.1, 13.2, 13.3, 13.4, 15.1, 15.2, 15.3, 15.4, 15.5, 18.4, 20.1, 20.2, 20.3, 20.4_

- [ ] 2. Data Models and Schemas
  - [-] 2.1 Create Pydantic request models (AnalyzeRequest with image_base64 and include_celebrity_match fields)
    - Define base model with field validation and examples
    - Add support for both base64 string and multipart file upload
    - _Requirements: 14.1, 14.2_

  - [-] 2.2 Create Pydantic response models (AnalysisResponse, AnalysisResultModel, FeatureBreakdownItem, etc.)
    - Implement success response schema with all required nested models
    - Implement error response schema with error codes and messages
    - Add JSON schema examples for API documentation
    - _Requirements: 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 15.1, 15.2, 15.3, 15.4, 15.5_

  - [-] 2.3 Create internal data structures (ImageData, Landmarks, SymmetryType, GoldenRatioResult, etc.)
    - Define dataclasses for internal component communication
    - Ensure type safety and validation at component boundaries
    - _Requirements: 1.2, 1.3, 2.2, 4.8, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 6.4, 7.1, 7.2, 7.3_

- [ ]* 2.4 Write unit tests for data model validation
    - Test request model validation (valid/invalid inputs, field constraints)
    - Test response model serialization (nested structures, optional fields)
    - Test dataclass construction and type validation
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 15.1, 15.2, 15.3, 15.4, 15.5_

- [ ] 3. Image Processor Component
  - [~] 3.1 Implement format detection and validation (JPEG, PNG, HEIC, WebP, BMP, GIF)
    - Use PIL to open and validate image formats
    - Detect corrupt files and raise INVALID_IMAGE error
    - _Requirements: 1.1, 1.5_

  - [~] 3.2 Implement EXIF rotation handling using ImageOps.exif_transpose
    - Apply rotation based on EXIF orientation metadata
    - Handle all 8 EXIF orientation values
    - _Requirements: 20.1_

  - [~] 3.3 Implement color space normalization (CMYK→RGB, RGBA→RGB with white background)
    - Convert CMYK images to RGB
    - Composite RGBA images onto white background using alpha channel
    - Handle grayscale and other color modes
    - _Requirements: 1.2, 20.2, 20.3_

  - [~] 3.4 Implement aspect ratio-preserving resize (max 640px dimension, Lanczos resampling)
    - Calculate scale factor to fit within 640px maximum dimension
    - Apply high-quality Lanczos resampling
    - Maintain original aspect ratio
    - _Requirements: 1.3_

  - [~] 3.5 Implement JPEG compression (85% quality) and validation checks
    - Compress to JPEG at 85% quality
    - Validate minimum resolution (300x300px) and reject POOR_QUALITY
    - Validate aspect ratio (reject if >3:1) and raise POOR_QUALITY error
    - _Requirements: 1.4, 1.6, 20.4_

  - [ ]* 3.6 Write unit tests for ImageProcessor
    - Test each supported format conversion
    - Test EXIF rotation for all 8 orientations
    - Test color space conversions (CMYK, RGBA, grayscale)
    - Test resize algorithm with various aspect ratios
    - Test validation errors (corrupt files, low resolution, extreme aspect ratio)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 20.1, 20.2, 20.3, 20.4_

- [ ] 4. Face Detector and Landmark Extractor Component
  - [~] 4.1 Initialize MediaPipe Face Mesh with correct configuration
    - Load face_landmarker.task model file
    - Configure with max_num_faces=1, min_detection_confidence=0.5
    - Set up running mode for single image processing
    - _Requirements: 2.1_

  - [~] 4.2 Implement face detection with count validation
    - Process image through MediaPipe Face Mesh
    - Raise NO_FACE_DETECTED if zero faces found
    - Raise MULTIPLE_FACES if more than one face found
    - _Requirements: 2.1, 2.3, 2.4_

  - [~] 4.3 Extract 468 3D landmark coordinates and validate confidence
    - Extract (x, y, z) coordinates for all 468 landmarks
    - Calculate average landmark confidence
    - Raise POOR_QUALITY if confidence < 0.6 threshold
    - _Requirements: 2.2, 2.5_

  - [~] 4.4 Implement occlusion detection for glasses and masks
    - Analyze eye region landmark patterns for glasses detection
    - Analyze mouth region visibility for mask detection
    - Raise OCCLUSION_DETECTED if significant obstruction found
    - _Requirements: 3.1, 3.2_

  - [ ]* 4.5 Write unit tests for FaceDetector
    - Test with single face images (should succeed)
    - Test with no face images (should raise NO_FACE_DETECTED)
    - Test with multiple face images (should raise MULTIPLE_FACES)
    - Test with poor lighting (should raise POOR_QUALITY)
    - Test occlusion detection (glasses, masks)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2_

- [ ] 5. Symmetry Classifier Component
  - [~] 5.1 Implement facial dimension measurements (face length, width, jaw width, etc.)
    - Extract key landmark indices for face boundaries
    - Calculate Euclidean distances between landmark pairs
    - Compute jawline angle using three-point angle calculation
    - _Requirements: 4.1_

  - [~] 5.2 Implement symmetry type classification logic with decision tree
    - Classify as oval: length/width ≈ 1.5, rounded jawline
    - Classify as rectangle: length > width, angular jawline
    - Classify as round: length ≈ width, soft angles
    - Classify as square: length ≈ width, angular jawline
    - Classify as heart: forehead > chin width significantly
    - Classify as diamond: cheekbone width > forehead and chin
    - Return classification label without numeric score
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8_

  - [ ]* 5.3 Write unit tests for SymmetryClassifier
    - Test classification for each of 6 symmetry types
    - Test edge cases (borderline ratios between types)
    - Verify no numeric scores are assigned
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8_

- [ ] 6. Golden Ratio Scorer Component
  - [~] 6.1 Implement 8 facial ratio measurement functions
    - Face length ÷ face width (landmarks 10, 152, 234, 454)
    - Face width ÷ jaw width (landmarks 234, 454, 172, 397)
    - Eye spacing ÷ eye width (landmarks 133, 362, 33)
    - Nose width ÷ mouth width (landmarks 219, 439, 61, 291)
    - Mouth width ÷ face width (landmarks 61, 291, 234, 454)
    - Vertical thirds: nose to chin ÷ hairline to nose (landmarks 10, 6, 152)
    - Eye to brow distance ÷ eye height (landmarks 159, 145, 70)
    - Chin width ÷ nose width (landmarks 172, 397, 219, 439)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8_

  - [~] 6.2 Implement per-feature scoring with golden ratio formula
    - Apply formula: score = 100 × (1 - |ratio - 1.618| / 1.618)
    - Clamp scores to [0, 100] range
    - Record measured_ratio, ideal_ratio (1.618), score, and deviation
    - Assign score of 100 when ratio equals 1.618 exactly
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [~] 6.3 Implement overall score calculation as arithmetic mean of 8 features
    - Sum all 8 feature scores
    - Divide by 8 to get arithmetic mean
    - Round to one decimal place
    - Ensure result is bounded [0, 100]
    - _Requirements: 7.1, 7.2, 7.3_

  - [~] 6.4 Implement asymmetric feature handling for bilateral measurements
    - Measure left and right sides separately for bilateral features
    - Compare measurements and detect asymmetry (>10% difference)
    - Score each side independently if asymmetric
    - Include separate entries in feature breakdown for asymmetric features
    - _Requirements: 19.1, 19.2, 19.3, 19.4_

  - [ ]* 6.5 Write unit tests for GoldenRatioScorer
    - Test each of 8 ratio measurement functions
    - Test scoring formula correctness (verify score = 100 when ratio = 1.618)
    - Test score clamping (0-100 bounds)
    - Test overall score calculation (mean of 8 features)
    - Test asymmetric feature detection and scoring
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 6.1, 6.2, 6.3, 6.4, 6.5, 7.1, 7.2, 7.3, 19.1, 19.2, 19.3, 19.4_

- [ ] 7. Percentile Engine Component
  - [~] 7.1 Implement bootstrap normal distribution (mean=70, std=10)
    - Initialize scipy.stats.norm distribution with bootstrap parameters
    - Document as placeholder until real data collected
    - _Requirements: 8.1_

  - [~] 7.2 Implement score-to-percentile conversion using CDF
    - Clamp input score to [0, 100] range
    - Calculate percentile using distribution CDF
    - Round to integer and ensure [0, 100] bounds
    - Format as "You score higher than X% of people"
    - _Requirements: 8.1, 8.2, 8.3_

  - [~] 7.3 Implement empirical distribution loading (future enhancement)
    - Create method to load real user scores
    - Fit normal distribution to empirical data
    - Update mean and std parameters
    - Require minimum 100 scores before updating
    - _Requirements: 8.1_

  - [ ]* 7.4 Write unit tests for PercentileEngine
    - Test percentile calculation accuracy with known scores
    - Test edge cases (score=0, score=100)
    - Test bounds enforcement
    - _Requirements: 8.1, 8.2, 8.3_

- [ ] 8. Celebrity Matcher Component
  - [~] 8.1 Implement FaceNet embedding generation using DeepFace
    - Initialize DeepFace with Facenet model
    - Generate 128-dimensional embedding vector for input face
    - Handle embedding generation failures gracefully
    - _Requirements: 10.1_

  - [~] 8.2 Implement cosine similarity calculation
    - Compute dot product of embedding vectors
    - Calculate vector norms
    - Return cosine similarity in [-1, 1] range
    - _Requirements: 10.2_

  - [~] 8.3 Implement celebrity database comparison and ranking
    - Load precomputed celebrity embeddings from pickle file
    - Compute similarity with all celebrities
    - Rank by similarity score in descending order
    - _Requirements: 10.2, 10.3_

  - [~] 8.4 Implement confidence thresholding (0.55 threshold)
    - Identify highest-scoring celebrity
    - Return match if confidence ≥ 0.55 with name, confidence, dataset_source, thumbnail_url
    - Return null name with message if confidence < 0.55
    - Round confidence to 2 decimal places
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

  - [~] 8.5 Implement celebrity database loading with error handling
    - Load celebrity_embeddings.pkl at startup
    - Log critical error and disable matching if file missing/corrupted
    - Allow golden ratio scoring to proceed even if celebrity matching disabled
    - _Requirements: 18.1, 18.2, 18.3, 18.4_

  - [ ]* 8.6 Write unit tests for CelebrityMatcher
    - Test embedding generation (verify 128 dimensions)
    - Test cosine similarity calculation
    - Test confidence thresholding (≥0.55 vs <0.55)
    - Test with empty/missing celebrity database
    - _Requirements: 10.1, 10.2, 10.3, 11.1, 11.2, 11.3, 11.4, 18.1, 18.2, 18.3, 18.4_

- [ ] 9. Overlay Generator Component
  - [~] 9.1 Implement 2D landmark extraction from 3D landmarks
    - Convert 468 3D landmarks to 2D by dropping z-coordinate
    - Format as list of [x, y] coordinate pairs
    - _Requirements: 9.1_

  - [~] 9.2 Implement golden ratio guide calculations
    - Calculate vertical midline (center x-coordinate)
    - Calculate horizontal thirds using golden ratio divisions
    - Calculate phi spiral center (near nose bridge)
    - Ensure all coordinates within image bounds
    - _Requirements: 9.2, 9.3, 9.4, 9.5_

  - [ ]* 9.3 Write unit tests for OverlayGenerator
    - Test landmark count (should be exactly 468 2D points)
    - Test guide coordinate bounds (within image dimensions)
    - Test golden ratio divisions correctness
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [ ] 10. API Gateway and Orchestration
  - [~] 10.1 Create FastAPI application with CORS middleware
    - Initialize FastAPI app with title and version
    - Configure CORS for allowed origins (production + localhost)
    - Set up logging and error handlers
    - _Requirements: 14.1_

  - [~] 10.2 Implement /api/analyze endpoint with request validation
    - Accept multipart file upload or base64 image string
    - Validate request parameters
    - Parse include_celebrity_match flag
    - _Requirements: 14.1, 14.2_

  - [~] 10.3 Implement timeout middleware (30-second threshold)
    - Wrap processing in asyncio.wait_for with 30s timeout
    - Return HTTP 408 on timeout with PROCESSING_TIMEOUT error
    - Include retry flag in timeout response
    - _Requirements: 12.1, 12.2, 12.3, 12.4_

  - [~] 10.4 Implement component orchestration in _process_analysis
    - Call ImageProcessor to normalize image
    - Call FaceDetector to extract landmarks
    - Run SymmetryClassifier, GoldenRatioScorer, OverlayGenerator in parallel (asyncio.create_task)
    - Run CelebrityMatcher asynchronously (optional, can fail)
    - Call PercentileEngine to compute percentile
    - Assemble final AnalysisResult
    - _Requirements: 14.3, 14.4, 14.5, 14.6, 14.7, 14.8_

  - [~] 10.5 Implement progressive degradation for celebrity matching failures
    - Catch exceptions from CelebrityMatcher
    - Log warning but continue processing
    - Set celebrity_match to null in response
    - Return HTTP 200 with partial results
    - _Requirements: 17.1, 17.2, 17.3, 17.4_

  - [~] 10.6 Implement comprehensive error handling with user-friendly messages
    - Catch ValidationError and return HTTP 400 with error details
    - Catch timeout and return HTTP 408
    - Catch unexpected errors and return HTTP 500
    - Log errors with metadata (no image data)
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5_

  - [~] 10.7 Implement privacy-preserving memory cleanup
    - Process images in-memory only (no disk writes)
    - Delete image data after processing
    - Force garbage collection
    - Log only metadata (timestamp, processing time, error codes)
    - _Requirements: 13.1, 13.2, 13.3, 13.4_

  - [ ]* 10.8 Write integration tests for API endpoint
    - Test happy path: upload valid image → receive complete analysis
    - Test NO_FACE_DETECTED error
    - Test MULTIPLE_FACES error
    - Test POOR_QUALITY error
    - Test OCCLUSION_DETECTED error
    - Test INVALID_IMAGE error
    - Test timeout scenario
    - Test progressive degradation (celebrity matching fails)
    - _Requirements: 1.5, 1.6, 2.3, 2.4, 2.5, 3.2, 12.4, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 15.1, 15.2, 15.3, 15.4, 15.5, 17.1, 17.2, 17.3, 17.4_

- [~] 11. Checkpoint - Core backend complete
  - Ensure all backend components pass unit tests
  - Verify API integration tests pass
  - Test end-to-end with Postman/curl before proceeding to frontend

- [ ] 12. Celebrity Database Preprocessing
  - [~] 12.1 Merge Pins Face Recognition and Bollywood datasets
    - Combine celebrity folders into single celebrity_dataset/ directory
    - Verify no overlapping celebrities between datasets
    - Remove "pins_" prefix from celebrity folder names
    - _Requirements: 18.3_

  - [~] 12.2 Generate FaceNet embeddings for all celebrity images
    - Iterate through all celebrity folders and images
    - Generate FaceNet embedding for each image using DeepFace
    - Handle detection failures gracefully (skip bad images)
    - Log progress and errors
    - _Requirements: 10.1, 18.1_

  - [~] 12.3 Average embeddings per celebrity and save to pickle file
    - Average multiple embeddings for same celebrity
    - Create dictionary: {celebrity_name: {embedding, dataset_source, thumbnail_path}}
    - Save to celebrity_embeddings.pkl using pickle
    - Verify 205 celebrities (105 Western + 100 Bollywood)
    - _Requirements: 18.1, 18.2, 18.3_

- [ ] 13. Frontend JavaScript Integration
  - [~] 13.1 Create api-client.js with API communication layer
    - Implement FaceMetricAPI class with analyzeFace method
    - Handle base64 image encoding/decoding
    - Implement APIError class for structured error handling
    - Configure API_BASE_URL (localhost for dev, production URL for deployment)
    - _Requirements: 14.1, 14.2, 15.1, 15.2, 15.3, 15.4, 15.5_

  - [~] 13.2 Create upload-handler.js for upload screen
    - Implement file upload button handler
    - Implement camera capture with MediaDevices API
    - Implement drag-and-drop zone with event listeners
    - Validate file size (10MB max) and format
    - Read file as base64 and store in sessionStorage
    - Navigate to analyzing screen after upload
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5_

  - [~] 13.3 Create analyzing-handler.js for analyzing screen
    - Display uploaded image from sessionStorage
    - Update progress steps with icons and animations
    - Call API analyzeFace method
    - Update progress bar (0% → 60% → 80% → 100%)
    - Handle API errors with user-friendly error display
    - Store analysis results in sessionStorage
    - Navigate to results screen on success
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 15.1, 15.2, 15.3, 15.4, 15.5_

  - [~] 13.4 Create results-handler.js for results screen
    - Load analysis results from sessionStorage
    - Update overall golden ratio score with circular progress
    - Update face symmetry type classification
    - Update feature breakdown with progress bars and ratios
    - Update percentile ranking display
    - Update celebrity match with photo, name, and confidence
    - Handle no celebrity match case (display message)
    - Display symmetry overlay on uploaded photo
    - Implement "NEW ANALYSIS" button to clear sessionStorage and restart
    - _Requirements: 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 11.2, 11.3_

  - [~] 13.5 Add script tags to existing HTML files
    - Add api-client.js and upload-handler.js to upload screen HTML
    - Add api-client.js and analyzing-handler.js to analyzing screen HTML
    - Add api-client.js and results-handler.js to results screen HTML
    - _Requirements: 14.1, 14.2, 16.1, 16.2, 16.3, 16.4, 16.5_

  - [ ]* 13.6 Test frontend integration end-to-end
    - Test file upload flow (select file → analyze → results)
    - Test camera capture flow (camera → capture → analyze → results)
    - Test drag-and-drop upload
    - Test error handling (display error messages)
    - Test celebrity match display (both match and no-match cases)
    - Test "NEW ANALYSIS" flow
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 15.1, 15.2, 15.3, 15.4, 15.5, 16.1, 16.2, 16.3, 16.4, 16.5, 11.2, 11.3_

- [ ] 14. Deployment Configuration
  - [~] 14.1 Configure backend deployment (Render/Railway)
    - Create requirements.txt with all dependencies
    - Configure Dockerfile with Python base image and system dependencies
    - Set environment variables (ALLOWED_ORIGINS, PROCESSING_TIMEOUT, CELEBRITY_DB_PATH, etc.)
    - Upload celebrity_embeddings.pkl and face_landmarker.task to data/ directory
    - Configure uvicorn startup command
    - _Requirements: 12.1, 12.2, 13.1, 13.2, 13.3, 13.4, 18.1, 18.4_

  - [~] 14.2 Configure frontend deployment (Vercel/Netlify)
    - Create index.html redirect to upload screen
    - Create vercel.json with routes and CORS headers
    - Update API_BASE_URL in api-client.js to production backend URL
    - Deploy static HTML/CSS/JS files (no build step needed)
    - _Requirements: 14.1_

  - [~] 14.3 Test production deployment end-to-end
    - Verify CORS configuration allows frontend domain
    - Test upload → analyze → results flow in production
    - Verify celebrity embeddings loading correctly
    - Test error handling in production
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 18.1, 18.4_

- [ ] 15. Final Testing and Validation
  - [ ]* 15.1 Create diverse test dataset
    - Collect photos with different ages, ethnicities, genders, face shapes
    - Include quality variations (good/moderate/poor lighting)
    - Include edge cases (angles, occlusions, multiple faces, no faces)
    - _Requirements: All requirements for comprehensive validation_

  - [ ]* 15.2 Run end-to-end validation tests
    - Test all 6 face symmetry type classifications
    - Verify golden ratio scores are in reasonable range (60-85 for most)
    - Test celebrity matching with known celebrities (high confidence expected)
    - Verify no perfect 100 scores (slight variance is expected)
    - Test all error scenarios with appropriate user guidance
    - _Requirements: 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 6.1, 6.2, 11.2, 11.3_

  - [ ]* 15.3 Performance testing
    - Measure processing time for golden ratio scoring (target <3s)
    - Measure total time with celebrity matching (target <5s)
    - Verify timeout handling if processing exceeds 30s
    - _Requirements: 12.1, 12.2, 12.4_

  - [ ]* 15.4 Security and privacy validation
    - Verify no images written to disk
    - Verify logs contain only metadata (no image data or embeddings)
    - Verify memory cleanup after processing
    - Test CORS restrictions
    - _Requirements: 13.1, 13.2, 13.3, 13.4_

## Notes

- Tasks marked with `*` are optional test tasks and can be skipped for faster MVP delivery
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation before moving to next phase
- Frontend integration reuses existing HTML/CSS - only JavaScript needed
- Celebrity database preprocessing is one-time setup before deployment
- All backend components must pass unit tests before integration
- Progressive degradation ensures celebrity matching failures don't block core features

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1", "2.1", "2.2", "2.3"] },
    { "id": 1, "tasks": ["2.4", "3.1", "3.2", "3.3", "3.4", "3.5"] },
    { "id": 2, "tasks": ["3.6", "4.1", "4.2", "4.3", "4.4"] },
    { "id": 3, "tasks": ["4.5", "5.1", "5.2"] },
    { "id": 4, "tasks": ["5.3", "6.1", "6.2", "6.3", "6.4"] },
    { "id": 5, "tasks": ["6.5", "7.1", "7.2", "7.3"] },
    { "id": 6, "tasks": ["7.4", "8.1", "8.2", "8.3", "8.4", "8.5"] },
    { "id": 7, "tasks": ["8.6", "9.1", "9.2"] },
    { "id": 8, "tasks": ["9.3", "10.1", "10.2", "10.3", "10.4", "10.5", "10.6", "10.7"] },
    { "id": 9, "tasks": ["10.8", "11"] },
    { "id": 10, "tasks": ["12.1", "12.2"] },
    { "id": 11, "tasks": ["12.3"] },
    { "id": 12, "tasks": ["13.1", "13.2", "13.3", "13.4", "13.5"] },
    { "id": 13, "tasks": ["13.6", "14.1", "14.2"] },
    { "id": 14, "tasks": ["14.3", "15.1"] },
    { "id": 15, "tasks": ["15.2", "15.3", "15.4"] }
  ]
}
```
