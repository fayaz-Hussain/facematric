# Requirements Document

## Introduction

FaceMatric is a facial geometry analysis application that evaluates facial proportions against the golden ratio (φ = 1.618) and classifies face symmetry types. The system accepts user-submitted facial photographs, processes them to extract 468 facial landmarks, computes geometric ratios, and returns a quantitative golden ratio score (0-100), face symmetry type classification, percentile ranking, and optional celebrity look-alike matches. The application prioritizes privacy by processing images ephemerally without server-side storage.

## Glossary

- **Image_Processor**: Backend component responsible for accepting, normalizing, compressing, and resizing uploaded images to optimal format for analysis
- **Face_Detector**: MediaPipe Face Mesh component that identifies facial regions in images and extracts 468 3D landmark coordinates
- **Landmark_Extractor**: Subcomponent of Face_Detector that produces (x, y, z) coordinates for 468 facial points
- **Symmetry_Classifier**: Component that analyzes landmark geometry to determine face shape classification (oval, rectangle, round, square, heart, diamond)
- **Golden_Ratio_Scorer**: Component that computes deviation from φ (1.618) for 8 facial ratios and generates 0-100 scores
- **Percentile_Engine**: Component that converts golden ratio scores into population-relative percentile rankings
- **Celebrity_Matcher**: FaceNet-based component that generates face embeddings and compares them against precomputed celebrity database
- **Confidence_Threshold**: Minimum similarity score (0.55) required for celebrity match to be displayed
- **Overlay_Generator**: Component that produces visual data for displaying golden ratio guides and landmarks on user photos
- **API_Gateway**: FastAPI endpoint that orchestrates image processing, face detection, scoring, and response assembly
- **Frontend_Client**: React-based web interface for photo upload, camera capture, and results display
- **Timeout_Threshold**: Maximum processing duration (30 seconds) before request is aborted
- **Celebrity_Database**: Precomputed FaceNet embedding vectors for 205 celebrities (105 Western + 100 Bollywood)
- **User**: End-user submitting a facial photograph for analysis

## Requirements

### Requirement 1: Image Upload and Normalization

**User Story:** As a User, I want to upload facial photos in any common format, so that I can receive analysis without worrying about technical image specifications.

#### Acceptance Criteria

1. WHEN a User submits an image file, THE Image_Processor SHALL accept JPEG, PNG, HEIC, WebP, BMP, and GIF formats
2. WHEN an image is accepted, THE Image_Processor SHALL normalize the image to RGB color space
3. WHEN an image is normalized, THE Image_Processor SHALL resize the image maintaining aspect ratio to a maximum dimension of 640 pixels
4. WHEN an image is resized, THE Image_Processor SHALL compress the image to JPEG format at 85% quality
5. IF an image file is corrupted or unreadable, THEN THE Image_Processor SHALL return error code INVALID_IMAGE with message "Corrupt or unsupported file format"
6. WHEN an image has dimensions below 300x300 pixels, THE Image_Processor SHALL return error code POOR_QUALITY with message "Image resolution too low for accurate analysis"

### Requirement 2: Face Detection and Landmark Extraction

**User Story:** As a User, I want the system to accurately detect my face and extract precise measurements, so that my analysis results are reliable.

#### Acceptance Criteria

1. WHEN a normalized image is received, THE Face_Detector SHALL execute MediaPipe Face Mesh detection
2. WHEN exactly one face is detected, THE Landmark_Extractor SHALL extract 468 (x, y, z) landmark coordinates
3. IF no faces are detected, THEN THE Face_Detector SHALL return error code NO_FACE_DETECTED with message "We couldn't detect a face in your photo. Please take a clear, front-facing photo with good lighting."
4. IF multiple faces are detected, THEN THE Face_Detector SHALL return error code MULTIPLE_FACES with message "Multiple faces detected. Please upload a photo with only one person."
5. IF landmark confidence scores indicate poor image quality, THEN THE Face_Detector SHALL return error code POOR_QUALITY with message "Photo quality is too low. Please ensure your face is well-lit and facing the camera directly."

### Requirement 3: Occlusion Detection

**User Story:** As a User, I want to be notified if accessories are obscuring my face, so that I can retake the photo for accurate analysis.

#### Acceptance Criteria

1. WHEN facial landmarks are extracted, THE Face_Detector SHALL analyze landmark positions to detect occlusions
2. IF glasses, masks, or significant facial obstructions are detected, THEN THE Face_Detector SHALL return error code OCCLUSION_DETECTED with message "We detected accessories covering your face. Please upload a photo without glasses, masks, or other obstructions for accurate analysis."

### Requirement 4: Face Symmetry Type Classification

**User Story:** As a User, I want to know my face symmetry type, so that I can understand my facial geometry structure.

#### Acceptance Criteria

1. WHEN valid landmarks are extracted, THE Symmetry_Classifier SHALL compute face length to face width ratio
2. WHEN face length to face width ratio is approximately 1.5, THE Symmetry_Classifier SHALL classify the face as oval
3. WHEN face length significantly exceeds face width with angular jawline, THE Symmetry_Classifier SHALL classify the face as rectangle
4. WHEN face length approximately equals face width with soft angles, THE Symmetry_Classifier SHALL classify the face as round
5. WHEN face length approximately equals face width with strong angular jawline, THE Symmetry_Classifier SHALL classify the face as square
6. WHEN forehead width exceeds chin width by significant margin, THE Symmetry_Classifier SHALL classify the face as heart
7. WHEN cheekbone width exceeds both forehead width and chin width, THE Symmetry_Classifier SHALL classify the face as diamond
8. WHEN symmetry type is determined, THE Symmetry_Classifier SHALL return the classification label without associating any numeric score

### Requirement 5: Golden Ratio Proportion Measurement

**User Story:** As a User, I want my facial proportions measured against the golden ratio, so that I receive an objective geometric score.

#### Acceptance Criteria

1. WHEN valid landmarks are extracted, THE Golden_Ratio_Scorer SHALL compute the ratio of face length to face width
2. WHEN valid landmarks are extracted, THE Golden_Ratio_Scorer SHALL compute the ratio of face width to jaw width
3. WHEN valid landmarks are extracted, THE Golden_Ratio_Scorer SHALL compute the ratio of eye spacing to eye width
4. WHEN valid landmarks are extracted, THE Golden_Ratio_Scorer SHALL compute the ratio of nose width to mouth width
5. WHEN valid landmarks are extracted, THE Golden_Ratio_Scorer SHALL compute the ratio of mouth width to face width
6. WHEN valid landmarks are extracted, THE Golden_Ratio_Scorer SHALL compute the ratio of nose bridge to chin distance to hairline to nose bridge distance
7. WHEN valid landmarks are extracted, THE Golden_Ratio_Scorer SHALL compute the ratio of eye to eyebrow distance to eye height
8. WHEN valid landmarks are extracted, THE Golden_Ratio_Scorer SHALL compute the ratio of chin width to nose width

### Requirement 6: Per-Feature Golden Ratio Scoring

**User Story:** As a User, I want to see individual scores for each facial ratio, so that I understand which features align closely with the golden ratio.

#### Acceptance Criteria

1. FOR each of the 8 facial ratios, THE Golden_Ratio_Scorer SHALL compute a score using the formula: score = 100 × (1 - |measured_ratio - 1.618| / 1.618)
2. WHEN a measured ratio equals 1.618, THE Golden_Ratio_Scorer SHALL assign a score of 100 for that feature
3. WHEN a measured ratio deviates from 1.618, THE Golden_Ratio_Scorer SHALL assign a score proportional to the deviation magnitude
4. WHEN per-feature scores are computed, THE Golden_Ratio_Scorer SHALL record the measured ratio, ideal ratio (1.618), score, and deviation for each feature
5. WHEN per-feature scores are computed, THE Golden_Ratio_Scorer SHALL ensure all scores are bounded between 0 and 100

### Requirement 7: Overall Golden Ratio Score Computation

**User Story:** As a User, I want a single overall score summarizing my facial proportions, so that I can quickly understand my golden ratio alignment.

#### Acceptance Criteria

1. WHEN all 8 per-feature scores are computed, THE Golden_Ratio_Scorer SHALL calculate the overall score as the arithmetic mean of the 8 feature scores
2. WHEN the overall score is calculated, THE Golden_Ratio_Scorer SHALL round the result to one decimal place
3. WHEN the overall score is calculated, THE Golden_Ratio_Scorer SHALL ensure the score is bounded between 0 and 100

### Requirement 8: Percentile Ranking

**User Story:** As a User, I want to see how my score compares to the general population, so that I can contextualize my numeric score.

#### Acceptance Criteria

1. WHEN an overall golden ratio score is computed, THE Percentile_Engine SHALL convert the score to a percentile rank
2. WHEN percentile rank is determined, THE Percentile_Engine SHALL return an integer value representing the percentage of the population scoring below the User
3. WHEN percentile rank is returned, THE Percentile_Engine SHALL format the result as "You score higher than X% of people"

### Requirement 9: Visual Overlay Generation

**User Story:** As a User, I want to see golden ratio guides overlaid on my photo, so that I can visually understand the geometric analysis.

#### Acceptance Criteria

1. WHEN facial landmarks are extracted, THE Overlay_Generator SHALL prepare the 468 landmark coordinates for client-side rendering
2. WHEN landmarks are prepared, THE Overlay_Generator SHALL compute horizontal third divisions for the golden ratio
3. WHEN horizontal thirds are computed, THE Overlay_Generator SHALL compute the vertical midline coordinate
4. WHEN midline is computed, THE Overlay_Generator SHALL compute the phi spiral center coordinates
5. WHEN all overlay data is computed, THE Overlay_Generator SHALL return a structured object containing landmarks and golden ratio guide coordinates

### Requirement 10: Celebrity Face Embedding Generation

**User Story:** As a User, I want my face compared to celebrities, so that I can discover look-alike matches.

#### Acceptance Criteria

1. WHEN a User photo contains a valid face, THE Celebrity_Matcher SHALL generate a FaceNet embedding vector for the User face
2. WHEN the User embedding is generated, THE Celebrity_Matcher SHALL compute cosine similarity scores between the User embedding and all Celebrity_Database embeddings
3. WHEN similarity scores are computed, THE Celebrity_Matcher SHALL rank celebrities by similarity score in descending order

### Requirement 11: Celebrity Match Confidence Thresholding

**User Story:** As a User, I want to see celebrity matches only when confidence is sufficiently high, so that I receive meaningful comparisons.

#### Acceptance Criteria

1. WHEN celebrity similarity scores are ranked, THE Celebrity_Matcher SHALL identify the highest-scoring celebrity
2. IF the highest similarity score is greater than or equal to 0.55, THEN THE Celebrity_Matcher SHALL return the celebrity name, confidence score, dataset source, and thumbnail URL
3. IF the highest similarity score is less than 0.55, THEN THE Celebrity_Matcher SHALL return a null name value with message "No close match found in our dataset"
4. WHEN a celebrity match is returned, THE Celebrity_Matcher SHALL include the confidence score as a decimal value rounded to two decimal places

### Requirement 12: Processing Timeout Handling

**User Story:** As a User, I want to be notified if processing takes too long, so that I can retry with a different photo.

#### Acceptance Criteria

1. WHEN image analysis begins, THE API_Gateway SHALL start a timeout timer set to 30 seconds
2. IF processing completes before 30 seconds elapse, THEN THE API_Gateway SHALL return results with HTTP status 200
3. IF processing exceeds 30 seconds, THEN THE API_Gateway SHALL abort processing and return HTTP status 408
4. WHEN a timeout occurs, THE API_Gateway SHALL return error code PROCESSING_TIMEOUT with message "Processing took too long. Please try again with a clearer photo."

### Requirement 13: Privacy-Preserving Image Processing

**User Story:** As a User, I want assurance that my photos are not stored, so that my privacy is protected.

#### Acceptance Criteria

1. WHEN an image is uploaded, THE API_Gateway SHALL process the image in memory without writing to persistent storage
2. WHEN analysis is complete, THE API_Gateway SHALL discard the image from memory
3. WHEN an error occurs during processing, THE API_Gateway SHALL discard the image from memory
4. WHEN analysis is complete, THE API_Gateway SHALL log only metadata (timestamp, processing time, error codes) without logging image data or face embeddings

### Requirement 14: API Response Structure for Successful Analysis

**User Story:** As a Frontend_Client, I want a structured JSON response with all analysis results, so that I can render comprehensive results to the User.

#### Acceptance Criteria

1. WHEN analysis completes successfully, THE API_Gateway SHALL return HTTP status 200 with Content-Type application/json
2. WHEN a successful response is constructed, THE API_Gateway SHALL include status field with value "success"
3. WHEN a successful response is constructed, THE API_Gateway SHALL include processing_time_ms field with integer milliseconds elapsed
4. WHEN a successful response is constructed, THE API_Gateway SHALL include a results object containing golden_ratio_score, percentile_rank, face_symmetry_type, feature_breakdown array, overlay_data object, and celebrity_match object
5. WHEN feature_breakdown is included, THE API_Gateway SHALL provide 8 feature objects each containing feature name, measured_ratio, ideal_ratio (1.618), score, and deviation
6. WHEN overlay_data is included, THE API_Gateway SHALL provide landmarks array and golden_ratio_guides object
7. WHEN a celebrity match meets the confidence threshold, THE API_Gateway SHALL include celebrity name, confidence, dataset_source, and thumbnail_url
8. WHEN no celebrity match meets the threshold, THE API_Gateway SHALL include celebrity_match object with null name and explanatory message

### Requirement 15: API Response Structure for Errors

**User Story:** As a Frontend_Client, I want structured error responses with actionable guidance, so that I can help Users resolve issues.

#### Acceptance Criteria

1. WHEN an error occurs during processing, THE API_Gateway SHALL return an HTTP status code appropriate to the error type (400 for client errors, 408 for timeout)
2. WHEN an error response is constructed, THE API_Gateway SHALL include status field with value "error"
3. WHEN an error response is constructed, THE API_Gateway SHALL include error_code field with a machine-readable error identifier
4. WHEN an error response is constructed, THE API_Gateway SHALL include message field with user-friendly guidance
5. WHEN an error is recoverable, THE API_Gateway SHALL include retry field with boolean value true

### Requirement 16: Camera Capture Support

**User Story:** As a User, I want to take a photo directly with my camera, so that I can analyze my face without pre-existing photos.

#### Acceptance Criteria

1. WHEN a User accesses the Frontend_Client, THE Frontend_Client SHALL provide a camera capture interface option
2. WHEN the User activates camera capture, THE Frontend_Client SHALL request camera permissions from the browser
3. WHEN camera permissions are granted, THE Frontend_Client SHALL display a live camera preview
4. WHEN the User captures a photo, THE Frontend_Client SHALL convert the captured image to a format accepted by Image_Processor
5. WHEN the captured image is converted, THE Frontend_Client SHALL submit the image to the API_Gateway using the same upload flow as file uploads

### Requirement 17: Progressive Degradation for Celebrity Matching

**User Story:** As a User, I want to receive golden ratio scores even if celebrity matching fails, so that partial service outages do not block my core analysis.

#### Acceptance Criteria

1. WHEN Celebrity_Matcher encounters an error during embedding generation, THE API_Gateway SHALL continue processing and return golden ratio scores
2. WHEN Celebrity_Matcher fails, THE API_Gateway SHALL set celebrity_match field to null in the response
3. WHEN Celebrity_Matcher fails, THE API_Gateway SHALL log the error for debugging without exposing technical details to the User
4. WHEN golden ratio scoring succeeds but celebrity matching fails, THE API_Gateway SHALL return HTTP status 200 with partial results

### Requirement 18: Celebrity Database Precomputation

**User Story:** As a System Administrator, I want celebrity embeddings precomputed at deployment time, so that runtime analysis is fast and does not regenerate embeddings.

#### Acceptance Criteria

1. WHEN the Celebrity_Database is initialized, THE Celebrity_Matcher SHALL load precomputed FaceNet embedding vectors from celebrity_embeddings.pkl
2. WHEN the Celebrity_Database contains multiple photos of the same celebrity, THE Celebrity_Matcher SHALL use the averaged embedding vector for that celebrity
3. WHEN the Celebrity_Database is loaded, THE Celebrity_Matcher SHALL verify that 205 celebrity embeddings are present (105 Western + 100 Bollywood)
4. IF the Celebrity_Database file is missing or corrupted at startup, THEN THE API_Gateway SHALL log a critical error and disable celebrity matching functionality while allowing golden ratio scoring to proceed

### Requirement 19: Asymmetric Feature Handling

**User Story:** As a User with asymmetric facial features, I want my score to reflect only the affected features, so that one asymmetric feature does not unfairly penalize my overall score.

#### Acceptance Criteria

1. WHEN computing ratios for bilateral features (left eye vs right eye), THE Golden_Ratio_Scorer SHALL compute separate measurements for each side
2. WHEN bilateral measurements differ significantly, THE Golden_Ratio_Scorer SHALL score each side independently
3. WHEN per-feature scores are computed, THE Golden_Ratio_Scorer SHALL include asymmetric features in the overall score calculation using their individual scores
4. WHEN feature_breakdown is returned, THE Golden_Ratio_Scorer SHALL display asymmetric bilateral features as separate entries if they differ by more than 10%

### Requirement 20: Image Format Conversion Edge Cases

**User Story:** As a User uploading images from various devices, I want format conversions to handle edge cases gracefully, so that my uploads succeed consistently.

#### Acceptance Criteria

1. WHEN an image has EXIF orientation metadata, THE Image_Processor SHALL rotate the image according to EXIF orientation before analysis
2. WHEN an image is in CMYK color space, THE Image_Processor SHALL convert to RGB color space
3. WHEN an image contains an alpha channel (transparency), THE Image_Processor SHALL composite the image onto a white background before analysis
4. WHEN an image has an aspect ratio more extreme than 3:1 or 1:3, THE Image_Processor SHALL return error code POOR_QUALITY with message "Image aspect ratio too extreme for accurate face detection"

