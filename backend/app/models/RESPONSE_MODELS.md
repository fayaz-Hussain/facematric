# Response Models Documentation

This document provides comprehensive documentation for the FaceMatric API response models.

## Overview

The FaceMatric API uses Pydantic v2 models for request/response validation and JSON schema generation. All response models are located in `app/models/responses.py`.

## Model Hierarchy

```
AnalysisResponse
├── status: str
├── processing_time_ms: int
└── results: AnalysisResultModel
    ├── golden_ratio_score: float
    ├── percentile_rank: int
    ├── face_symmetry_type: SymmetryTypeEnum
    ├── feature_breakdown: List[FeatureBreakdownItem]
    │   ├── feature: str
    │   ├── measured_ratio: float
    │   ├── ideal_ratio: float
    │   ├── score: float
    │   └── deviation: float
    ├── overlay_data: OverlayData
    │   ├── landmarks: List[List[float]]
    │   └── golden_ratio_guides: GoldenRatioGuides
    │       ├── vertical_midline: float
    │       ├── horizontal_thirds: List[float]
    │       └── phi_spiral_center: List[float]
    └── celebrity_match: Optional[CelebrityMatch]
        ├── name: Optional[str]
        ├── confidence: Optional[float]
        ├── dataset_source: Optional[str]
        ├── thumbnail_url: Optional[str]
        └── message: Optional[str]

ErrorResponse
├── status: str
├── error_code: ErrorCodeEnum
├── message: str
├── retry: bool
└── details: Optional[Dict[str, Any]]
```

## Response Models

### AnalysisResponse

**Description:** Main success response for facial analysis.

**Fields:**
- `status` (str): Always "success" for successful responses
- `processing_time_ms` (int): Total processing time in milliseconds
- `results` (AnalysisResultModel): Complete analysis results

**Requirements Coverage:**
- Requirement 14.1: HTTP status 200 with Content-Type application/json
- Requirement 14.2: Status field with value "success"
- Requirement 14.3: Processing time field

**Example:**
```json
{
  "status": "success",
  "processing_time_ms": 2847,
  "results": { ... }
}
```

### AnalysisResultModel

**Description:** Complete facial analysis results.

**Fields:**
- `golden_ratio_score` (float, 0-100): Overall golden ratio score
- `percentile_rank` (int, 0-100): Population percentile ranking
- `face_symmetry_type` (SymmetryTypeEnum): Face shape classification
- `feature_breakdown` (List[FeatureBreakdownItem], length=8): Individual feature scores
- `overlay_data` (OverlayData): Visual overlay rendering data
- `celebrity_match` (Optional[CelebrityMatch]): Celebrity look-alike match

**Requirements Coverage:**
- Requirement 14.4: Results object with all required fields
- Requirement 14.5: Feature breakdown array with 8 items
- Requirement 14.6: Overlay data object
- Requirement 14.7: Celebrity match when confidence meets threshold
- Requirement 14.8: Celebrity match null when below threshold

**Progressive Degradation:**
When celebrity matching fails, the `celebrity_match` field is set to `None` while other fields remain populated. This satisfies Requirements 17.1-17.4.

### FeatureBreakdownItem

**Description:** Score for a single facial ratio feature.

**Fields:**
- `feature` (str): Name of the facial ratio (e.g., "Face Length ÷ Face Width")
- `measured_ratio` (float): Actual measured ratio from landmarks
- `ideal_ratio` (float): Ideal golden ratio value (always 1.618)
- `score` (float, 0-100): Score for this feature
- `deviation` (float): Deviation from ideal (positive = above, negative = below)

**Validation:**
- Score must be between 0 and 100 (enforced by Pydantic)

**Example:**
```json
{
  "feature": "Face Length ÷ Face Width",
  "measured_ratio": 1.625,
  "ideal_ratio": 1.618,
  "score": 99.6,
  "deviation": 0.007
}
```

### OverlayData

**Description:** Data for rendering visual overlays on analyzed faces.

**Fields:**
- `landmarks` (List[List[float]]): 468 facial landmark coordinates as [x, y] pairs
- `golden_ratio_guides` (GoldenRatioGuides): Golden ratio guide lines

**Example:**
```json
{
  "landmarks": [[100.5, 200.3], [101.2, 201.5], ...],
  "golden_ratio_guides": { ... }
}
```

### GoldenRatioGuides

**Description:** Visual guide data for golden ratio overlay.

**Fields:**
- `vertical_midline` (float): X-coordinate of vertical center line
- `horizontal_thirds` (List[float], length=4): Y-coordinates for horizontal divisions [top, third_1, third_2, bottom]
- `phi_spiral_center` (List[float], length=2): [x, y] coordinates for phi spiral center

**Validation:**
- horizontal_thirds must have exactly 4 values
- phi_spiral_center must have exactly 2 values

### CelebrityMatch

**Description:** Celebrity look-alike match result.

**Fields:**
- `name` (Optional[str]): Celebrity name (null if confidence below threshold)
- `confidence` (Optional[float], 0-1): Match confidence score
- `dataset_source` (Optional[str]): Source dataset ("Western" or "Bollywood")
- `thumbnail_url` (Optional[str]): URL to celebrity thumbnail
- `message` (Optional[str]): Message when no match found

**States:**
1. **Successful Match** (confidence ≥ 0.55):
   ```json
   {
     "name": "Ryan Gosling",
     "confidence": 0.72,
     "dataset_source": "Western",
     "thumbnail_url": "/celebrities/ryan_gosling.jpg",
     "message": null
   }
   ```

2. **No Match** (confidence < 0.55):
   ```json
   {
     "name": null,
     "confidence": 0.48,
     "dataset_source": null,
     "thumbnail_url": null,
     "message": "No close match found in our dataset"
   }
   ```

3. **Feature Unavailable** (progressive degradation):
   ```json
   null
   ```

### ErrorResponse

**Description:** Error response for failed analysis.

**Fields:**
- `status` (str): Always "error" for error responses
- `error_code` (ErrorCodeEnum): Machine-readable error code
- `message` (str): User-friendly error message with guidance
- `retry` (bool): Whether the request can be retried
- `details` (Optional[Dict]): Additional error details

**Requirements Coverage:**
- Requirement 15.1: Appropriate HTTP status code (400, 408, 500)
- Requirement 15.2: Status field with value "error"
- Requirement 15.3: Error code field
- Requirement 15.4: User-friendly message field
- Requirement 15.5: Retry boolean field

## Error Codes

### ErrorCodeEnum

**Values:**
- `NO_FACE_DETECTED`: No face found in image (HTTP 400, retry=True)
- `MULTIPLE_FACES`: Multiple faces detected (HTTP 400, retry=True)
- `POOR_QUALITY`: Image quality too low (HTTP 400, retry=True)
- `OCCLUSION_DETECTED`: Face accessories detected (HTTP 400, retry=True)
- `INVALID_IMAGE`: Corrupt or unsupported format (HTTP 400, retry=True)
- `PROCESSING_TIMEOUT`: Processing exceeded 30 seconds (HTTP 408, retry=True)
- `INTERNAL_ERROR`: Server error (HTTP 500, retry=False)
- `NO_IMAGE_PROVIDED`: No image in request (HTTP 400, retry=True)
- `NETWORK_ERROR`: Network connectivity issue (retry=True)

## Symmetry Types

### SymmetryTypeEnum

**Values:**
- `oval`: Balanced and proportioned face shape
- `rectangle`: Long face with angular jawline
- `round`: Equal length and width with soft angles
- `square`: Equal length and width with angular jawline
- `heart`: Wide forehead tapering to narrow chin
- `diamond`: Prominent cheekbones with narrow forehead and chin

## Validation Rules

### Field Constraints

1. **golden_ratio_score**: 0.0 ≤ score ≤ 100.0
2. **percentile_rank**: 0 ≤ rank ≤ 100
3. **feature_breakdown**: Must contain exactly 8 items
4. **score** (per feature): 0.0 ≤ score ≤ 100.0
5. **confidence**: 0.0 ≤ confidence ≤ 1.0
6. **horizontal_thirds**: Must contain exactly 4 values
7. **phi_spiral_center**: Must contain exactly 2 values

### Validation Examples

```python
# Valid: Score within bounds
FeatureBreakdownItem(
    feature="Face Length ÷ Face Width",
    measured_ratio=1.625,
    ideal_ratio=1.618,
    score=99.6,  # Valid: 0 ≤ 99.6 ≤ 100
    deviation=0.007
)

# Invalid: Score exceeds 100
FeatureBreakdownItem(
    feature="Face Length ÷ Face Width",
    measured_ratio=1.625,
    ideal_ratio=1.618,
    score=150.0,  # Invalid: exceeds 100
    deviation=0.007
)
# Raises: ValidationError
```

## JSON Schema Examples

All models include comprehensive JSON schema examples for API documentation. These examples are automatically included in the FastAPI OpenAPI specification.

### Accessing Examples

```python
from app.models.responses import AnalysisResponse

# Get JSON schema with examples
schema = AnalysisResponse.model_json_schema()
examples = schema.get('examples', [])
```

### Example Scenarios

1. **Successful Analysis with Celebrity Match**
   - All 8 features scored
   - Celebrity match with confidence ≥ 0.55
   - Full overlay data included

2. **Successful Analysis without Celebrity Match**
   - All 8 features scored
   - celebrity_match = null (progressive degradation)
   - Full overlay data included

3. **Error Responses**
   - Examples for all 7 error codes
   - User-friendly messages with guidance
   - Retry flag indicates if request can be retried

## Backward Compatibility

The following aliases are provided for backward compatibility:

- `FeatureScore` → `FeatureBreakdownItem`
- `AnalysisResult` → `AnalysisResultModel`

These aliases can be used interchangeably with the primary model names.

## Usage Examples

### Creating a Successful Response

```python
from app.models.responses import (
    AnalysisResponse,
    AnalysisResultModel,
    FeatureBreakdownItem,
    OverlayData,
    GoldenRatioGuides,
    CelebrityMatch,
    SymmetryTypeEnum
)

# Create feature breakdown
features = [
    FeatureBreakdownItem(
        feature="Face Length ÷ Face Width",
        measured_ratio=1.625,
        ideal_ratio=1.618,
        score=99.6,
        deviation=0.007
    ),
    # ... 7 more features
]

# Create overlay data
overlay = OverlayData(
    landmarks=[[100.0, 200.0], ...],  # 468 landmarks
    golden_ratio_guides=GoldenRatioGuides(
        vertical_midline=320.5,
        horizontal_thirds=[50.0, 165.5, 280.8, 396.0],
        phi_spiral_center=[320.5, 180.0]
    )
)

# Create celebrity match
celebrity = CelebrityMatch(
    name="Ryan Gosling",
    confidence=0.72,
    dataset_source="Western",
    thumbnail_url="/celebrities/ryan_gosling.jpg"
)

# Create result
result = AnalysisResultModel(
    golden_ratio_score=78.3,
    percentile_rank=82,
    face_symmetry_type=SymmetryTypeEnum.OVAL,
    feature_breakdown=features,
    overlay_data=overlay,
    celebrity_match=celebrity
)

# Create response
response = AnalysisResponse(
    processing_time_ms=2847,
    results=result
)

# Serialize to JSON
json_response = response.model_dump_json()
```

### Creating an Error Response

```python
from app.models.responses import ErrorResponse, ErrorCodeEnum

error = ErrorResponse(
    error_code=ErrorCodeEnum.NO_FACE_DETECTED,
    message="We couldn't detect a face in your photo. Please take a clear, front-facing photo with good lighting.",
    retry=True
)

# Serialize to JSON
json_error = error.model_dump_json()
```

### Progressive Degradation Example

```python
# When celebrity matching fails, set celebrity_match to None
result = AnalysisResultModel(
    golden_ratio_score=78.3,
    percentile_rank=82,
    face_symmetry_type=SymmetryTypeEnum.OVAL,
    feature_breakdown=features,
    overlay_data=overlay,
    celebrity_match=None  # Celebrity matching failed
)

response = AnalysisResponse(
    processing_time_ms=1850,
    results=result
)
```

## Testing

A comprehensive test suite is available in `test_response_models.py`:

```bash
python test_response_models.py
```

Tests include:
- Model instantiation
- Field validation
- JSON serialization
- Error code coverage
- Progressive degradation scenarios
- JSON schema examples

## FastAPI Integration

These models integrate seamlessly with FastAPI:

```python
from fastapi import FastAPI
from app.models.responses import AnalysisResponse, ErrorResponse

app = FastAPI()

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_face():
    # ... processing logic
    return AnalysisResponse(...)

@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    return ErrorResponse(
        error_code=ErrorCodeEnum.NO_FACE_DETECTED,
        message="...",
        retry=True
    )
```

FastAPI automatically:
- Validates response data against the model
- Generates OpenAPI schema with examples
- Provides interactive API documentation
- Handles serialization/deserialization

## Migration Guide

If updating from a previous version:

1. **Model Renames:**
   - Use `AnalysisResultModel` instead of `AnalysisResult` (or use the alias)
   - Use `FeatureBreakdownItem` instead of `FeatureScore` (or use the alias)

2. **New Fields:**
   - `ErrorResponse.error_code` is now typed as `ErrorCodeEnum` (was `str`)
   - `GoldenRatioGuides.vertical_midline` renamed from `midline`

3. **Validation Changes:**
   - `horizontal_thirds` now requires exactly 4 values (was 3)
   - `feature_breakdown` strictly enforces 8 items

## References

- Pydantic v2 Documentation: https://docs.pydantic.dev/latest/
- FastAPI Response Models: https://fastapi.tiangolo.com/tutorial/response-model/
- Requirements Document: `requirements.md`
- Design Document: `design.md`
