"""
Pydantic response models for FaceMatric API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class ErrorCodeEnum(str, Enum):
    """Standardized error codes for FaceMatric API."""
    NO_FACE_DETECTED = "NO_FACE_DETECTED"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    POOR_QUALITY = "POOR_QUALITY"
    OCCLUSION_DETECTED = "OCCLUSION_DETECTED"
    INVALID_IMAGE = "INVALID_IMAGE"
    PROCESSING_TIMEOUT = "PROCESSING_TIMEOUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    NO_IMAGE_PROVIDED = "NO_IMAGE_PROVIDED"
    NETWORK_ERROR = "NETWORK_ERROR"


class SymmetryTypeEnum(str, Enum):
    """Face symmetry type classifications."""
    OVAL = "oval"
    RECTANGLE = "rectangle"
    ROUND = "round"
    SQUARE = "square"
    HEART = "heart"
    DIAMOND = "diamond"


class FeatureBreakdownItem(BaseModel):
    """Score for a single facial ratio feature."""
    
    feature: str = Field(
        ...,
        description="Name of the facial ratio feature",
        example="Face Length ÷ Face Width"
    )
    
    measured_ratio: float = Field(
        ...,
        description="Actual measured ratio from facial landmarks",
        example=1.625
    )
    
    ideal_ratio: float = Field(
        ...,
        description="Ideal golden ratio value (φ = 1.618)",
        example=1.618
    )
    
    score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Score for this feature (0-100)",
        example=99.6
    )
    
    deviation: float = Field(
        ...,
        description="Deviation from ideal ratio (positive = above ideal, negative = below)",
        example=0.007
    )
    
    ratio_left: Optional[float] = Field(
        None,
        description="Left-side ratio measurement for bilateral features",
        example=1.610
    )
    
    ratio_right: Optional[float] = Field(
        None,
        description="Right-side ratio measurement for bilateral features",
        example=1.626
    )
    
    asymmetry_pct: Optional[float] = Field(
        None,
        description="Bilateral asymmetry percentage between left and right measurements",
        example=1.0
    )


# Alias for backward compatibility
FeatureScore = FeatureBreakdownItem


class GoldenRatioGuides(BaseModel):
    """Visual guide data for golden ratio overlay."""
    
    vertical_midline: float = Field(
        ...,
        description="Vertical midline x-coordinate",
        example=320.5
    )
    
    horizontal_thirds: List[float] = Field(
        ...,
        description="Y-coordinates for horizontal golden ratio divisions (4 values: top, third_1, third_2, bottom)",
        min_length=4,
        max_length=4,
        example=[50.0, 120.0, 240.0, 360.0]
    )
    
    phi_spiral_center: List[float] = Field(
        ...,
        description="[x, y] coordinates for phi spiral center point",
        min_length=2,
        max_length=2,
        example=[320.5, 180.0]
    )


class OverlayData(BaseModel):
    """Data for rendering visual overlays on the analyzed face."""
    
    landmarks: List[List[float]] = Field(
        ...,
        description="468 facial landmark coordinates as [x, y] pairs",
        example=[[100.5, 200.3], [101.2, 201.5]]
    )
    
    golden_ratio_guides: GoldenRatioGuides = Field(
        ...,
        description="Golden ratio guide lines and reference points"
    )


class CelebrityMatchItem(BaseModel):
    """Individual celebrity match in ranked look-alike results."""
    rank: int = Field(..., description="Rank position (1 to 5)", example=1)
    name: str = Field(..., description="Celebrity name", example="Shah Rukh Khan")
    similarity: float = Field(..., description="Raw cosine similarity score (2 decimals)", example=0.72)
    strength: str = Field(..., description="Match strength: strong, moderate, or weak", example="strong")
    confidence: float = Field(..., description="Deprecated alias equal to similarity", example=0.72)
    dataset_source: Optional[str] = Field(None, description="Dataset source", example="Bollywood")
    thumbnail_url: Optional[str] = Field(None, description="URL to thumbnail image")


class CelebrityMatch(BaseModel):
    """Celebrity look-alike match result."""
    
    name: Optional[str] = Field(
        None,
        description="Celebrity name (null if best match is weak)",
        example="Ryan Gosling"
    )
    
    similarity: Optional[float] = Field(
        None,
        description="Raw cosine similarity score (2 decimals)",
        example=0.72
    )

    strength: Optional[str] = Field(
        None,
        description="Match strength: strong (>= 0.55), moderate (0.40 to 0.55), or weak (< 0.40)",
        example="strong"
    )

    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Deprecated alias equal to similarity",
        example=0.72
    )
    
    dataset_source: Optional[str] = Field(
        None,
        description="Source dataset (Western or Bollywood)",
        example="Western"
    )
    
    thumbnail_url: Optional[str] = Field(
        None,
        description="URL to celebrity thumbnail image",
        example="/celebrities/ryan_gosling.jpg"
    )
    
    message: Optional[str] = Field(
        None,
        description="Message when no match found or weak resemblance",
        example="No close resemblance found"
    )

    top_matches: Optional[List[CelebrityMatchItem]] = Field(
        default_factory=list,
        description="Top 5 ranked celebrity matches"
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "description": "Successful celebrity match",
                    "value": {
                        "name": "Ryan Gosling",
                        "confidence": 0.72,
                        "dataset_source": "Western",
                        "thumbnail_url": "/celebrities/ryan_gosling.jpg",
                        "message": None
                    }
                },
                {
                    "description": "No match found (below confidence threshold)",
                    "value": {
                        "name": None,
                        "confidence": 0.48,
                        "dataset_source": None,
                        "thumbnail_url": None,
                        "message": "No close match found in our dataset"
                    }
                }
            ]
        }


class AnalysisResultModel(BaseModel):
    """Complete facial analysis results."""
    
    golden_ratio_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Overall golden ratio score (0-100)",
        example=78.3
    )
    
    percentile_rank: int = Field(
        ...,
        ge=0,
        le=100,
        description="Percentile rank compared to population (0-100)",
        example=82
    )
    
    face_symmetry_type: SymmetryTypeEnum = Field(
        ...,
        description="Classified face symmetry type",
        example="oval"
    )
    
    feature_breakdown: List[FeatureBreakdownItem] = Field(
        ...,
        description="Detailed scores for each of the facial ratios (including bilateral splits for asymmetry)",
        min_length=8,
        max_length=16
    )
    
    overlay_data: OverlayData = Field(
        ...,
        description="Data for rendering visual overlays on the analyzed face"
    )
    
    celebrity_match: Optional[CelebrityMatch] = Field(
        None,
        description="Celebrity look-alike match (null if disabled or failed)"
    )


# Alias for backward compatibility
AnalysisResult = AnalysisResultModel


class AnalysisResponse(BaseModel):
    """Successful analysis response."""
    
    status: str = Field(
        "success",
        description="Response status",
        example="success"
    )
    
    processing_time_ms: int = Field(
        ...,
        description="Total processing time in milliseconds",
        example=2847
    )
    
    results: AnalysisResultModel = Field(
        ...,
        description="Complete analysis results"
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "description": "Successful analysis with celebrity match",
                    "value": {
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
                                },
                                {
                                    "feature": "Face Width ÷ Jaw Width",
                                    "measured_ratio": 1.580,
                                    "ideal_ratio": 1.618,
                                    "score": 97.7,
                                    "deviation": -0.038
                                },
                                {
                                    "feature": "Eye Spacing ÷ Eye Width",
                                    "measured_ratio": 1.650,
                                    "ideal_ratio": 1.618,
                                    "score": 98.0,
                                    "deviation": 0.032
                                },
                                {
                                    "feature": "Nose Width ÷ Mouth Width",
                                    "measured_ratio": 1.520,
                                    "ideal_ratio": 1.618,
                                    "score": 93.9,
                                    "deviation": -0.098
                                },
                                {
                                    "feature": "Mouth Width ÷ Face Width",
                                    "measured_ratio": 1.600,
                                    "ideal_ratio": 1.618,
                                    "score": 98.9,
                                    "deviation": -0.018
                                },
                                {
                                    "feature": "Nose Bridge to Chin ÷ Hairline to Nose Bridge",
                                    "measured_ratio": 1.595,
                                    "ideal_ratio": 1.618,
                                    "score": 98.6,
                                    "deviation": -0.023
                                },
                                {
                                    "feature": "Eye-Eyebrow Distance ÷ Inner-Eye Distance",
                                    "measured_ratio": 1.450,
                                    "ideal_ratio": 1.618,
                                    "score": 89.6,
                                    "deviation": -0.168
                                },
                                {
                                    "feature": "Jaw Width ÷ Chin Width",
                                    "measured_ratio": 1.680,
                                    "ideal_ratio": 1.618,
                                    "score": 96.2,
                                    "deviation": 0.062
                                }
                            ],
                            "overlay_data": {
                                "landmarks": [
                                    [100.5, 200.3], [101.2, 201.5], [102.0, 202.8]
                                ],
                                "golden_ratio_guides": {
                                    "vertical_midline": 320.5,
                                    "horizontal_thirds": [50.0, 165.5, 280.8, 396.0],
                                    "phi_spiral_center": [320.5, 180.0]
                                }
                            },
                            "celebrity_match": {
                                "name": "Ryan Gosling",
                                "confidence": 0.72,
                                "dataset_source": "Western",
                                "thumbnail_url": "/celebrities/ryan_gosling.jpg",
                                "message": None
                            }
                        }
                    }
                },
                {
                    "description": "Successful analysis without celebrity match (progressive degradation)",
                    "value": {
                        "status": "success",
                        "processing_time_ms": 1950,
                        "results": {
                            "golden_ratio_score": 72.5,
                            "percentile_rank": 68,
                            "face_symmetry_type": "rectangle",
                            "feature_breakdown": [
                                {
                                    "feature": "Face Length ÷ Face Width",
                                    "measured_ratio": 1.720,
                                    "ideal_ratio": 1.618,
                                    "score": 93.7,
                                    "deviation": 0.102
                                },
                                {
                                    "feature": "Face Width ÷ Jaw Width",
                                    "measured_ratio": 1.550,
                                    "ideal_ratio": 1.618,
                                    "score": 95.8,
                                    "deviation": -0.068
                                },
                                {
                                    "feature": "Eye Spacing ÷ Eye Width",
                                    "measured_ratio": 1.480,
                                    "ideal_ratio": 1.618,
                                    "score": 91.5,
                                    "deviation": -0.138
                                },
                                {
                                    "feature": "Nose Width ÷ Mouth Width",
                                    "measured_ratio": 1.390,
                                    "ideal_ratio": 1.618,
                                    "score": 85.9,
                                    "deviation": -0.228
                                },
                                {
                                    "feature": "Mouth Width ÷ Face Width",
                                    "measured_ratio": 1.650,
                                    "ideal_ratio": 1.618,
                                    "score": 98.0,
                                    "deviation": 0.032
                                },
                                {
                                    "feature": "Nose Bridge to Chin ÷ Hairline to Nose Bridge",
                                    "measured_ratio": 1.500,
                                    "ideal_ratio": 1.618,
                                    "score": 92.7,
                                    "deviation": -0.118
                                },
                                {
                                    "feature": "Eye-Eyebrow Distance ÷ Inner-Eye Distance",
                                    "measured_ratio": 1.350,
                                    "ideal_ratio": 1.618,
                                    "score": 83.4,
                                    "deviation": -0.268
                                },
                                {
                                    "feature": "Jaw Width ÷ Chin Width",
                                    "measured_ratio": 1.590,
                                    "ideal_ratio": 1.618,
                                    "score": 98.3,
                                    "deviation": -0.028
                                }
                            ],
                            "overlay_data": {
                                "landmarks": [
                                    [105.2, 195.8], [106.0, 197.2], [107.5, 198.9]
                                ],
                                "golden_ratio_guides": {
                                    "vertical_midline": 315.0,
                                    "horizontal_thirds": [45.0, 160.0, 275.0, 390.0],
                                    "phi_spiral_center": [315.0, 175.5]
                                }
                            },
                            "celebrity_match": None
                        }
                    }
                }
            ]
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    
    status: str = Field(
        "error",
        description="Response status",
        example="error"
    )
    
    error_code: ErrorCodeEnum = Field(
        ...,
        description="Machine-readable error code",
        example="NO_FACE_DETECTED"
    )
    
    message: str = Field(
        ...,
        description="User-friendly error message with guidance",
        example="We couldn't detect a face in your photo. Please take a clear, front-facing photo with good lighting."
    )
    
    retry: bool = Field(
        ...,
        description="Whether the request can be retried",
        example=True
    )
    
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details (optional)",
        example={}
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "status": "error",
                    "error_code": "NO_FACE_DETECTED",
                    "message": "We couldn't detect a face in your photo. Please take a clear, front-facing photo with good lighting.",
                    "retry": True,
                    "details": {}
                },
                {
                    "status": "error",
                    "error_code": "MULTIPLE_FACES",
                    "message": "Multiple faces detected. Please upload a photo with only one person.",
                    "retry": True,
                    "details": {}
                },
                {
                    "status": "error",
                    "error_code": "POOR_QUALITY",
                    "message": "Photo quality is too low. Please ensure your face is well-lit and facing the camera directly.",
                    "retry": True,
                    "details": {}
                },
                {
                    "status": "error",
                    "error_code": "OCCLUSION_DETECTED",
                    "message": "We detected accessories covering your face. Please upload a photo without glasses, masks, or other obstructions for accurate analysis.",
                    "retry": True,
                    "details": {}
                },
                {
                    "status": "error",
                    "error_code": "INVALID_IMAGE",
                    "message": "Corrupt or unsupported file format",
                    "retry": True,
                    "details": {}
                },
                {
                    "status": "error",
                    "error_code": "PROCESSING_TIMEOUT",
                    "message": "Processing took too long. Please try again with a clearer photo.",
                    "retry": True,
                    "details": {}
                },
                {
                    "status": "error",
                    "error_code": "INTERNAL_ERROR",
                    "message": "Internal server error",
                    "retry": False,
                    "details": {}
                }
            ]
        }


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(
        "healthy",
        description="Service health status",
        example="healthy"
    )
    
    version: str = Field(
        ...,
        description="API version",
        example="1.0.0"
    )
    
    celebrity_matching_available: bool = Field(
        ...,
        description="Whether celebrity matching is available",
        example=True
    )
