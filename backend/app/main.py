"""
FaceMatric API Gateway

FastAPI application that orchestrates facial analysis components.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import time
import uuid
from typing import Optional

from app.config import settings
from app.models.requests import AnalyzeRequest
from app.models.responses import (
    AnalysisResponse,
    AnalysisResult,
    FeatureScore,
    OverlayData,
    GoldenRatioGuides,
    CelebrityMatch,
    ErrorResponse,
    HealthCheckResponse,
    SymmetryTypeEnum
)
from app.models.internal import ProcessingMetadata
from app.components import (
    image_processor,
    get_face_detector,
    symmetry_classifier,
    golden_ratio_scorer,
    percentile_engine,
    get_celebrity_matcher,
    overlay_generator
)
from app.utils.errors import ValidationError, ErrorCode, get_error_message
from app.utils.logger import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Facial geometry analysis using the Golden Ratio (φ = 1.618)",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    
    # Initialize face detector
    try:
        get_face_detector()
        logger.info("Face detector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize face detector: {str(e)}")
    
    # Initialize celebrity matcher
    if settings.enable_celebrity_matching:
        try:
            matcher = get_celebrity_matcher()
            if matcher.is_available:
                logger.info("Celebrity matcher initialized successfully")
            else:
                logger.warning("Celebrity matcher unavailable")
        except Exception as e:
            logger.error(f"Failed to initialize celebrity matcher: {str(e)}")


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns service health status and feature availability.
    """
    celebrity_available = False
    if settings.enable_celebrity_matching:
        try:
            matcher = get_celebrity_matcher()
            celebrity_available = matcher.is_available
        except Exception:
            pass
    
    return HealthCheckResponse(
        status="healthy",
        version=settings.app_version,
        celebrity_matching_available=celebrity_available
    )


@app.post("/api/analyze", response_model=AnalysisResponse, responses={
    400: {"model": ErrorResponse},
    408: {"model": ErrorResponse},
    500: {"model": ErrorResponse}
})
async def analyze_face(
    request: Request,
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = None,
    include_celebrity_match: bool = True
):
    """
    Analyze facial proportions against the golden ratio.
    
    **Accepts:**
    - Multipart file upload (image/jpeg, image/png, etc.)
    - Base64-encoded image string in request body
    
    **Returns:**
    - Golden ratio score (0-100)
    - Percentile ranking
    - Face symmetry type classification
    - Feature breakdown (8 facial ratios)
    - Visual overlay data
    - Celebrity match (optional)
    
    **Processing Time:** Typically 2-5 seconds
    **Timeout:** 30 seconds maximum
    """
    # Generate request ID for tracking
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(f"[{request_id}] Analysis request received")
    
    try:
        # Enforce timeout
        result = await asyncio.wait_for(
            _process_analysis(
                request_id=request_id,
                file=file,
                image_base64=image_base64,
                include_celebrity=include_celebrity_match
            ),
            timeout=settings.processing_timeout
        )
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        logger.info(
            f"[{request_id}] Analysis completed successfully "
            f"in {processing_time}ms"
        )
        
        return AnalysisResponse(
            status="success",
            processing_time_ms=processing_time,
            results=result
        )
        
    except asyncio.TimeoutError:
        logger.error(f"[{request_id}] Processing timeout exceeded")
        raise HTTPException(
            status_code=408,
            detail={
                "status": "error",
                "error_code": ErrorCode.PROCESSING_TIMEOUT.value,
                "message": get_error_message(ErrorCode.PROCESSING_TIMEOUT),
                "retry": True
            }
        )
    
    except ValidationError as e:
        logger.warning(f"[{request_id}] Validation error: {e.error_code.value}")
        raise HTTPException(
            status_code=400,
            detail=e.to_dict()
        )
    
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "error_code": ErrorCode.INTERNAL_ERROR.value,
                "message": get_error_message(ErrorCode.INTERNAL_ERROR),
                "retry": True
            }
        )


async def _process_analysis(
    request_id: str,
    file: Optional[UploadFile],
    image_base64: Optional[str],
    include_celebrity: bool
) -> AnalysisResult:
    """
    Internal processing orchestration.
    
    Coordinates all components and handles progressive degradation.
    """
    metadata = ProcessingMetadata(
        request_id=request_id,
        start_time=time.time(),
        component_times={}
    )
    
    # === 1. Image Preprocessing ===
    component_start = time.time()
    
    # Get image bytes
    file_bytes = None
    if file:
        file_bytes = await file.read()
    
    # Process image
    image_data = await image_processor.process_upload(
        file_bytes=file_bytes,
        base64_str=image_base64
    )
    
    metadata.add_component_time("image_processing", int((time.time() - component_start) * 1000))
    logger.debug(f"[{request_id}] Image processed")
    
    # === 2. Face Detection and Landmark Extraction ===
    component_start = time.time()
    
    face_detector = get_face_detector()
    landmarks = face_detector.detect_faces(image_data)
    
    metadata.add_component_time("face_detection", int((time.time() - component_start) * 1000))
    logger.debug(f"[{request_id}] Face detected with {len(landmarks.points)} landmarks")
    
    # === 3. Parallel Processing of Core Components ===
    # Run symmetry classification, golden ratio scoring, and overlay generation in parallel
    
    async def compute_symmetry():
        """Classify face symmetry type."""
        component_start = time.time()
        result = symmetry_classifier.classify(landmarks)
        metadata.add_component_time("symmetry_classification", int((time.time() - component_start) * 1000))
        return result
    
    async def compute_golden_ratio():
        """Compute golden ratio scores."""
        component_start = time.time()
        result = golden_ratio_scorer.score_all(landmarks)
        metadata.add_component_time("golden_ratio_scoring", int((time.time() - component_start) * 1000))
        return result
    
    async def compute_overlay():
        """Generate overlay data."""
        component_start = time.time()
        result = overlay_generator.generate(landmarks)
        metadata.add_component_time("overlay_generation", int((time.time() - component_start) * 1000))
        return result
    
    # Execute in parallel
    symmetry_type, golden_ratio_result, overlay_data_internal = await asyncio.gather(
        compute_symmetry(),
        compute_golden_ratio(),
        compute_overlay()
    )
    
    logger.debug(f"[{request_id}] Core analysis complete")
    
    # === 4. Compute Percentile ===
    component_start = time.time()
    
    percentile = percentile_engine.score_to_percentile(golden_ratio_result.overall_score)
    
    metadata.add_component_time("percentile_calculation", int((time.time() - component_start) * 1000))
    
    # === 5. Optional Celebrity Matching (Progressive Degradation) ===
    celebrity_match_internal = None
    
    if include_celebrity and settings.enable_celebrity_matching:
        try:
            component_start = time.time()
            
            celebrity_matcher = get_celebrity_matcher()
            if celebrity_matcher.is_available:
                celebrity_match_internal = await celebrity_matcher.find_match(image_data)
                metadata.add_component_time("celebrity_matching", int((time.time() - component_start) * 1000))
            else:
                logger.debug(f"[{request_id}] Celebrity matching skipped: unavailable")
        
        except Exception as e:
            # Graceful degradation - don't fail entire request
            logger.warning(f"[{request_id}] Celebrity matching failed: {str(e)}")
            celebrity_match_internal = None
    
    # === 6. Convert Internal Models to Response Models ===
    
    # Convert feature scores
    feature_scores = [
        FeatureScore(
            feature=fs.feature,
            measured_ratio=fs.measured_ratio,
            ideal_ratio=fs.ideal_ratio,
            score=fs.score,
            deviation=fs.deviation
        )
        for fs in golden_ratio_result.features
    ]
    
    # Convert overlay data
    overlay_data = OverlayData(
        landmarks=overlay_data_internal.landmarks_2d.tolist(),
        golden_ratio_guides=GoldenRatioGuides(
            midline=overlay_data_internal.midline_x,
            horizontal_thirds=overlay_data_internal.horizontal_thirds_y,
            phi_spiral_center=list(overlay_data_internal.phi_spiral_center)
        )
    )
    
    # Convert celebrity match
    celebrity_match = None
    if celebrity_match_internal:
        celebrity_match = CelebrityMatch(
            name=celebrity_match_internal.name,
            confidence=celebrity_match_internal.confidence,
            dataset_source=celebrity_match_internal.dataset_source,
            thumbnail_url=celebrity_match_internal.thumbnail_path,
            message=celebrity_match_internal.message
        )
    
    # === 7. Privacy-Preserving Cleanup ===
    # Image data will be garbage collected automatically
    # No persistent storage of user images
    
    logger.info(
        f"[{request_id}] Component times: {metadata.component_times}"
    )
    
    # Return complete analysis result
    return AnalysisResult(
        golden_ratio_score=golden_ratio_result.overall_score,
        percentile_rank=percentile,
        face_symmetry_type=SymmetryTypeEnum(symmetry_type.value),
        feature_breakdown=feature_scores,
        overlay_data=overlay_data,
        celebrity_match=celebrity_match
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle ValidationError exceptions."""
    return JSONResponse(
        status_code=400,
        content=exc.to_dict()
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
