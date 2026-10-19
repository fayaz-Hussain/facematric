"""
FaceMatric API Gateway

FastAPI application that orchestrates facial analysis components.
Implements privacy-preserving in-memory image processing with no persistent storage.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import asyncio
import time
import uuid
import gc
from typing import Optional
from contextlib import asynccontextmanager

from app.config import settings
from app.models.requests import AnalyzeRequest
from app.models.responses import (
    AnalysisResponse,
    AnalysisResult,
    FeatureScore,
    OverlayData,
    GoldenRatioGuides,
    CelebrityMatch,
    CelebrityMatchItem,
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

# Mount static files for celebrity thumbnails
import os
celebrity_dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "celebrity_dataset"))
if not os.path.exists(celebrity_dataset_dir):
    alt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "FaceMatric", "celebrity_dataset"))
    if os.path.exists(alt_dir):
        celebrity_dataset_dir = alt_dir
if os.path.exists(celebrity_dataset_dir):
    app.mount("/celebrities", StaticFiles(directory=celebrity_dataset_dir), name="celebrities")


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
    
    **Privacy Note:** All image data is processed in-memory only and deleted
    immediately after analysis completes. No images are written to disk or
    persisted on the server.
    
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
    request.state.request_id = request_id  # Store for middleware/logging
    start_time = time.time()
    
    # Track whether analysis succeeded for proper cleanup
    analysis_succeeded = False
    
    logger.info(f"[{request_id}] Analysis request received")
    
    file_bytes = None
    extracted_file = file
    extracted_base64 = image_base64
    
    # Extract image_base64 and include_celebrity_match from body (JSON or Form) if not in query/file
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                extracted_base64 = extracted_base64 or body.get("image_base64") or body.get("image")
                if "include_celebrity_match" in body:
                    val = body["include_celebrity_match"]
                    include_celebrity_match = val if isinstance(val, bool) else str(val).strip().lower() in ("true", "1", "yes")
        except Exception as e:
            logger.debug(f"[{request_id}] Failed to read JSON body: {e}")
    elif "form" in content_type:
        try:
            form = await request.form()
            extracted_base64 = extracted_base64 or form.get("image_base64") or form.get("image")
            if not extracted_file and "image" in form and hasattr(form["image"], "filename"):
                extracted_file = form["image"]
            if "include_celebrity_match" in form:
                val = form["include_celebrity_match"]
                include_celebrity_match = val if isinstance(val, bool) else str(val).strip().lower() in ("true", "1", "yes")
        except Exception as e:
            logger.debug(f"[{request_id}] Failed to read form data: {e}")
    else:
        # Fallback for omitted/generic content-type
        if not extracted_base64:
            try:
                body = await request.json()
                if isinstance(body, dict):
                    extracted_base64 = body.get("image_base64") or body.get("image")
                    if "include_celebrity_match" in body:
                        val = body["include_celebrity_match"]
                        include_celebrity_match = val if isinstance(val, bool) else str(val).strip().lower() in ("true", "1", "yes")
            except Exception:
                try:
                    form = await request.form()
                    extracted_base64 = form.get("image_base64") or form.get("image")
                    if not extracted_file and "image" in form and hasattr(form["image"], "filename"):
                        extracted_file = form["image"]
                    if "include_celebrity_match" in form:
                        val = form["include_celebrity_match"]
                        include_celebrity_match = val if isinstance(val, bool) else str(val).strip().lower() in ("true", "1", "yes")
                except Exception:
                    pass

    try:
        # Validate request format: must provide exactly one input method
        await _validate_request_format(request_id, extracted_file, extracted_base64)
        
        # If file upload, validate file properties
        if extracted_file:
            file_bytes = await _validate_and_read_file(request_id, extracted_file)
        
        try:
            # Enforce timeout
            # Note: _process_analysis will handle all cleanup internally
            result = await asyncio.wait_for(
                _process_analysis(
                    request_id=request_id,
                    file_bytes=file_bytes,
                    image_base64=extracted_base64,
                    include_celebrity=include_celebrity_match
                ),
                timeout=settings.processing_timeout
            )
            
            analysis_succeeded = True
            
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
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"[{request_id}] Processing timeout exceeded: "
                f"{settings.processing_timeout}s timeout after {elapsed_ms}ms"
            )
            return JSONResponse(
                status_code=408,
                content={
                    "status": "error",
                    "error_code": ErrorCode.PROCESSING_TIMEOUT.value,
                    "message": get_error_message(ErrorCode.PROCESSING_TIMEOUT),
                    "retry": True,
                    "detail": {
                        "status": "error",
                        "error_code": ErrorCode.PROCESSING_TIMEOUT.value,
                        "message": get_error_message(ErrorCode.PROCESSING_TIMEOUT),
                        "retry": True
                    }
                }
            )
        
        except ValidationError as e:
            logger.warning(f"[{request_id}] Validation error: {e.error_code.value}")
            content = e.to_dict()
            content["detail"] = {
                "status": content.get("status", "error"),
                "error_code": content.get("error_code"),
                "message": content.get("message"),
                "retry": content.get("retry", True)
            }
            return JSONResponse(
                status_code=400,
                content=content
            )
        
        except Exception as e:
            logger.error(f"[{request_id}] Unexpected error: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "error_code": ErrorCode.INTERNAL_ERROR.value,
                    "message": get_error_message(ErrorCode.INTERNAL_ERROR),
                    "retry": True,
                    "detail": {
                        "status": "error",
                        "error_code": ErrorCode.INTERNAL_ERROR.value,
                        "message": get_error_message(ErrorCode.INTERNAL_ERROR),
                        "retry": True
                    }
                }
            )
    except ValidationError as e:
        logger.warning(f"[{request_id}] Validation error: {e.error_code.value}")
        content = e.to_dict()
        content["detail"] = {
            "status": content.get("status", "error"),
            "error_code": content.get("error_code"),
            "message": content.get("message"),
            "retry": content.get("retry", True)
        }
        return JSONResponse(
            status_code=400,
            content=content
        )
    finally:
        # === PRIVACY-PRESERVING CLEANUP ===
        # Requirement 13.1, 13.2: Clean up image data after processing (success or error)
        
        # Clear any remaining references to image data
        if file_bytes is not None:
            del file_bytes
            logger.debug(f"[{request_id}] File bytes deleted")
        
        if image_base64 is not None:
            del image_base64
            logger.debug(f"[{request_id}] Base64 image data deleted")
        
        if extracted_base64 is not None:
            del extracted_base64
        
        # Force garbage collection to ensure memory is freed
        # This is recommended for long-running processes (Requirement 13.3)
        gc.collect()
        logger.debug(f"[{request_id}] Garbage collection completed")
        
        # Log metadata only (no image data, embeddings, or raw coordinates)
        # Requirement 13.4: Log only: request_id, timestamp, processing time, error codes
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"[{request_id}] Request completed: "
            f"status={'success' if analysis_succeeded else 'error'}, "
            f"processing_time_ms={elapsed_ms}"
        )


async def _validate_request_format(
    request_id: str,
    file: Optional[UploadFile],
    image_base64: Optional[str]
) -> None:
    """
    Validate that exactly one input method is provided.
    
    Requirements: 14.1, 14.2
    - Must provide either multipart file OR base64 string, not both or neither
    
    Args:
        request_id: Request identifier for logging
        file: Multipart file upload (if provided)
        image_base64: Base64-encoded image string (if provided)
    
    Raises:
        ValidationError: If both or neither inputs are provided
    """
    has_file = file is not None
    has_base64 = image_base64 is not None and image_base64.strip() != ""
    
    if has_file and has_base64:
        logger.warning(
            f"[{request_id}] Both file and base64 provided; only one input method allowed"
        )
        raise ValidationError(
            ErrorCode.INVALID_REQUEST,
            "Provide either a file upload or base64 image string, not both"
        )
    
    if not has_file and not has_base64:
        logger.warning(f"[{request_id}] No image input provided")
        raise ValidationError(
            ErrorCode.NO_IMAGE,
            "No image provided. Please upload a file or provide a base64-encoded image."
        )
    
    logger.debug(
        f"[{request_id}] Request format valid: "
        f"file={'yes' if has_file else 'no'}, base64={'yes' if has_base64 else 'no'}"
    )


async def _validate_and_read_file(
    request_id: str,
    file: UploadFile
) -> bytes:
    """
    Validate and read uploaded file.
    
    Validation includes:
    - File size (max 10MB)
    - MIME type (image/* only)
    - File extension (JPEG, PNG, HEIC, WebP, BMP, GIF)
    
    Requirements: 14.1, 14.2
    
    Args:
        request_id: Request identifier for logging
        file: Uploaded file object
    
    Returns:
        File bytes
    
    Raises:
        ValidationError: If file doesn't meet requirements
    """
    # Get file size
    # Note: UploadFile doesn't provide direct size, so we track as we read
    max_file_size_bytes = 10 * 1024 * 1024  # 10MB
    
    # Validate MIME type
    if file.content_type is None or not file.content_type.startswith('image/'):
        logger.warning(
            f"[{request_id}] Invalid MIME type: {file.content_type}"
        )
        raise ValidationError(
            ErrorCode.INVALID_IMAGE,
            f"Invalid file type: {file.content_type}. Please upload an image file."
        )
    
    # Validate file extension
    filename = file.filename or ""
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.heic', '.webp', '.bmp', '.gif'}
    file_ext = None
    if '.' in filename:
        file_ext = '.' + filename.rsplit('.', 1)[-1].lower()
    
    if file_ext not in allowed_extensions:
        logger.warning(
            f"[{request_id}] Invalid file extension: {file_ext}"
        )
        raise ValidationError(
            ErrorCode.INVALID_IMAGE,
            f"Unsupported file format: {file_ext}. Supported formats: JPEG, PNG, HEIC, WebP, BMP, GIF"
        )
    
    # Read file while checking size
    file_bytes = b""
    try:
        while True:
            chunk = await file.read(8192)  # Read in 8KB chunks
            if not chunk:
                break
            file_bytes += chunk
            if len(file_bytes) > max_file_size_bytes:
                logger.warning(
                    f"[{request_id}] File size exceeds limit: {len(file_bytes)} > {max_file_size_bytes}"
                )
                raise ValidationError(
                    ErrorCode.FILE_TOO_LARGE,
                    f"File size exceeds 10MB limit. Your file: {len(file_bytes) / (1024*1024):.1f}MB"
                )
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error reading file: {str(e)}")
        raise ValidationError(
            ErrorCode.INVALID_IMAGE,
            "Failed to read uploaded file"
        )
    
    if not file_bytes:
        logger.warning(f"[{request_id}] Empty file uploaded")
        raise ValidationError(
            ErrorCode.INVALID_IMAGE,
            "Uploaded file is empty"
        )
    
    logger.debug(
        f"[{request_id}] File validated: {filename}, "
        f"size={len(file_bytes) / 1024:.1f}KB, type={file.content_type}"
    )
    
    return file_bytes



async def _process_analysis(
    request_id: str,
    file_bytes: Optional[bytes],
    image_base64: Optional[str],
    include_celebrity: bool
) -> AnalysisResult:
    """
    Internal processing orchestration with privacy-preserving cleanup.
    
    Implements Requirements 13.1, 13.2, 13.3:
    - Process images in-memory only (no disk writes)
    - Delete image data after processing completes
    - Force garbage collection to ensure memory is freed
    - Cleanup happens even if processing fails or timeout occurs (via finally block)
    
    Args:
        request_id: Request identifier for logging
        file_bytes: Raw image bytes (if file upload)
        image_base64: Base64-encoded image (if base64 upload)
        include_celebrity: Whether to perform celebrity matching
    
    Returns:
        AnalysisResult with all analysis data
    
    Raises:
        ValidationError: If processing fails at any step
    """
    metadata = ProcessingMetadata(
        request_id=request_id,
        start_time=time.time(),
        component_times={}
    )
    
    # Local variables to track and cleanup
    image_data = None
    
    try:
        # === 1. Image Preprocessing ===
        component_start = time.time()
        
        # Process image (accepts either file bytes or base64)
        # Image is processed in-memory via BytesIO (Requirement 13.1)
        image_data = await image_processor.process_upload(
            file_bytes=file_bytes,
            base64_str=image_base64
        )
        
        metadata.add_component_time("image_processing", int((time.time() - component_start) * 1000))
        logger.debug(f"[{request_id}] Image processed: {image_data.width}x{image_data.height}")
        
        # === 2. Face Detection and Landmark Extraction ===
        component_start = time.time()
        
        face_detector = get_face_detector()
        landmarks = face_detector.detect_faces(image_data)
        
        metadata.add_component_time("face_detection", int((time.time() - component_start) * 1000))
        logger.debug(f"[{request_id}] Face detected with {len(landmarks.points)} landmarks")
        
        # Keep image_data for celebrity matching if enabled
        # Memory cleanup will happen at the end in finally block (Requirement 13.2, 13.3)
        
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
                    celebrity_match_internal = await celebrity_matcher.find_match(
                        image_data=image_data,
                        landmarks=landmarks
                    )
                    metadata.add_component_time("celebrity_matching", int((time.time() - component_start) * 1000))
                else:
                    logger.debug(f"[{request_id}] Celebrity matching skipped: unavailable")
            
            except Exception as e:
                # Graceful degradation - don't fail entire request
                # Requirement 17.1, 17.2: Celebrity matching can fail without blocking analysis
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
                vertical_midline=overlay_data_internal.midline_x,
                horizontal_thirds=overlay_data_internal.horizontal_thirds_y,
                phi_spiral_center=list(overlay_data_internal.phi_spiral_center)
            )
        )
        
        # Convert celebrity match
        celebrity_match = None
        if celebrity_match_internal:
            def normalize_thumb_url(raw_path):
                if not raw_path:
                    return None
                norm_path = raw_path.replace('\\', '/')
                if 'celebrity_dataset/' in norm_path:
                    rel_part = norm_path.split('celebrity_dataset/')[-1]
                    return f"/celebrities/{rel_part}"
                elif not norm_path.startswith('http') and not norm_path.startswith('/'):
                    return f"/celebrities/{norm_path}"
                return norm_path

            thumb_url = normalize_thumb_url(celebrity_match_internal.thumbnail_path)
            
            top_matches = []
            if celebrity_match_internal.top_matches:
                for item in celebrity_match_internal.top_matches:
                    item_thumb = normalize_thumb_url(item.get("thumbnail_path"))
                    sim_val = item.get("similarity", item.get("confidence", 0.0))
                    top_matches.append(
                        CelebrityMatchItem(
                            rank=item["rank"],
                            name=item["name"],
                            similarity=sim_val,
                            strength=item.get("strength", "weak"),
                            confidence=item.get("confidence", sim_val),
                            dataset_source=item.get("dataset_source"),
                            thumbnail_url=item_thumb
                        )
                    )
            
            celebrity_match = CelebrityMatch(
                name=celebrity_match_internal.name,
                similarity=celebrity_match_internal.similarity,
                strength=celebrity_match_internal.strength,
                confidence=celebrity_match_internal.confidence,
                dataset_source=celebrity_match_internal.dataset_source,
                thumbnail_url=thumb_url,
                message=celebrity_match_internal.message,
                top_matches=top_matches
            )
        
        # === 7. Log Metadata Only (Requirement 13.4) ===
        # Do NOT log:
        # - Image bytes or pixel data
        # - Face embeddings
        # - Raw landmark coordinates
        # - Personally identifiable information
        #
        # DO log:
        # - request_id, timestamp, processing_time
        # - Component names and their processing times
        # - Error codes (no technical details)
        
        logger.info(
            f"[{request_id}] Component times: {metadata.component_times}"
        )
        
        # Return complete analysis result (no image data in response)
        return AnalysisResult(
            golden_ratio_score=golden_ratio_result.overall_score,
            percentile_rank=percentile,
            face_symmetry_type=SymmetryTypeEnum(symmetry_type.value),
            feature_breakdown=feature_scores,
            overlay_data=overlay_data,
            celebrity_match=celebrity_match
        )
    
    except Exception as e:
        # Even on error, cleanup image data (Requirement 13.2)
        logger.debug(f"[{request_id}] Error in _process_analysis: {str(e)}")
        
        # Re-raise the exception so caller can handle it
        raise
    
    finally:
        # Ensure image_data is cleaned up even if exception occurs
        # This implements Requirement 13.2: "Image deleted even if processing fails"
        if image_data is not None:
            try:
                del image_data
                logger.debug(f"[{request_id}] Image data deleted in finally block")
            except Exception as e:
                logger.warning(f"[{request_id}] Error deleting image_data: {str(e)}")



@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """
    Handle ValidationError exceptions.
    
    Logs error with metadata only (no image data):
    - request_id for tracing
    - error_code for categorization
    - timestamp implicit in JSON formatter
    
    Requirements: 15.1, 15.4, 15.5
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.warning(
        f"[{request_id}] Validation error: {exc.error_code.value}",
        extra={
            "request_id": request_id,
            "component": "request_validation",
            "error_code": exc.error_code.value
        }
    )
    
    content = exc.to_dict()
    content["detail"] = {
        "status": content.get("status", "error"),
        "error_code": content.get("error_code"),
        "message": content.get("message"),
        "retry": content.get("retry", True)
    }
    return JSONResponse(
        status_code=400,
        content=content
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict):
        if "error_code" in detail:
            content = {
                "status": detail.get("status", "error"),
                "error_code": detail["error_code"],
                "message": detail.get("message", str(detail)),
                "retry": detail.get("retry", True),
                "detail": detail
            }
        elif "detail" in detail and isinstance(detail["detail"], dict) and "error_code" in detail["detail"]:
            d = detail["detail"]
            content = {
                "status": d.get("status", "error"),
                "error_code": d["error_code"],
                "message": d.get("message", str(d)),
                "retry": d.get("retry", True),
                "detail": d
            }
        else:
            error_code = "INVALID_REQUEST" if exc.status_code == 400 else "INTERNAL_ERROR"
            content = {
                "status": "error",
                "error_code": error_code,
                "message": str(detail),
                "retry": True,
                "detail": str(detail)
            }
    else:
        error_code = "INVALID_REQUEST" if exc.status_code == 400 else "INTERNAL_ERROR"
        content = {
            "status": "error",
            "error_code": error_code,
            "message": str(detail),
            "retry": True,
            "detail": str(detail)
        }
    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.warning(f"[{request_id}] Request validation error: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "status": "error",
            "error_code": ErrorCode.INVALID_REQUEST.value,
            "message": f"Invalid request parameters: {str(exc)}",
            "retry": True
        }
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
