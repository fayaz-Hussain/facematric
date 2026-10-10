"""
Unit tests for Pydantic response models.

Tests validation logic, serialization, nested structures, and field constraints
for all API response schemas in the FaceMatric backend.

Requirements tested:
- 14.1: API returns HTTP 200 with application/json
- 14.2: Successful response includes status field with "success"
- 14.3: Response includes processing_time_ms integer field
- 14.4: Response includes results object with all required fields
- 14.5: feature_breakdown contains 8 feature objects
- 14.6: overlay_data includes landmarks and golden_ratio_guides
- 14.7: celebrity_match includes name, confidence, dataset_source, thumbnail_url
- 14.8: celebrity_match can be null when no match found or feature unavailable
- 15.1: Error response includes appropriate HTTP status codes
- 15.2: Error response includes status field with "error"
- 15.3: Error response includes error_code field with machine-readable code
- 15.4: Error response includes message field with user-friendly guidance
- 15.5: Error response includes retry boolean field
"""

import pytest
from pydantic import ValidationError
from app.models.responses import (
    ErrorCodeEnum,
    SymmetryTypeEnum,
    FeatureBreakdownItem,
    GoldenRatioGuides,
    OverlayData,
    CelebrityMatch,
    AnalysisResultModel,
    AnalysisResponse,
    ErrorResponse,
    HealthCheckResponse
)


class TestErrorCodeEnum:
    """Test ErrorCodeEnum (Requirement 15.3)."""
    
    def test_all_error_codes_exist(self):
        """Verify all required error codes exist."""
        assert ErrorCodeEnum.NO_FACE_DETECTED.value == "NO_FACE_DETECTED"
        assert ErrorCodeEnum.MULTIPLE_FACES.value == "MULTIPLE_FACES"
        assert ErrorCodeEnum.POOR_QUALITY.value == "POOR_QUALITY"
        assert ErrorCodeEnum.OCCLUSION_DETECTED.value == "OCCLUSION_DETECTED"
        assert ErrorCodeEnum.INVALID_IMAGE.value == "INVALID_IMAGE"
        assert ErrorCodeEnum.PROCESSING_TIMEOUT.value == "PROCESSING_TIMEOUT"
        assert ErrorCodeEnum.INTERNAL_ERROR.value == "INTERNAL_ERROR"
    
    def test_error_code_is_string_enum(self):
        """Test that error codes are string values."""
        for code in ErrorCodeEnum:
            assert isinstance(code.value, str)


class TestSymmetryTypeEnum:
    """Test SymmetryTypeEnum (Requirement 14.4)."""
    
    def test_all_symmetry_types_exist(self):
        """Verify all 6 symmetry types exist."""
        assert SymmetryTypeEnum.OVAL.value == "oval"
        assert SymmetryTypeEnum.RECTANGLE.value == "rectangle"
        assert SymmetryTypeEnum.ROUND.value == "round"
        assert SymmetryTypeEnum.SQUARE.value == "square"
        assert SymmetryTypeEnum.HEART.value == "heart"
        assert SymmetryTypeEnum.DIAMOND.value == "diamond"
    
    def test_symmetry_type_count(self):
        """Verify exactly 6 symmetry types exist."""
        assert len(SymmetryTypeEnum) == 6


class TestFeatureBreakdownItem:
    """Test FeatureBreakdownItem (Requirement 14.5)."""
    
    def test_valid_feature_breakdown_item(self):
        """Test creating valid feature breakdown item."""
        item = FeatureBreakdownItem(
            feature="Face Length ÷ Face Width",
            measured_ratio=1.625,
            ideal_ratio=1.618,
            score=99.6,
            deviation=0.007
        )
        
        assert item.feature == "Face Length ÷ Face Width"
        assert item.measured_ratio == 1.625
        assert item.ideal_ratio == 1.618
        assert item.score == 99.6
        assert item.deviation == 0.007
    
    def test_score_bounds_validation(self):
        """Test that score must be within [0, 100] range."""
        # Score too high
        with pytest.raises(ValidationError) as exc_info:
            FeatureBreakdownItem(
                feature="Test",
                measured_ratio=1.5,
                ideal_ratio=1.618,
                score=105.0,
                deviation=-0.118
            )
        assert "score" in str(exc_info.value).lower()
        
        # Score too low
        with pytest.raises(ValidationError) as exc_info:
            FeatureBreakdownItem(
                feature="Test",
                measured_ratio=1.5,
                ideal_ratio=1.618,
                score=-5.0,
                deviation=-0.118
            )
        assert "score" in str(exc_info.value).lower()
    
    def test_score_boundary_values(self):
        """Test that score boundaries (0 and 100) are valid."""
        # Score = 0
        item_min = FeatureBreakdownItem(
            feature="Test Min",
            measured_ratio=0.0,
            ideal_ratio=1.618,
            score=0.0,
            deviation=-1.618
        )
        assert item_min.score == 0.0
        
        # Score = 100
        item_max = FeatureBreakdownItem(
            feature="Test Max",
            measured_ratio=1.618,
            ideal_ratio=1.618,
            score=100.0,
            deviation=0.0
        )
        assert item_max.score == 100.0
    
    def test_serialization(self):
        """Test that item serializes to correct JSON structure."""
        item = FeatureBreakdownItem(
            feature="Eye Spacing ÷ Eye Width",
            measured_ratio=1.650,
            ideal_ratio=1.618,
            score=98.0,
            deviation=0.032
        )
        
        data = item.model_dump()
        
        assert data["feature"] == "Eye Spacing ÷ Eye Width"
        assert data["measured_ratio"] == 1.650
        assert data["ideal_ratio"] == 1.618
        assert data["score"] == 98.0
        assert data["deviation"] == 0.032


class TestGoldenRatioGuides:
    """Test GoldenRatioGuides (Requirement 14.6)."""
    
    def test_valid_golden_ratio_guides(self):
        """Test creating valid golden ratio guides."""
        guides = GoldenRatioGuides(
            vertical_midline=320.5,
            horizontal_thirds=[50.0, 120.0, 240.0, 360.0],
            phi_spiral_center=[320.5, 180.0]
        )
        
        assert guides.vertical_midline == 320.5
        assert len(guides.horizontal_thirds) == 4
        assert len(guides.phi_spiral_center) == 2
    
    def test_horizontal_thirds_must_have_4_values(self):
        """Test that horizontal_thirds requires exactly 4 values."""
        # Too few values
        with pytest.raises(ValidationError) as exc_info:
            GoldenRatioGuides(
                vertical_midline=320.0,
                horizontal_thirds=[50.0, 120.0, 240.0],  # Only 3
                phi_spiral_center=[320.0, 180.0]
            )
        assert "horizontal_thirds" in str(exc_info.value).lower()
        
        # Too many values
        with pytest.raises(ValidationError) as exc_info:
            GoldenRatioGuides(
                vertical_midline=320.0,
                horizontal_thirds=[50.0, 120.0, 240.0, 360.0, 480.0],  # 5
                phi_spiral_center=[320.0, 180.0]
            )
        assert "horizontal_thirds" in str(exc_info.value).lower()
    
    def test_phi_spiral_center_must_have_2_values(self):
        """Test that phi_spiral_center requires exactly 2 values (x, y)."""
        # Only 1 value
        with pytest.raises(ValidationError) as exc_info:
            GoldenRatioGuides(
                vertical_midline=320.0,
                horizontal_thirds=[50.0, 120.0, 240.0, 360.0],
                phi_spiral_center=[320.0]  # Missing y
            )
        assert "phi_spiral_center" in str(exc_info.value).lower()
        
        # Too many values
        with pytest.raises(ValidationError) as exc_info:
            GoldenRatioGuides(
                vertical_midline=320.0,
                horizontal_thirds=[50.0, 120.0, 240.0, 360.0],
                phi_spiral_center=[320.0, 180.0, 50.0]  # Extra value
            )
        assert "phi_spiral_center" in str(exc_info.value).lower()
    
    def test_serialization(self):
        """Test golden ratio guides serialization."""
        guides = GoldenRatioGuides(
            vertical_midline=315.0,
            horizontal_thirds=[45.0, 160.0, 275.0, 390.0],
            phi_spiral_center=[315.0, 175.5]
        )
        
        data = guides.model_dump()
        
        assert data["vertical_midline"] == 315.0
        assert data["horizontal_thirds"] == [45.0, 160.0, 275.0, 390.0]
        assert data["phi_spiral_center"] == [315.0, 175.5]


class TestOverlayData:
    """Test OverlayData (Requirement 14.6)."""
    
    def test_valid_overlay_data(self):
        """Test creating valid overlay data."""
        landmarks = [[100.5, 200.3], [101.2, 201.5], [102.0, 202.8]]
        guides = GoldenRatioGuides(
            vertical_midline=320.5,
            horizontal_thirds=[50.0, 165.5, 280.8, 396.0],
            phi_spiral_center=[320.5, 180.0]
        )
        
        overlay = OverlayData(
            landmarks=landmarks,
            golden_ratio_guides=guides
        )
        
        assert len(overlay.landmarks) == 3
        assert overlay.golden_ratio_guides.vertical_midline == 320.5
    
    def test_landmarks_are_list_of_coordinate_pairs(self):
        """Test that landmarks is a list of [x, y] coordinate pairs."""
        guides = GoldenRatioGuides(
            vertical_midline=320.0,
            horizontal_thirds=[50.0, 120.0, 240.0, 360.0],
            phi_spiral_center=[320.0, 180.0]
        )
        
        overlay = OverlayData(
            landmarks=[[x, x + 100] for x in range(468)],  # 468 landmarks
            golden_ratio_guides=guides
        )
        
        assert len(overlay.landmarks) == 468
        assert len(overlay.landmarks[0]) == 2
    
    def test_nested_structure_serialization(self):
        """Test that nested golden_ratio_guides serializes correctly."""
        guides = GoldenRatioGuides(
            vertical_midline=320.0,
            horizontal_thirds=[50.0, 165.5, 280.8, 396.0],
            phi_spiral_center=[320.0, 180.0]
        )
        
        overlay = OverlayData(
            landmarks=[[100.0, 200.0]],
            golden_ratio_guides=guides
        )
        
        data = overlay.model_dump()
        
        assert "landmarks" in data
        assert "golden_ratio_guides" in data
        assert "vertical_midline" in data["golden_ratio_guides"]
        assert "horizontal_thirds" in data["golden_ratio_guides"]
        assert "phi_spiral_center" in data["golden_ratio_guides"]


class TestCelebrityMatch:
    """Test CelebrityMatch (Requirements 14.7, 14.8)."""
    
    def test_valid_celebrity_match(self):
        """Test creating valid celebrity match (Requirement 14.7)."""
        match = CelebrityMatch(
            name="Ryan Gosling",
            confidence=0.72,
            dataset_source="Western",
            thumbnail_url="/celebrities/ryan_gosling.jpg",
            message=None
        )
        
        assert match.name == "Ryan Gosling"
        assert match.confidence == 0.72
        assert match.dataset_source == "Western"
        assert match.thumbnail_url == "/celebrities/ryan_gosling.jpg"
        assert match.message is None
    
    def test_no_match_case(self):
        """Test celebrity match when no match found (Requirement 14.8)."""
        match = CelebrityMatch(
            name=None,
            confidence=0.48,
            dataset_source=None,
            thumbnail_url=None,
            message="No close match found in our dataset"
        )
        
        assert match.name is None
        assert match.confidence == 0.48
        assert match.message == "No close match found in our dataset"
    
    def test_all_fields_optional(self):
        """Test that all celebrity match fields are optional (Requirement 14.8)."""
        match = CelebrityMatch()
        
        assert match.name is None
        assert match.confidence is None
        assert match.dataset_source is None
        assert match.thumbnail_url is None
        assert match.message is None
    
    def test_confidence_bounds_validation(self):
        """Test that confidence must be within [0, 1] range."""
        # Confidence too high
        with pytest.raises(ValidationError) as exc_info:
            CelebrityMatch(
                name="Test",
                confidence=1.5,
                dataset_source="Test",
                thumbnail_url="/test.jpg"
            )
        assert "confidence" in str(exc_info.value).lower()
        
        # Confidence negative
        with pytest.raises(ValidationError) as exc_info:
            CelebrityMatch(
                name="Test",
                confidence=-0.1,
                dataset_source="Test",
                thumbnail_url="/test.jpg"
            )
        assert "confidence" in str(exc_info.value).lower()
    
    def test_confidence_boundary_values(self):
        """Test that confidence boundaries (0.0 and 1.0) are valid."""
        match_min = CelebrityMatch(confidence=0.0)
        assert match_min.confidence == 0.0
        
        match_max = CelebrityMatch(confidence=1.0)
        assert match_max.confidence == 1.0
    
    def test_serialization_with_match(self):
        """Test serialization when match is found."""
        match = CelebrityMatch(
            name="Emma Watson",
            confidence=0.85,
            dataset_source="Western",
            thumbnail_url="/celebrities/emma_watson.jpg"
        )
        
        data = match.model_dump()
        
        assert data["name"] == "Emma Watson"
        assert data["confidence"] == 0.85
        assert data["dataset_source"] == "Western"
        assert data["thumbnail_url"] == "/celebrities/emma_watson.jpg"
    
    def test_serialization_without_match(self):
        """Test serialization when no match found."""
        match = CelebrityMatch(
            name=None,
            confidence=None,
            message="No close match found in our dataset"
        )
        
        data = match.model_dump()
        
        assert data["name"] is None
        assert data["confidence"] is None
        assert data["message"] == "No close match found in our dataset"


class TestAnalysisResultModel:
    """Test AnalysisResultModel (Requirement 14.4)."""
    
    def create_valid_feature_breakdown(self):
        """Helper to create valid feature breakdown list."""
        return [
            FeatureBreakdownItem(
                feature=f"Feature {i+1}",
                measured_ratio=1.6 + (i * 0.01),
                ideal_ratio=1.618,
                score=90.0 + i,
                deviation=0.018 - (i * 0.01)
            )
            for i in range(8)
        ]
    
    def test_valid_analysis_result(self):
        """Test creating valid analysis result with all required fields."""
        feature_breakdown = self.create_valid_feature_breakdown()
        overlay = OverlayData(
            landmarks=[[100.0, 200.0]],
            golden_ratio_guides=GoldenRatioGuides(
                vertical_midline=320.0,
                horizontal_thirds=[50.0, 165.5, 280.8, 396.0],
                phi_spiral_center=[320.0, 180.0]
            )
        )
        celebrity = CelebrityMatch(
            name="Test Celebrity",
            confidence=0.75,
            dataset_source="Western",
            thumbnail_url="/test.jpg"
        )
        
        result = AnalysisResultModel(
            golden_ratio_score=78.3,
            percentile_rank=82,
            face_symmetry_type=SymmetryTypeEnum.OVAL,
            feature_breakdown=feature_breakdown,
            overlay_data=overlay,
            celebrity_match=celebrity
        )
        
        assert result.golden_ratio_score == 78.3
        assert result.percentile_rank == 82
        assert result.face_symmetry_type == SymmetryTypeEnum.OVAL
        assert len(result.feature_breakdown) == 8
        assert result.overlay_data is not None
        assert result.celebrity_match is not None
    
    def test_golden_ratio_score_bounds(self):
        """Test that golden_ratio_score must be within [0, 100]."""
        feature_breakdown = self.create_valid_feature_breakdown()
        overlay = OverlayData(
            landmarks=[[100.0, 200.0]],
            golden_ratio_guides=GoldenRatioGuides(
                vertical_midline=320.0,
                horizontal_thirds=[50.0, 165.5, 280.8, 396.0],
                phi_spiral_center=[320.0, 180.0]
            )
        )
        
        # Score too high
        with pytest.raises(ValidationError) as exc_info:
            AnalysisResultModel(
                golden_ratio_score=105.0,
                percentile_rank=50,
                face_symmetry_type=SymmetryTypeEnum.OVAL,
                feature_breakdown=feature_breakdown,
                overlay_data=overlay
            )
        assert "golden_ratio_score" in str(exc_info.value).lower()
        
        # Score negative
        with pytest.raises(ValidationError) as exc_info:
            AnalysisResultModel(
                golden_ratio_score=-5.0,
                percentile_rank=50,
                face_symmetry_type=SymmetryTypeEnum.OVAL,
                feature_breakdown=feature_breakdown,
                overlay_data=overlay
            )
        assert "golden_ratio_score" in str(exc_info.value).lower()
    
    def test_percentile_rank_bounds(self):
        """Test that percentile_rank must be within [0, 100]."""
        feature_breakdown = self.create_valid_feature_breakdown()
        overlay = OverlayData(
            landmarks=[[100.0, 200.0]],
            golden_ratio_guides=GoldenRatioGuides(
                vertical_midline=320.0,
                horizontal_thirds=[50.0, 165.5, 280.8, 396.0],
                phi_spiral_center=[320.0, 180.0]
            )
        )
        
        # Percentile too high
        with pytest.raises(ValidationError) as exc_info:
            AnalysisResultModel(
                golden_ratio_score=75.0,
                percentile_rank=150,
                face_symmetry_type=SymmetryTypeEnum.OVAL,
                feature_breakdown=feature_breakdown,
                overlay_data=overlay
            )
        assert "percentile_rank" in str(exc_info.value).lower()
        
        # Percentile negative
        with pytest.raises(ValidationError) as exc_info:
            AnalysisResultModel(
                golden_ratio_score=75.0,
                percentile_rank=-10,
                face_symmetry_type=SymmetryTypeEnum.OVAL,
                feature_breakdown=feature_breakdown,
                overlay_data=overlay
            )
        assert "percentile_rank" in str(exc_info.value).lower()
    
    def test_feature_breakdown_must_have_8_items(self):
        """Test that feature_breakdown requires exactly 8 items (Requirement 14.5)."""
        overlay = OverlayData(
            landmarks=[[100.0, 200.0]],
            golden_ratio_guides=GoldenRatioGuides(
                vertical_midline=320.0,
                horizontal_thirds=[50.0, 165.5, 280.8, 396.0],
                phi_spiral_center=[320.0, 180.0]
            )
        )
        
        # Only 7 features
        feature_breakdown_short = self.create_valid_feature_breakdown()[:7]
        
        with pytest.raises(ValidationError) as exc_info:
            AnalysisResultModel(
                golden_ratio_score=75.0,
                percentile_rank=70,
                face_symmetry_type=SymmetryTypeEnum.OVAL,
                feature_breakdown=feature_breakdown_short,
                overlay_data=overlay
            )
        assert "feature_breakdown" in str(exc_info.value).lower()
    
    def test_celebrity_match_is_optional(self):
        """Test that celebrity_match can be None (Requirement 14.8)."""
        feature_breakdown = self.create_valid_feature_breakdown()
        overlay = OverlayData(
            landmarks=[[100.0, 200.0]],
            golden_ratio_guides=GoldenRatioGuides(
                vertical_midline=320.0,
                horizontal_thirds=[50.0, 165.5, 280.8, 396.0],
                phi_spiral_center=[320.0, 180.0]
            )
        )
        
        result = AnalysisResultModel(
            golden_ratio_score=75.0,
            percentile_rank=70,
            face_symmetry_type=SymmetryTypeEnum.OVAL,
            feature_breakdown=feature_breakdown,
            overlay_data=overlay,
            celebrity_match=None  # Progressive degradation
        )
        
        assert result.celebrity_match is None
    
    def test_complex_nested_serialization(self):
        """Test serialization of deeply nested structure."""
        feature_breakdown = self.create_valid_feature_breakdown()
        overlay = OverlayData(
            landmarks=[[x, x + 100] for x in range(468)],
            golden_ratio_guides=GoldenRatioGuides(
                vertical_midline=320.0,
                horizontal_thirds=[50.0, 165.5, 280.8, 396.0],
                phi_spiral_center=[320.0, 180.0]
            )
        )
        celebrity = CelebrityMatch(
            name="Test",
            confidence=0.65,
            dataset_source="Bollywood",
            thumbnail_url="/test.jpg"
        )
        
        result = AnalysisResultModel(
            golden_ratio_score=78.5,
            percentile_rank=82,
            face_symmetry_type=SymmetryTypeEnum.RECTANGLE,
            feature_breakdown=feature_breakdown,
            overlay_data=overlay,
            celebrity_match=celebrity
        )
        
        data = result.model_dump()
        
        # Verify top-level fields
        assert data["golden_ratio_score"] == 78.5
        assert data["percentile_rank"] == 82
        assert data["face_symmetry_type"] == "rectangle"
        
        # Verify nested feature_breakdown
        assert len(data["feature_breakdown"]) == 8
        assert "feature" in data["feature_breakdown"][0]
        assert "measured_ratio" in data["feature_breakdown"][0]
        
        # Verify nested overlay_data
        assert "landmarks" in data["overlay_data"]
        assert "golden_ratio_guides" in data["overlay_data"]
        assert len(data["overlay_data"]["landmarks"]) == 468
        
        # Verify nested celebrity_match
        assert data["celebrity_match"]["name"] == "Test"
        assert data["celebrity_match"]["confidence"] == 0.65


class TestAnalysisResponse:
    """Test AnalysisResponse (Requirements 14.1, 14.2, 14.3)."""
    
    def create_valid_analysis_result(self):
        """Helper to create valid AnalysisResultModel."""
        feature_breakdown = [
            FeatureBreakdownItem(
                feature=f"Feature {i+1}",
                measured_ratio=1.6 + (i * 0.01),
                ideal_ratio=1.618,
                score=90.0 + i,
                deviation=0.018 - (i * 0.01)
            )
            for i in range(8)
        ]
        
        overlay = OverlayData(
            landmarks=[[100.0, 200.0]],
            golden_ratio_guides=GoldenRatioGuides(
                vertical_midline=320.0,
                horizontal_thirds=[50.0, 165.5, 280.8, 396.0],
                phi_spiral_center=[320.0, 180.0]
            )
        )
        
        return AnalysisResultModel(
            golden_ratio_score=78.3,
            percentile_rank=82,
            face_symmetry_type=SymmetryTypeEnum.OVAL,
            feature_breakdown=feature_breakdown,
            overlay_data=overlay
        )
    
    def test_valid_analysis_response(self):
        """Test creating valid analysis response."""
        result = self.create_valid_analysis_result()
        
        response = AnalysisResponse(
            status="success",
            processing_time_ms=2847,
            results=result
        )
        
        assert response.status == "success"
        assert response.processing_time_ms == 2847
        assert response.results is not None
    
    def test_status_defaults_to_success(self):
        """Test that status field defaults to 'success' (Requirement 14.2)."""
        result = self.create_valid_analysis_result()
        
        response = AnalysisResponse(
            processing_time_ms=2000,
            results=result
        )
        
        assert response.status == "success"
    
    def test_processing_time_is_integer(self):
        """Test that processing_time_ms is an integer (Requirement 14.3)."""
        result = self.create_valid_analysis_result()
        
        response = AnalysisResponse(
            processing_time_ms=2847,
            results=result
        )
        
        assert isinstance(response.processing_time_ms, int)
        assert response.processing_time_ms == 2847
    
    def test_serialization_complete_response(self):
        """Test serialization of complete successful response."""
        result = self.create_valid_analysis_result()
        
        response = AnalysisResponse(
            status="success",
            processing_time_ms=2847,
            results=result
        )
        
        data = response.model_dump()
        
        assert data["status"] == "success"
        assert data["processing_time_ms"] == 2847
        assert "results" in data
        assert "golden_ratio_score" in data["results"]
        assert "percentile_rank" in data["results"]
        assert "face_symmetry_type" in data["results"]
        assert "feature_breakdown" in data["results"]
        assert "overlay_data" in data["results"]


class TestErrorResponse:
    """Test ErrorResponse (Requirements 15.1-15.5)."""
    
    def test_valid_error_response(self):
        """Test creating valid error response."""
        response = ErrorResponse(
            status="error",
            error_code=ErrorCodeEnum.NO_FACE_DETECTED,
            message="We couldn't detect a face in your photo.",
            retry=True
        )
        
        assert response.status == "error"
        assert response.error_code == ErrorCodeEnum.NO_FACE_DETECTED
        assert "face" in response.message.lower()
        assert response.retry is True
    
    def test_status_defaults_to_error(self):
        """Test that status field defaults to 'error' (Requirement 15.2)."""
        response = ErrorResponse(
            error_code=ErrorCodeEnum.INVALID_IMAGE,
            message="Invalid image",
            retry=True
        )
        
        assert response.status == "error"
    
    def test_error_code_is_machine_readable(self):
        """Test that error_code is machine-readable enum (Requirement 15.3)."""
        response = ErrorResponse(
            error_code=ErrorCodeEnum.MULTIPLE_FACES,
            message="Multiple faces detected.",
            retry=True
        )
        
        assert isinstance(response.error_code, ErrorCodeEnum)
        assert response.error_code.value == "MULTIPLE_FACES"
    
    def test_message_is_user_friendly(self):
        """Test that message provides user-friendly guidance (Requirement 15.4)."""
        response = ErrorResponse(
            error_code=ErrorCodeEnum.POOR_QUALITY,
            message="Photo quality is too low. Please ensure your face is well-lit.",
            retry=True
        )
        
        # Message should be descriptive and helpful
        assert len(response.message) > 20
        assert "." in response.message  # Proper punctuation
    
    def test_retry_field_exists(self):
        """Test that retry boolean field exists (Requirement 15.5)."""
        response = ErrorResponse(
            error_code=ErrorCodeEnum.PROCESSING_TIMEOUT,
            message="Processing timeout",
            retry=True
        )
        
        assert hasattr(response, 'retry')
        assert isinstance(response.retry, bool)
        assert response.retry is True
    
    def test_details_field_optional(self):
        """Test that details field is optional."""
        response = ErrorResponse(
            error_code=ErrorCodeEnum.OCCLUSION_DETECTED,
            message="Accessories detected",
            retry=True
        )
        
        assert response.details is None
        
        # With details
        response_with_details = ErrorResponse(
            error_code=ErrorCodeEnum.OCCLUSION_DETECTED,
            message="Accessories detected",
            retry=True,
            details={"detected": "glasses"}
        )
        
        assert response_with_details.details is not None
        assert response_with_details.details["detected"] == "glasses"
    
    def test_all_error_codes_can_be_used(self):
        """Test that all error codes work in ErrorResponse."""
        for error_code in ErrorCodeEnum:
            response = ErrorResponse(
                error_code=error_code,
                message=f"Test message for {error_code.value}",
                retry=True
            )
            
            assert response.error_code == error_code
    
    def test_serialization(self):
        """Test error response serialization."""
        response = ErrorResponse(
            status="error",
            error_code=ErrorCodeEnum.INVALID_IMAGE,
            message="Corrupt or unsupported file format",
            retry=True,
            details={"format": "unknown"}
        )
        
        data = response.model_dump()
        
        assert data["status"] == "error"
        assert data["error_code"] == "INVALID_IMAGE"
        assert data["message"] == "Corrupt or unsupported file format"
        assert data["retry"] is True
        assert data["details"]["format"] == "unknown"


class TestHealthCheckResponse:
    """Test HealthCheckResponse."""
    
    def test_valid_health_check_response(self):
        """Test creating valid health check response."""
        response = HealthCheckResponse(
            status="healthy",
            version="1.0.0",
            celebrity_matching_available=True
        )
        
        assert response.status == "healthy"
        assert response.version == "1.0.0"
        assert response.celebrity_matching_available is True
    
    def test_status_defaults_to_healthy(self):
        """Test that status defaults to 'healthy'."""
        response = HealthCheckResponse(
            version="1.0.0",
            celebrity_matching_available=False
        )
        
        assert response.status == "healthy"
    
    def test_celebrity_matching_can_be_unavailable(self):
        """Test health check when celebrity matching is unavailable."""
        response = HealthCheckResponse(
            status="healthy",
            version="1.0.0",
            celebrity_matching_available=False
        )
        
        assert response.celebrity_matching_available is False
    
    def test_serialization(self):
        """Test health check response serialization."""
        response = HealthCheckResponse(
            status="healthy",
            version="2.5.1",
            celebrity_matching_available=True
        )
        
        data = response.model_dump()
        
        assert data["status"] == "healthy"
        assert data["version"] == "2.5.1"
        assert data["celebrity_matching_available"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
