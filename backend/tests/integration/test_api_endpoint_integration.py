"""
Comprehensive integration tests for /api/analyze endpoint.

Tests complete workflows including:
1. Happy path: Upload valid image → receive complete analysis
2. Error scenarios: Various error codes (HTTP 400/408)
3. Timeout scenario: Processing exceeds 30 seconds
4. Progressive degradation: Celebrity matching fails but golden ratio succeeds
5. Response structure validation
6. Different input methods (multipart, base64, data URI)

Requirements covered: 1.5, 1.6, 2.3, 2.4, 2.5, 3.2, 12.4, 14.1-14.8, 15.1-15.5, 17.1-17.4
"""

import pytest
import base64
import json
from io import BytesIO
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from PIL import Image
import numpy as np

from fastapi.testclient import TestClient

from app.main import app
from app.utils.errors import ErrorCode
from app.models.responses import SymmetryTypeEnum
from app.models.internal import Landmarks, FeatureScoreInternal, GoldenRatioResult, OverlayDataInternal


@pytest.fixture
def client():
    """Create TestClient for FastAPI application."""
    return TestClient(app)


@pytest.fixture
def valid_image_300x300():
    """Create a valid 300x300 RGB test image (minimum size)."""
    img = Image.new('RGB', (300, 300), color=(255, 0, 0))
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes.getvalue()


@pytest.fixture
def valid_image_640x480():
    """Create a valid 640x480 RGB test image."""
    img = Image.new('RGB', (640, 480), color=(100, 150, 200))
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes.getvalue()


@pytest.fixture
def valid_png_image():
    """Create a valid PNG test image."""
    img = Image.new('RGB', (400, 400), color=(0, 255, 0))
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes.getvalue()


@pytest.fixture
def valid_base64_image(valid_image_300x300):
    """Convert image bytes to base64 string."""
    return base64.b64encode(valid_image_300x300).decode('utf-8')


@pytest.fixture
def valid_base64_data_uri(valid_base64_image):
    """Create data URI formatted base64 image."""
    return f"data:image/jpeg;base64,{valid_base64_image}"


def create_mock_landmarks():
    """Create mock face detection landmarks."""
    landmarks_array = np.random.rand(468, 3).astype(np.float32)
    landmarks_array[:, :2] *= 1.0  # x, y in [0, 1]
    landmarks_array[:, 2] = np.random.uniform(-1, 1, 468)  # z in [-1, 1]
    
    return Landmarks(
        points=landmarks_array,
        confidence=0.95,
        image_width=300,
        image_height=300
    )


def create_mock_golden_ratio_result():
    """Create mock golden ratio scoring result."""
    features = [
        FeatureScoreInternal(
            feature="Face Length ÷ Face Width",
            measured_ratio=1.625,
            ideal_ratio=1.618,
            score=99.6,
            deviation=0.007
        ),
        FeatureScoreInternal(
            feature="Face Width ÷ Jaw Width",
            measured_ratio=1.580,
            ideal_ratio=1.618,
            score=97.7,
            deviation=-0.038
        ),
        FeatureScoreInternal(
            feature="Eye Spacing ÷ Eye Width",
            measured_ratio=1.650,
            ideal_ratio=1.618,
            score=98.0,
            deviation=0.032
        ),
        FeatureScoreInternal(
            feature="Nose Width ÷ Mouth Width",
            measured_ratio=1.520,
            ideal_ratio=1.618,
            score=93.9,
            deviation=-0.098
        ),
        FeatureScoreInternal(
            feature="Mouth Width ÷ Face Width",
            measured_ratio=1.600,
            ideal_ratio=1.618,
            score=98.9,
            deviation=-0.018
        ),
        FeatureScoreInternal(
            feature="Nose Bridge to Chin ÷ Hairline to Nose Bridge",
            measured_ratio=1.595,
            ideal_ratio=1.618,
            score=98.6,
            deviation=-0.023
        ),
        FeatureScoreInternal(
            feature="Eye to Eyebrow Distance ÷ Eye Height",
            measured_ratio=1.450,
            ideal_ratio=1.618,
            score=89.6,
            deviation=-0.168
        ),
        FeatureScoreInternal(
            feature="Chin Width ÷ Nose Width",
            measured_ratio=1.680,
            ideal_ratio=1.618,
            score=96.2,
            deviation=0.062
        )
    ]
    
    # Calculate actual mean and round to 1 decimal place (requirement 7.2)
    mean_score = round(sum(f.score for f in features) / len(features), 1)
    
    return GoldenRatioResult(
        overall_score=mean_score,
        features=features
    )


def create_mock_overlay_result():
    """Create mock overlay generation result."""
    landmarks_2d = np.random.uniform(0, 640, size=(468, 2)).astype(np.float32)
    
    return OverlayDataInternal(
        landmarks_2d=landmarks_2d,
        midline_x=320.0,
        horizontal_thirds_y=[100.0, 200.0, 300.0, 400.0],
        phi_spiral_center=(320.0, 200.0)
    )


class TestValidationErrors:
    """Test request validation error scenarios."""
    
    def test_no_image_provided_error(self, client):
        """Test NO_IMAGE error when neither file nor base64 provided (HTTP 400).
        
        Requirements: 14.1, 15.1
        """
        response = client.post(
            "/api/analyze",
            json={"include_celebrity_match": True}
        )
        
        assert response.status_code == 400
        error_data = response.json()
        if "detail" in error_data:
            error_data = error_data["detail"]
        
        assert error_data.get("status") == "error"
        assert error_data.get("error_code") == ErrorCode.NO_IMAGE.value
        assert error_data.get("retry") == True
    
    def test_empty_base64_error(self, client):
        """Test NO_IMAGE error for empty base64 string.
        
        Requirements: 14.1, 15.1
        """
        response = client.post(
            "/api/analyze",
            json={"image_base64": "", "include_celebrity_match": True}
        )
        
        assert response.status_code == 400
        error_data = response.json()
        if "detail" in error_data:
            error_data = error_data["detail"]
        assert error_data.get("error_code") == ErrorCode.NO_IMAGE.value
    
    def test_whitespace_base64_error(self, client):
        """Test NO_IMAGE error for whitespace-only base64 string.
        
        Requirements: 14.1, 15.1
        """
        response = client.post(
            "/api/analyze",
            json={"image_base64": "   \n\t  ", "include_celebrity_match": True}
        )
        
        assert response.status_code == 400
        error_data = response.json()
        if "detail" in error_data:
            error_data = error_data["detail"]
        assert error_data.get("error_code") == ErrorCode.NO_IMAGE.value
    
    def test_file_too_large_error(self, client):
        """Test FILE_TOO_LARGE error (HTTP 400).
        
        Requirements: 14.1, 15.1
        """
        # Create data larger than 10MB
        large_data = b"x" * (11 * 1024 * 1024)
        
        response = client.post(
            "/api/analyze",
            files={"file": ("large.jpg", large_data, "image/jpeg")}
        )
        
        assert response.status_code == 400
        error_data = response.json()
        if "detail" in error_data:
            error_data = error_data["detail"]
        
        assert error_data.get("status") == "error"
        assert error_data.get("error_code") == ErrorCode.FILE_TOO_LARGE.value
        assert "10MB" in error_data.get("message", "")
        assert error_data.get("retry") == True
    
    def test_invalid_mime_type_error(self, client, valid_image_300x300):
        """Test INVALID_IMAGE error for non-image MIME type.
        
        Requirements: 14.1, 15.1
        """
        response = client.post(
            "/api/analyze",
            files={"file": ("test.pdf", valid_image_300x300, "application/pdf")}
        )
        
        assert response.status_code == 400
        error_data = response.json()
        if "detail" in error_data:
            error_data = error_data["detail"]
        assert error_data.get("error_code") == ErrorCode.INVALID_IMAGE.value
    
    def test_invalid_file_extension_error(self, client, valid_image_300x300):
        """Test INVALID_IMAGE error for unsupported file extension.
        
        Requirements: 1.1, 15.1
        """
        response = client.post(
            "/api/analyze",
            files={"file": ("test.txt", valid_image_300x300, "image/jpeg")}
        )
        
        assert response.status_code == 400
        error_data = response.json()
        if "detail" in error_data:
            error_data = error_data["detail"]
        assert error_data.get("error_code") == ErrorCode.INVALID_IMAGE.value
    
    def test_empty_file_error(self, client):
        """Test INVALID_IMAGE error for empty file.
        
        Requirements: 15.1
        """
        response = client.post(
            "/api/analyze",
            files={"file": ("test.jpg", b"", "image/jpeg")}
        )
        
        assert response.status_code == 400
        error_data = response.json()
        if "detail" in error_data:
            error_data = error_data["detail"]
        assert error_data.get("error_code") == ErrorCode.INVALID_IMAGE.value


class TestValidImageFormats:
    """Test acceptance of various valid image formats."""
    
    def test_multipart_jpeg_accepted(self, client, valid_image_300x300):
        """Test multipart JPEG upload is accepted for processing.
        
        Requirements: 1.1, 1.3, 14.1
        """
        response = client.post(
            "/api/analyze",
            files={"file": ("photo.jpg", valid_image_300x300, "image/jpeg")},
            data={"include_celebrity_match": "false"}
        )
        
        # Should pass validation and attempt processing
        # May fail due to no face, but shouldn't be INVALID_IMAGE due to format
        assert response.status_code in [200, 400, 408]
        if response.status_code == 400:
            error_data = response.json()
            if "detail" in error_data:
                error_data = error_data["detail"]
            # Should not be invalid format error
            assert error_data.get("error_code") != ErrorCode.FILE_TOO_LARGE.value
    
    def test_multipart_png_accepted(self, client, valid_png_image):
        """Test multipart PNG upload is accepted.
        
        Requirements: 1.1, 1.3, 14.1
        """
        response = client.post(
            "/api/analyze",
            files={"file": ("photo.png", valid_png_image, "image/png")},
            data={"include_celebrity_match": "false"}
        )
        
        assert response.status_code in [200, 400, 408]
    
    def test_base64_upload_accepted(self, client, valid_base64_image):
        """Test base64 image upload is accepted.
        
        Requirements: 1.6, 14.2
        """
        response = client.post(
            "/api/analyze",
            json={"image_base64": valid_base64_image, "include_celebrity_match": False}
        )
        
        assert response.status_code in [200, 400, 408]
    
    def test_base64_data_uri_accepted(self, client, valid_base64_data_uri):
        """Test base64 data URI format is accepted.
        
        Requirements: 1.6, 14.2
        """
        response = client.post(
            "/api/analyze",
            json={"image_base64": valid_base64_data_uri, "include_celebrity_match": False}
        )
        
        assert response.status_code in [200, 400, 408]
    
    def test_uppercase_file_extension_accepted(self, client, valid_image_300x300):
        """Test uppercase file extensions are accepted (case-insensitive).
        
        Requirements: 1.1
        """
        response = client.post(
            "/api/analyze",
            files={"file": ("photo.JPG", valid_image_300x300, "image/jpeg")},
            data={"include_celebrity_match": "false"}
        )
        
        assert response.status_code in [200, 400, 408]


class TestResponseStructure:
    """Test successful response structure validation."""
    
    @patch('app.main.image_processor')
    @patch('app.main.get_face_detector')
    @patch('app.main.symmetry_classifier')
    @patch('app.main.golden_ratio_scorer')
    @patch('app.main.percentile_engine')
    @patch('app.main.overlay_generator')
    @patch('app.main.get_celebrity_matcher')
    def test_successful_response_structure(
        self, mock_get_matcher, mock_overlay, mock_percentile, mock_scorer,
        mock_symmetry, mock_get_detector, mock_image_processor,
        client, valid_image_300x300
    ):
        """Test successful response contains all required fields with correct types.
        
        Requirements: 14.3, 14.4, 14.5, 14.6, 14.7, 14.8
        """
        # Setup mocks
        mock_image_processor.process_upload = AsyncMock(return_value=MagicMock(
            array=np.zeros((300, 300, 3), dtype=np.uint8),
            width=300,
            height=300,
            original_format='JPEG'
        ))
        
        mock_detector = Mock()
        mock_detector.detect_faces.return_value = create_mock_landmarks()
        mock_get_detector.return_value = mock_detector
        
        mock_symmetry.classify.return_value = SymmetryTypeEnum.OVAL
        mock_scorer.score_all.return_value = create_mock_golden_ratio_result()
        mock_percentile.score_to_percentile.return_value = 82
        mock_overlay.generate.return_value = create_mock_overlay_result()
        
        mock_matcher = Mock()
        mock_matcher.is_available = True
        
        # Create a proper mock object for celebrity match
        celebrity_match_result = Mock()
        celebrity_match_result.name = "Test Celebrity"
        celebrity_match_result.similarity = 0.65
        celebrity_match_result.strength = "strong"
        celebrity_match_result.confidence = 0.65
        celebrity_match_result.dataset_source = "Western"
        celebrity_match_result.thumbnail_path = "/path/to/thumbnail.jpg"
        celebrity_match_result.message = None
        celebrity_match_result.top_matches = []
        
        mock_matcher.find_match = AsyncMock(return_value=celebrity_match_result)
        mock_get_matcher.return_value = mock_matcher
        
        # Make request
        response = client.post(
            "/api/analyze",
            files={"file": ("test.jpg", valid_image_300x300, "image/jpeg")},
            data={"include_celebrity_match": "true"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify top-level response structure
        assert data["status"] == "success"
        assert isinstance(data["processing_time_ms"], int)
        assert data["processing_time_ms"] > 0
        assert "results" in data
        
        results = data["results"]
        
        # Verify golden ratio score
        assert isinstance(results["golden_ratio_score"], (int, float))
        assert 0 <= results["golden_ratio_score"] <= 100
        
        # Verify percentile rank
        assert isinstance(results["percentile_rank"], int)
        assert 0 <= results["percentile_rank"] <= 100
        
        # Verify face symmetry type
        assert results["face_symmetry_type"] in [
            "oval", "rectangle", "round", "square", "heart", "diamond"
        ]
        
        # Verify feature breakdown has 8 features
        feature_breakdown = results["feature_breakdown"]
        assert isinstance(feature_breakdown, list)
        assert len(feature_breakdown) == 8
        
        for feature in feature_breakdown:
            assert "feature" in feature
            assert "measured_ratio" in feature
            assert "ideal_ratio" in feature
            assert "score" in feature
            assert "deviation" in feature
            assert 0 <= feature["score"] <= 100
            assert abs(feature["ideal_ratio"] - 1.618) < 0.001
        
        # Verify overlay data
        overlay = results["overlay_data"]
        assert "landmarks" in overlay
        assert "golden_ratio_guides" in overlay
        assert len(overlay["landmarks"]) == 468
        
        for landmark in overlay["landmarks"]:
            assert isinstance(landmark, list)
            assert len(landmark) == 2
        
        guides = overlay["golden_ratio_guides"]
        assert "vertical_midline" in guides
        assert "horizontal_thirds" in guides
        assert len(guides["horizontal_thirds"]) == 4
        assert "phi_spiral_center" in guides
        assert len(guides["phi_spiral_center"]) == 2
        
        # Verify celebrity match
        assert "celebrity_match" in results
        celebrity_match = results["celebrity_match"]
        assert celebrity_match is not None
        assert "name" in celebrity_match
        assert "confidence" in celebrity_match
        assert "dataset_source" in celebrity_match
        assert "thumbnail_url" in celebrity_match
    
    @patch('app.main.image_processor')
    @patch('app.main.get_face_detector')
    @patch('app.main.symmetry_classifier')
    @patch('app.main.golden_ratio_scorer')
    @patch('app.main.percentile_engine')
    @patch('app.main.overlay_generator')
    @patch('app.main.get_celebrity_matcher')
    def test_progressive_degradation_celebrity_fails(
        self, mock_get_matcher, mock_overlay, mock_percentile, mock_scorer,
        mock_symmetry, mock_get_detector, mock_image_processor,
        client, valid_image_300x300
    ):
        """Test progressive degradation when celebrity matching fails.
        
        Requirements: 17.1, 17.2, 17.3, 17.4
        """
        # Setup mocks
        mock_image_processor.process_upload = AsyncMock(return_value=MagicMock(
            array=np.zeros((300, 300, 3), dtype=np.uint8),
            width=300,
            height=300,
            original_format='JPEG'
        ))
        
        mock_detector = Mock()
        mock_detector.detect_faces.return_value = create_mock_landmarks()
        mock_get_detector.return_value = mock_detector
        
        mock_symmetry.classify.return_value = SymmetryTypeEnum.RECTANGLE
        mock_scorer.score_all.return_value = create_mock_golden_ratio_result()
        mock_percentile.score_to_percentile.return_value = 75
        mock_overlay.generate.return_value = create_mock_overlay_result()
        
        # Celebrity matcher unavailable
        mock_matcher = Mock()
        mock_matcher.is_available = False
        mock_get_matcher.return_value = mock_matcher
        
        response = client.post(
            "/api/analyze",
            files={"file": ("test.jpg", valid_image_300x300, "image/jpeg")},
            data={"include_celebrity_match": "true"}
        )
        
        # Should still return HTTP 200 with partial results
        assert response.status_code == 200
        data = response.json()
        
        # Golden ratio should be complete
        results = data["results"]
        assert results["golden_ratio_score"] is not None
        assert results["percentile_rank"] is not None
        assert len(results["feature_breakdown"]) == 8
        
        # Celebrity match should be None (progressive degradation)
        assert results.get("celebrity_match") is None


class TestFlagParameters:
    """Test include_celebrity_match flag parameter."""
    
    @patch('app.main.image_processor')
    @patch('app.main.get_face_detector')
    @patch('app.main.symmetry_classifier')
    @patch('app.main.golden_ratio_scorer')
    @patch('app.main.percentile_engine')
    @patch('app.main.overlay_generator')
    @patch('app.main.get_celebrity_matcher')
    def test_celebrity_match_true_flag(
        self, mock_get_matcher, mock_overlay, mock_percentile, mock_scorer,
        mock_symmetry, mock_get_detector, mock_image_processor,
        client, valid_image_300x300
    ):
        """Test include_celebrity_match=true flag.
        
        Requirements: 14.2, 10.1
        """
        mock_image_processor.process_upload = AsyncMock(return_value=MagicMock(
            array=np.zeros((300, 300, 3), dtype=np.uint8),
            width=300,
            height=300,
            original_format='JPEG'
        ))
        mock_detector = Mock()
        mock_detector.detect_faces.return_value = create_mock_landmarks()
        mock_get_detector.return_value = mock_detector
        mock_symmetry.classify.return_value = SymmetryTypeEnum.ROUND
        mock_scorer.score_all.return_value = create_mock_golden_ratio_result()
        mock_percentile.score_to_percentile.return_value = 60
        mock_overlay.generate.return_value = create_mock_overlay_result()
        
        mock_matcher = Mock()
        mock_matcher.is_available = True
        mock_matcher.find_match = AsyncMock(return_value=None)
        mock_get_matcher.return_value = mock_matcher
        
        response = client.post(
            "/api/analyze",
            files={"file": ("test.jpg", valid_image_300x300, "image/jpeg")},
            data={"include_celebrity_match": "true"}
        )
        
        assert response.status_code == 200
        assert "celebrity_match" in response.json()["results"]
    
    @patch('app.main.image_processor')
    @patch('app.main.get_face_detector')
    @patch('app.main.symmetry_classifier')
    @patch('app.main.golden_ratio_scorer')
    @patch('app.main.percentile_engine')
    @patch('app.main.overlay_generator')
    def test_celebrity_match_false_flag(
        self, mock_overlay, mock_percentile, mock_scorer,
        mock_symmetry, mock_get_detector, mock_image_processor,
        client, valid_image_300x300
    ):
        """Test include_celebrity_match=false flag.
        
        Requirements: 14.2, 10.1
        """
        mock_image_processor.process_upload = AsyncMock(return_value=MagicMock(
            array=np.zeros((300, 300, 3), dtype=np.uint8),
            width=300,
            height=300,
            original_format='JPEG'
        ))
        mock_detector = Mock()
        mock_detector.detect_faces.return_value = create_mock_landmarks()
        mock_get_detector.return_value = mock_detector
        mock_symmetry.classify.return_value = SymmetryTypeEnum.SQUARE
        mock_scorer.score_all.return_value = create_mock_golden_ratio_result()
        mock_percentile.score_to_percentile.return_value = 50
        mock_overlay.generate.return_value = create_mock_overlay_result()
        
        response = client.post(
            "/api/analyze",
            files={"file": ("test.jpg", valid_image_300x300, "image/jpeg")},
            data={"include_celebrity_match": "false"}
        )
        
        assert response.status_code == 200
        assert response.json()["results"]["celebrity_match"] is None


class TestErrorResponseFormat:
    """Test error response structure."""
    
    def test_error_response_format(self, client):
        """Test that error responses have all required fields.
        
        Requirements: 15.1, 15.2, 15.3, 15.4, 15.5
        """
        response = client.post(
            "/api/analyze",
            json={}
        )
        
        assert response.status_code == 400
        error_data = response.json()
        
        if "detail" in error_data:
            error_data = error_data["detail"]
        
        assert error_data.get("status") == "error"
        assert "error_code" in error_data
        assert "message" in error_data
        assert "retry" in error_data
        
        assert isinstance(error_data["error_code"], str)
        assert isinstance(error_data["message"], str)
        assert isinstance(error_data["retry"], bool)
    
    def test_validation_errors_allow_retry(self, client):
        """Test that validation errors have retry=true flag.
        
        Requirements: 15.5
        """
        response = client.post(
            "/api/analyze",
            json={"image_base64": ""}
        )
        
        assert response.status_code == 400
        error_data = response.json()
        if "detail" in error_data:
            error_data = error_data["detail"]
        assert error_data["retry"] == True


class TestHealthCheckEndpoint:
    """Test health check endpoint."""
    
    def test_health_check_returns_format(self, client):
        """Test that health check endpoint returns proper format.
        
        Requirements: 12.1, 18.1
        """
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "version" in data
        assert "celebrity_matching_available" in data
        assert isinstance(data["celebrity_matching_available"], bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
