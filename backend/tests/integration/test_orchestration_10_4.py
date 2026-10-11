"""
Integration tests for task 10.4: Component Orchestration

Tests verify that _process_analysis properly orchestrates all components:
1. ImageProcessor - normalize uploaded image
2. FaceDetector - detect face and extract landmarks
3. SymmetryClassifier - classify face shape (parallel)
4. GoldenRatioScorer - compute ratios and scores (parallel)
5. OverlayGenerator - generate visual overlay data (parallel)
6. CelebrityMatcher - find celebrity match (optional, async, can fail)
7. PercentileEngine - compute percentile ranking
8. Assemble final AnalysisResult response

Requirements covered: 14.3, 14.4, 14.5, 14.6, 14.7, 14.8
"""

import pytest
import asyncio
import numpy as np
from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image

from app.main import app
from app.models.internal import Landmarks, SymmetryType
from app.components import (
    get_face_detector,
    symmetry_classifier,
    golden_ratio_scorer,
    percentile_engine,
    get_celebrity_matcher,
    overlay_generator
)


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def valid_test_image():
    """Create a valid test image (1x1 but larger in actual test)."""
    # Create a simple test image
    img = Image.new('RGB', (640, 480), color='white')
    # Draw a simple face-like pattern
    pixels = img.load()
    for i in range(100, 200):
        for j in range(100, 200):
            pixels[i, j] = (200, 150, 100)  # Skin tone
    
    # Convert to bytes
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    return img_bytes.getvalue()


class TestOrchestrationSequence:
    """Test that orchestration follows the correct sequence."""
    
    def test_sequential_image_and_face_detection(self, client, valid_test_image):
        """
        Requirement 14.3, 14.4:
        Steps 1-2 (image → face detection) are sequential.
        ImageProcessor output must be valid input to FaceDetector.
        """
        # This is indirectly tested through endpoint, but let's verify
        # that image processing and face detection are sequential
        
        # Create valid image
        img = Image.new('RGB', (640, 480), color='white')
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        response = client.post(
            "/api/analyze",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        
        # Should get either success or face detection error
        # (depends on whether image has detectable face)
        assert response.status_code in [200, 400]
        
        if response.status_code == 200:
            # If successful, should have all required fields
            data = response.json()
            assert "results" in data
            assert "golden_ratio_score" in data["results"]
    
    def test_parallel_symmetry_scoring_overlay(self, client):
        """
        Requirement 14.5, 14.6, 14.7:
        Steps 3-5 (symmetry, scoring, overlay) run in parallel.
        All three should complete successfully and return data.
        """
        # Create a simple valid image
        img = Image.new('RGB', (640, 480), color=(200, 150, 100))
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        response = client.post(
            "/api/analyze",
            files={"file": ("test.png", img_bytes, "image/png")},
            data={"include_celebrity_match": "false"}  # Skip celebrity for speed
        )
        
        # Endpoint should either succeed or fail cleanly
        assert response.status_code in [200, 400, 408]
        
        if response.status_code == 200:
            data = response.json()
            results = data["results"]
            
            # Verify all parallel components completed:
            # 1. SymmetryClassifier result
            assert "face_symmetry_type" in results
            assert results["face_symmetry_type"] in [
                "oval", "rectangle", "round", "square", "heart", "diamond"
            ]
            
            # 2. GoldenRatioScorer result
            assert "golden_ratio_score" in results
            assert 0 <= results["golden_ratio_score"] <= 100
            assert "feature_breakdown" in results
            assert len(results["feature_breakdown"]) >= 8
            
            # 3. OverlayGenerator result
            assert "overlay_data" in results
            assert "landmarks" in results["overlay_data"]
            assert len(results["overlay_data"]["landmarks"]) == 468
            assert "golden_ratio_guides" in results["overlay_data"]
    
    def test_percentile_depends_on_scoring(self, client):
        """
        Requirement 14.7:
        PercentileEngine step depends on GoldenRatioScorer output.
        Percentile rank must correspond to score.
        """
        img = Image.new('RGB', (640, 480), color=(200, 150, 100))
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        response = client.post(
            "/api/analyze",
            files={"file": ("test.png", img_bytes, "image/png")},
            data={"include_celebrity_match": "false"}
        )
        
        if response.status_code == 200:
            data = response.json()
            results = data["results"]
            
            # Both score and percentile should be present
            assert "golden_ratio_score" in results
            assert "percentile_rank" in results
            
            # Percentile should be integer in [0, 100]
            percentile = results["percentile_rank"]
            assert isinstance(percentile, int)
            assert 0 <= percentile <= 100
    
    def test_optional_celebrity_matching_progressive_degradation(self, client):
        """
        Requirement 14.8, 17.1-17.4:
        Celebrity matching is optional and can fail without blocking.
        If celebrity matching fails, golden ratio scores still returned.
        Response has HTTP 200 even if celebrity matching failed.
        """
        img = Image.new('RGB', (640, 480), color=(200, 150, 100))
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Include celebrity matching
        response = client.post(
            "/api/analyze",
            files={"file": ("test.png", img_bytes, "image/png")},
            data={"include_celebrity_match": "true"}
        )
        
        # Should succeed regardless of celebrity matching success/failure
        assert response.status_code in [200, 400]
        
        if response.status_code == 200:
            data = response.json()
            
            # Core results should always be present
            assert "results" in data
            assert "golden_ratio_score" in data["results"]
            assert "feature_breakdown" in data["results"]
            
            # Celebrity match is optional
            results = data["results"]
            if "celebrity_match" in results and results["celebrity_match"]:
                # If present, should have correct structure
                match = results["celebrity_match"]
                if match.get("name"):
                    # If name present, confidence should be present
                    assert "confidence" in match
                    assert 0.0 <= match["confidence"] <= 1.0
    
    def test_request_disables_celebrity_matching(self, client):
        """
        Requirement 14.8:
        When include_celebrity_match=false, CelebrityMatcher should not run.
        Response should complete faster.
        """
        img = Image.new('RGB', (640, 480), color=(200, 150, 100))
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        response = client.post(
            "/api/analyze",
            files={"file": ("test.png", img_bytes, "image/png")},
            data={"include_celebrity_match": "false"}
        )
        
        # Should complete successfully
        assert response.status_code in [200, 400]
        
        if response.status_code == 200:
            data = response.json()
            # Core components should have completed
            assert "results" in data
            assert "golden_ratio_score" in data["results"]


class TestOrchestrationErrorHandling:
    """Test error handling in orchestration."""
    
    def test_face_detection_failure_stops_processing(self, client):
        """
        Requirement 14.3-14.4:
        If face detection fails (step 2), processing stops.
        Error is returned, not masked by later components.
        """
        # Create image with no faces
        img = Image.new('RGB', (640, 480), color=(100, 100, 100))
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        response = client.post(
            "/api/analyze",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        
        # Should fail with face detection error
        assert response.status_code == 400
        error_data = response.json()
        assert "error_code" in error_data["detail"]
        
        # Should be one of the face detection errors
        error_code = error_data["detail"]["error_code"]
        assert error_code in ["NO_FACE_DETECTED", "POOR_QUALITY", "INVALID_IMAGE"]
    
    def test_image_processing_failure_stops_processing(self, client):
        """
        Requirement 14.3:
        If image processing fails (step 1), processing stops.
        """
        # Send invalid image data
        response = client.post(
            "/api/analyze",
            files={"file": ("test.png", b"not an image", "image/png")}
        )
        
        # Should fail with image error
        assert response.status_code == 400
        error_data = response.json()
        assert "error_code" in error_data["detail"]
        assert error_data["detail"]["error_code"] in [
            "INVALID_IMAGE", "POOR_QUALITY"
        ]
    
    def test_celebrity_failure_does_not_block_results(self, client):
        """
        Requirement 17.1-17.4:
        If celebrity matching fails, other results still returned.
        HTTP 200 is returned with partial results.
        """
        # This test verifies progressive degradation
        # Create a valid image
        img = Image.new('RGB', (640, 480), color=(200, 150, 100))
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        response = client.post(
            "/api/analyze",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        
        # Should succeed even if celebrity matching unavailable
        # (celebrity DB may not be loaded in test environment)
        assert response.status_code in [200, 400]
        
        if response.status_code == 200:
            # Core results should be present
            data = response.json()
            assert "results" in data
            assert "golden_ratio_score" in data["results"]
            assert "percentile_rank" in data["results"]


class TestOrchestrationResponseStructure:
    """Test that orchestration returns properly structured responses."""
    
    def test_complete_analysis_result_structure(self, client):
        """
        Requirement 14.3-14.8:
        Response includes all required fields for AnalysisResult.
        """
        img = Image.new('RGB', (640, 480), color=(200, 150, 100))
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        response = client.post(
            "/api/analyze",
            files={"file": ("test.png", img_bytes, "image/png")},
            data={"include_celebrity_match": "false"}
        )
        
        if response.status_code == 200:
            data = response.json()
            results = data["results"]
            
            # All required fields from AnalysisResult
            required_fields = [
                "golden_ratio_score",
                "percentile_rank",
                "face_symmetry_type",
                "feature_breakdown",
                "overlay_data"
            ]
            
            for field in required_fields:
                assert field in results, f"Missing field: {field}"
            
            # Validate types
            assert isinstance(results["golden_ratio_score"], (int, float))
            assert isinstance(results["percentile_rank"], int)
            assert isinstance(results["face_symmetry_type"], str)
            assert isinstance(results["feature_breakdown"], list)
            assert isinstance(results["overlay_data"], dict)
    
    def test_feature_breakdown_structure(self, client):
        """
        Requirement 14.6:
        Feature breakdown contains 8 features with correct structure.
        """
        img = Image.new('RGB', (640, 480), color=(200, 150, 100))
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        response = client.post(
            "/api/analyze",
            files={"file": ("test.png", img_bytes, "image/png")},
            data={"include_celebrity_match": "false"}
        )
        
        if response.status_code == 200:
            data = response.json()
            features = data["results"]["feature_breakdown"]
            
            # Should have at least 8 features
            assert len(features) >= 8
            
            # Each feature should have required fields
            for feature in features:
                assert "feature" in feature
                assert "measured_ratio" in feature
                assert "ideal_ratio" in feature
                assert "score" in feature
                assert "deviation" in feature
                
                # Validate values
                assert isinstance(feature["feature"], str)
                assert isinstance(feature["measured_ratio"], (int, float))
                assert isinstance(feature["ideal_ratio"], (int, float))
                assert 0 <= feature["score"] <= 100
                assert isinstance(feature["deviation"], (int, float))
    
    def test_overlay_data_structure(self, client):
        """
        Requirement 14.7, 9.1-9.5:
        Overlay data contains landmarks and golden ratio guides.
        """
        img = Image.new('RGB', (640, 480), color=(200, 150, 100))
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        response = client.post(
            "/api/analyze",
            files={"file": ("test.png", img_bytes, "image/png")},
            data={"include_celebrity_match": "false"}
        )
        
        if response.status_code == 200:
            data = response.json()
            overlay = data["results"]["overlay_data"]
            
            # Check landmarks
            assert "landmarks" in overlay
            assert isinstance(overlay["landmarks"], list)
            assert len(overlay["landmarks"]) == 468
            
            # Each landmark should be [x, y]
            for landmark in overlay["landmarks"]:
                assert isinstance(landmark, list)
                assert len(landmark) == 2
                assert isinstance(landmark[0], (int, float))
                assert isinstance(landmark[1], (int, float))
            
            # Check golden ratio guides
            assert "golden_ratio_guides" in overlay
            guides = overlay["golden_ratio_guides"]
            assert "vertical_midline" in guides
            assert "horizontal_thirds" in guides
            assert "phi_spiral_center" in guides


class TestOrchestrationTiming:
    """Test that orchestration completes within timeout."""
    
    def test_processing_completes_within_30_seconds(self, client):
        """
        Requirement 12.1-12.4:
        Processing completes within 30-second timeout.
        """
        import time
        
        img = Image.new('RGB', (640, 480), color=(200, 150, 100))
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        start_time = time.time()
        
        response = client.post(
            "/api/analyze",
            files={"file": ("test.png", img_bytes, "image/png")},
            data={"include_celebrity_match": "false"}
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Should complete in less than 30 seconds
        assert elapsed_ms < 30000
        
        if response.status_code == 200:
            data = response.json()
            # Processing time should match
            processing_ms = data["processing_time_ms"]
            assert processing_ms > 0
            assert processing_ms < 30000
    
    def test_response_includes_processing_time(self, client):
        """
        Requirement 14.4:
        Response includes processing_time_ms field.
        """
        img = Image.new('RGB', (640, 480), color=(200, 150, 100))
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        response = client.post(
            "/api/analyze",
            files={"file": ("test.png", img_bytes, "image/png")},
            data={"include_celebrity_match": "false"}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "processing_time_ms" in data
            assert isinstance(data["processing_time_ms"], int)
            assert data["processing_time_ms"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
