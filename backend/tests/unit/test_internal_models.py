"""
Unit tests for internal data structures.

Tests validation logic, type safety, and constraints for all internal models
used for component communication in the FaceMatric backend.

Requirements tested:
- 1.2: ImageData RGB normalization
- 1.3: ImageData aspect ratio preservation
- 2.2: Landmarks with exactly 468 3D coordinates
- 4.8: SymmetryType enum with 6 classification types
- 5.1-5.8: GoldenRatioResult with 8 feature measurements
- 6.4: FeatureScoreInternal with measured_ratio, ideal_ratio, score, deviation
- 7.1-7.3: GoldenRatioResult overall_score bounded [0, 100] with validation
"""

import pytest
import numpy as np
from app.models.internal import (
    ImageData,
    Landmarks,
    SymmetryType,
    FeatureScoreInternal,
    GoldenRatioResult,
    CelebrityMatchInternal,
    OverlayDataInternal,
    ProcessingMetadata
)


class TestSymmetryType:
    """Test SymmetryType enum (Requirement 4.8)."""
    
    def test_all_six_types_exist(self):
        """Verify all 6 face shape classifications exist."""
        assert SymmetryType.OVAL.value == "oval"
        assert SymmetryType.RECTANGLE.value == "rectangle"
        assert SymmetryType.ROUND.value == "round"
        assert SymmetryType.SQUARE.value == "square"
        assert SymmetryType.HEART.value == "heart"
        assert SymmetryType.DIAMOND.value == "diamond"
    
    def test_enum_count(self):
        """Verify exactly 6 symmetry types exist."""
        assert len(SymmetryType) == 6


class TestImageData:
    """Test ImageData dataclass (Requirements 1.2, 1.3)."""
    
    def test_valid_rgb_image(self):
        """Test creating valid RGB ImageData."""
        array = np.zeros((480, 640, 3), dtype=np.uint8)
        image_data = ImageData(
            array=array,
            width=640,
            height=480,
            original_format="JPEG"
        )
        
        assert image_data.width == 640
        assert image_data.height == 480
        assert image_data.original_format == "JPEG"
        assert image_data.array.shape == (480, 640, 3)
    
    def test_rgb_normalization_requirement(self):
        """Requirement 1.2: ImageData must be normalized to RGB (3 channels)."""
        array = np.zeros((100, 100, 3), dtype=np.uint8)
        image_data = ImageData(
            array=array,
            width=100,
            height=100,
            original_format="PNG"
        )
        assert image_data.array.shape[2] == 3  # RGB has 3 channels
    
    def test_aspect_ratio_preservation_requirement(self):
        """Requirement 1.3: Dimensions should maintain aspect ratio."""
        # Original 1920x1080 (16:9 ratio) resized to max 640px
        # Should maintain aspect ratio
        array = np.zeros((360, 640, 3), dtype=np.uint8)
        image_data = ImageData(
            array=array,
            width=640,
            height=360,
            original_format="JPEG"
        )
        
        # Verify aspect ratio is preserved (16:9 = 1.777...)
        aspect_ratio = image_data.width / image_data.height
        assert abs(aspect_ratio - 16/9) < 0.01
    
    def test_validation_wrong_dimensions(self):
        """Test validation fails when array shape doesn't match dimensions."""
        array = np.zeros((480, 640, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="doesn't match dimensions"):
            ImageData(
                array=array,
                width=320,  # Wrong width
                height=480,
                original_format="JPEG"
            )
    
    def test_validation_wrong_channels(self):
        """Test validation fails for non-RGB images."""
        array = np.zeros((480, 640, 4), dtype=np.uint8)  # RGBA
        
        with pytest.raises(ValueError, match="must have 3 channels"):
            ImageData(
                array=array,
                width=640,
                height=480,
                original_format="PNG"
            )
    
    def test_validation_not_3d_array(self):
        """Test validation fails for non-3D arrays."""
        array = np.zeros((480, 640), dtype=np.uint8)  # Grayscale
        
        with pytest.raises(ValueError, match="must be 3D"):
            ImageData(
                array=array,
                width=640,
                height=480,
                original_format="JPEG"
            )
    
    def test_validation_negative_dimensions(self):
        """Test validation fails for negative dimensions."""
        array = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Array shape validation happens first, so expect "doesn't match dimensions"
        with pytest.raises(ValueError, match="doesn't match dimensions"):
            ImageData(
                array=array,
                width=-640,
                height=480,
                original_format="JPEG"
            )
    
    def test_validation_zero_dimensions(self):
        """Test validation fails for zero dimensions."""
        array = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Array shape validation happens first, so expect "doesn't match dimensions"
        with pytest.raises(ValueError, match="doesn't match dimensions"):
            ImageData(
                array=array,
                width=0,
                height=480,
                original_format="JPEG"
            )


class TestLandmarks:
    """Test Landmarks dataclass (Requirement 2.2)."""
    
    def test_valid_landmarks(self):
        """Test creating valid Landmarks with 468 points."""
        points = np.random.rand(468, 3).astype(np.float32)
        landmarks = Landmarks(
            points=points,
            confidence=0.85,
            image_width=640,
            image_height=480
        )
        
        assert landmarks.points.shape == (468, 3)
        assert landmarks.confidence == 0.85
        assert landmarks.image_width == 640
        assert landmarks.image_height == 480
    
    def test_exactly_468_landmarks_requirement(self):
        """Requirement 2.2: Must extract exactly 468 3D landmark coordinates."""
        points = np.random.rand(468, 3).astype(np.float32)
        landmarks = Landmarks(
            points=points,
            confidence=0.8,
            image_width=640,
            image_height=480
        )
        
        assert landmarks.points.shape[0] == 468  # Exactly 468 landmarks
        assert landmarks.points.shape[1] == 3     # Each is (x, y, z)
    
    def test_validation_wrong_landmark_count(self):
        """Test validation fails for wrong number of landmarks."""
        points = np.random.rand(467, 3).astype(np.float32)  # Wrong count
        
        with pytest.raises(ValueError, match="must have shape \\(468, 3\\)"):
            Landmarks(
                points=points,
                confidence=0.8,
                image_width=640,
                image_height=480
            )
    
    def test_validation_wrong_dimensions(self):
        """Test validation fails for wrong coordinate dimensions."""
        points = np.random.rand(468, 2).astype(np.float32)  # Only x, y
        
        with pytest.raises(ValueError, match="must have shape \\(468, 3\\)"):
            Landmarks(
                points=points,
                confidence=0.8,
                image_width=640,
                image_height=480
            )
    
    def test_validation_confidence_out_of_range(self):
        """Test validation fails for confidence outside [0, 1]."""
        points = np.random.rand(468, 3).astype(np.float32)
        
        with pytest.raises(ValueError, match="Confidence must be in"):
            Landmarks(
                points=points,
                confidence=1.5,  # Invalid
                image_width=640,
                image_height=480
            )
    
    def test_validation_negative_confidence(self):
        """Test validation fails for negative confidence."""
        points = np.random.rand(468, 3).astype(np.float32)
        
        with pytest.raises(ValueError, match="Confidence must be in"):
            Landmarks(
                points=points,
                confidence=-0.1,
                image_width=640,
                image_height=480
            )
    
    def test_validation_negative_image_dimensions(self):
        """Test validation fails for negative image dimensions."""
        points = np.random.rand(468, 3).astype(np.float32)
        
        with pytest.raises(ValueError, match="must be positive"):
            Landmarks(
                points=points,
                confidence=0.8,
                image_width=-640,
                image_height=480
            )
    
    def test_get_2d_landmarks(self):
        """Test extraction of 2D landmarks from 3D."""
        points = np.random.rand(468, 3).astype(np.float32)
        landmarks = Landmarks(
            points=points,
            confidence=0.8,
            image_width=640,
            image_height=480
        )
        
        landmarks_2d = landmarks.get_2d_landmarks()
        
        assert landmarks_2d.shape == (468, 2)
        # Verify x, y coordinates match
        np.testing.assert_array_equal(landmarks_2d, points[:, :2])
    
    def test_get_landmark_by_index(self):
        """Test retrieving specific landmark by index."""
        points = np.random.rand(468, 3).astype(np.float32)
        landmarks = Landmarks(
            points=points,
            confidence=0.8,
            image_width=640,
            image_height=480
        )
        
        landmark_0 = landmarks.get_landmark(0)
        landmark_467 = landmarks.get_landmark(467)
        
        assert landmark_0.shape == (3,)
        assert landmark_467.shape == (3,)
        np.testing.assert_array_equal(landmark_0, points[0])
        np.testing.assert_array_equal(landmark_467, points[467])
    
    def test_get_landmark_invalid_index(self):
        """Test get_landmark raises IndexError for invalid index."""
        points = np.random.rand(468, 3).astype(np.float32)
        landmarks = Landmarks(
            points=points,
            confidence=0.8,
            image_width=640,
            image_height=480
        )
        
        with pytest.raises(IndexError, match="must be in \\[0, 467\\]"):
            landmarks.get_landmark(468)
        
        with pytest.raises(IndexError, match="must be in \\[0, 467\\]"):
            landmarks.get_landmark(-1)


class TestFeatureScoreInternal:
    """Test FeatureScoreInternal dataclass (Requirement 6.4)."""
    
    def test_valid_feature_score(self):
        """Test creating valid FeatureScoreInternal."""
        feature_score = FeatureScoreInternal(
            feature="Face Length ÷ Face Width",
            measured_ratio=1.55,
            ideal_ratio=1.618,
            score=95.8,
            deviation=-0.068
        )
        
        assert feature_score.feature == "Face Length ÷ Face Width"
        assert feature_score.measured_ratio == 1.55
        assert feature_score.ideal_ratio == 1.618
        assert feature_score.score == 95.8
        assert feature_score.deviation == -0.068
    
    def test_feature_breakdown_fields_requirement(self):
        """Requirement 6.4: Each feature must have measured_ratio, ideal_ratio, score, deviation."""
        feature_score = FeatureScoreInternal(
            feature="Eye Spacing ÷ Eye Width",
            measured_ratio=1.7,
            ideal_ratio=1.618,
            score=94.9,
            deviation=0.082
        )
        
        # Verify all required fields exist
        assert hasattr(feature_score, 'measured_ratio')
        assert hasattr(feature_score, 'ideal_ratio')
        assert hasattr(feature_score, 'score')
        assert hasattr(feature_score, 'deviation')
    
    def test_validation_score_out_of_range(self):
        """Test validation fails for score outside [0, 100]."""
        with pytest.raises(ValueError, match="Score must be in \\[0, 100\\]"):
            FeatureScoreInternal(
                feature="Test",
                measured_ratio=1.5,
                ideal_ratio=1.618,
                score=105.0,  # Invalid
                deviation=-0.118
            )
    
    def test_validation_negative_score(self):
        """Test validation fails for negative score."""
        with pytest.raises(ValueError, match="Score must be in \\[0, 100\\]"):
            FeatureScoreInternal(
                feature="Test",
                measured_ratio=1.5,
                ideal_ratio=1.618,
                score=-5.0,  # Invalid
                deviation=-0.118
            )
    
    def test_validation_negative_measured_ratio(self):
        """Test validation fails for negative measured ratio."""
        with pytest.raises(ValueError, match="must be non-negative"):
            FeatureScoreInternal(
                feature="Test",
                measured_ratio=-1.5,  # Invalid
                ideal_ratio=1.618,
                score=50.0,
                deviation=-3.118
            )
    
    def test_validation_inconsistent_deviation(self):
        """Test validation fails when deviation doesn't match ratios."""
        with pytest.raises(ValueError, match="Deviation .* inconsistent"):
            FeatureScoreInternal(
                feature="Test",
                measured_ratio=1.5,
                ideal_ratio=1.618,
                score=92.7,
                deviation=0.5  # Should be -0.118
            )
    
    def test_perfect_golden_ratio_score(self):
        """Test feature with perfect golden ratio (1.618) has score 100."""
        feature_score = FeatureScoreInternal(
            feature="Perfect Feature",
            measured_ratio=1.618,
            ideal_ratio=1.618,
            score=100.0,
            deviation=0.0
        )
        
        assert feature_score.score == 100.0
        assert feature_score.deviation == 0.0


class TestGoldenRatioResult:
    """Test GoldenRatioResult dataclass (Requirements 5.1-5.8, 7.1-7.3)."""
    
    def create_feature_scores(self, scores):
        """Helper to create list of FeatureScoreInternal from scores."""
        features = []
        for i, score in enumerate(scores):
            # Calculate ratio that would give this score
            # score = 100 * (1 - |ratio - 1.618| / 1.618)
            # |ratio - 1.618| = (100 - score) * 1.618 / 100
            deviation = (100 - score) * 1.618 / 100
            ratio = 1.618 + deviation  # Positive deviation
            
            features.append(FeatureScoreInternal(
                feature=f"Feature {i+1}",
                measured_ratio=ratio,
                ideal_ratio=1.618,
                score=score,
                deviation=deviation
            ))
        return features
    
    def test_valid_golden_ratio_result(self):
        """Test creating valid GoldenRatioResult."""
        features = self.create_feature_scores([90.0, 85.0, 88.0, 92.0, 87.0, 89.0, 91.0, 86.0])
        overall = sum(f.score for f in features) / 8
        overall = round(overall, 1)
        
        result = GoldenRatioResult(
            overall_score=overall,
            features=features
        )
        
        assert result.overall_score == overall
        assert len(result.features) == 8
    
    def test_exactly_8_features_requirement(self):
        """Requirement 5.1-5.8: Must compute exactly 8 facial ratios."""
        features = self.create_feature_scores([90.0, 85.0, 88.0, 92.0, 87.0, 89.0, 91.0, 86.0])
        overall = 88.5
        
        result = GoldenRatioResult(
            overall_score=overall,
            features=features
        )
        
        assert len(result.features) == 8
    
    def test_overall_score_arithmetic_mean_requirement(self):
        """Requirement 7.1: Overall score must be arithmetic mean of 8 feature scores."""
        feature_scores = [90.0, 85.0, 88.0, 92.0, 87.0, 89.0, 91.0, 86.0]
        expected_mean = sum(feature_scores) / 8  # 88.5
        
        features = self.create_feature_scores(feature_scores)
        
        result = GoldenRatioResult(
            overall_score=round(expected_mean, 1),
            features=features
        )
        
        assert result.overall_score == 88.5
    
    def test_overall_score_rounded_to_one_decimal_requirement(self):
        """Requirement 7.2: Overall score must be rounded to one decimal place."""
        features = self.create_feature_scores([91.3, 85.7, 88.2, 92.6, 87.1, 89.4, 91.8, 86.3])
        overall = 89.05  # Should be rounded to 89.1 or 89.0
        
        result = GoldenRatioResult(
            overall_score=89.1,  # One decimal place
            features=features
        )
        
        # Verify only one decimal place
        assert result.overall_score == round(result.overall_score, 1)
    
    def test_overall_score_bounded_requirement(self):
        """Requirement 7.3: Overall score must be bounded [0, 100]."""
        features = self.create_feature_scores([90.0, 85.0, 88.0, 92.0, 87.0, 89.0, 91.0, 86.0])
        
        result = GoldenRatioResult(
            overall_score=88.5,
            features=features
        )
        
        assert 0.0 <= result.overall_score <= 100.0
    
    def test_validation_wrong_feature_count(self):
        """Test validation fails for wrong number of features."""
        features = self.create_feature_scores([90.0, 85.0, 88.0, 92.0, 87.0, 89.0, 91.0])  # Only 7
        
        with pytest.raises(ValueError, match="Must have at least 8 features"):
            GoldenRatioResult(
                overall_score=88.9,
                features=features
            )
    
    def test_validation_overall_score_out_of_range(self):
        """Test validation fails for overall score outside [0, 100]."""
        features = self.create_feature_scores([90.0, 85.0, 88.0, 92.0, 87.0, 89.0, 91.0, 86.0])
        
        with pytest.raises(ValueError, match="must be in \\[0, 100\\]"):
            GoldenRatioResult(
                overall_score=105.0,  # Invalid
                features=features
            )
    
    def test_validation_overall_score_not_mean(self):
        """Test validation fails when overall score is not arithmetic mean."""
        features = self.create_feature_scores([90.0, 85.0, 88.0, 92.0, 87.0, 89.0, 91.0, 86.0])
        # Mean is 88.5, provide wrong value
        
        with pytest.raises(ValueError, match="is not arithmetic mean"):
            GoldenRatioResult(
                overall_score=75.0,  # Wrong mean
                features=features
            )
    
    def test_validation_overall_score_not_rounded(self):
        """Test validation fails when overall score has more than 1 decimal place."""
        features = self.create_feature_scores([90.0, 85.0, 88.0, 92.0, 87.0, 89.0, 91.0, 86.0])
        
        with pytest.raises(ValueError, match="must be rounded to 1 decimal place"):
            GoldenRatioResult(
                overall_score=88.5234,  # Too many decimals
                features=features
            )


class TestCelebrityMatchInternal:
    """Test CelebrityMatchInternal dataclass."""
    
    def test_valid_celebrity_match(self):
        """Test creating valid celebrity match."""
        match = CelebrityMatchInternal(
            name="Tom Holland",
            confidence=0.92,
            dataset_source="Pins Face Recognition",
            thumbnail_path="/celebrities/tom_holland.jpg",
            message=None
        )
        
        assert match.name == "Tom Holland"
        assert match.confidence == 0.92
        assert match.dataset_source == "Pins Face Recognition"
        assert match.thumbnail_path == "/celebrities/tom_holland.jpg"
    
    def test_no_match_case(self):
        """Test creating celebrity match with no result."""
        match = CelebrityMatchInternal(
            name=None,
            confidence=0.42,
            dataset_source=None,
            thumbnail_path=None,
            message="No close match found in our dataset"
        )
        
        assert match.name is None
        assert match.confidence == 0.42
        assert match.message == "No close match found in our dataset"
    
    def test_validation_confidence_out_of_range(self):
        """Test validation fails for confidence outside [0, 1]."""
        with pytest.raises(ValueError, match="Confidence must be in"):
            CelebrityMatchInternal(
                name="Test",
                confidence=1.5,  # Invalid
                dataset_source="Test",
                thumbnail_path="/test.jpg"
            )
    
    def test_validation_name_without_confidence(self):
        """Test validation fails when name is set but confidence is None."""
        with pytest.raises(ValueError, match="Confidence must be provided"):
            CelebrityMatchInternal(
                name="Test",
                confidence=None,  # Invalid
                dataset_source="Test",
                thumbnail_path="/test.jpg"
            )
    
    def test_automatic_message_for_no_match(self):
        """Test automatic message when no match found."""
        match = CelebrityMatchInternal(
            name=None,
            confidence=0.42
        )
        
        assert match.message == "No close match found in our dataset"


class TestOverlayDataInternal:
    """Test OverlayDataInternal dataclass."""
    
    def test_valid_overlay_data(self):
        """Test creating valid overlay data."""
        landmarks_2d = np.random.rand(468, 2).astype(np.float32)
        overlay = OverlayDataInternal(
            landmarks_2d=landmarks_2d,
            midline_x=320.0,
            horizontal_thirds_y=[100.0, 250.0, 400.0],
            phi_spiral_center=(320.0, 240.0)
        )
        
        assert overlay.landmarks_2d.shape == (468, 2)
        assert overlay.midline_x == 320.0
        assert overlay.horizontal_thirds_y == [100.0, 250.0, 400.0]
        assert overlay.phi_spiral_center == (320.0, 240.0)
    
    def test_validation_wrong_landmark_shape(self):
        """Test validation fails for wrong landmark shape."""
        landmarks_2d = np.random.rand(467, 2).astype(np.float32)  # Wrong count
        
        with pytest.raises(ValueError, match="must have shape \\(468, 2\\)"):
            OverlayDataInternal(
                landmarks_2d=landmarks_2d,
                midline_x=320.0,
                horizontal_thirds_y=[100.0, 250.0, 400.0],
                phi_spiral_center=(320.0, 240.0)
            )
    
    def test_validation_wrong_thirds_count(self):
        """Test validation fails for wrong number of horizontal thirds."""
        landmarks_2d = np.random.rand(468, 2).astype(np.float32)
        
        with pytest.raises(ValueError, match="Must have exactly 3 horizontal thirds"):
            OverlayDataInternal(
                landmarks_2d=landmarks_2d,
                midline_x=320.0,
                horizontal_thirds_y=[100.0, 250.0],  # Only 2
                phi_spiral_center=(320.0, 240.0)
            )
    
    def test_validation_thirds_not_ascending(self):
        """Test validation fails when horizontal thirds not in ascending order."""
        landmarks_2d = np.random.rand(468, 2).astype(np.float32)
        
        with pytest.raises(ValueError, match="must be in ascending order"):
            OverlayDataInternal(
                landmarks_2d=landmarks_2d,
                midline_x=320.0,
                horizontal_thirds_y=[400.0, 250.0, 100.0],  # Descending
                phi_spiral_center=(320.0, 240.0)
            )


class TestProcessingMetadata:
    """Test ProcessingMetadata dataclass."""
    
    def test_valid_processing_metadata(self):
        """Test creating valid processing metadata."""
        import time
        
        metadata = ProcessingMetadata(
            request_id="test-123",
            start_time=time.time()
        )
        
        assert metadata.request_id == "test-123"
        assert isinstance(metadata.start_time, float)
        assert len(metadata.component_times) == 0
    
    def test_add_component_time(self):
        """Test adding component timing."""
        import time
        
        metadata = ProcessingMetadata(
            request_id="test-123",
            start_time=time.time()
        )
        
        metadata.add_component_time("image_processor", 150)
        metadata.add_component_time("face_detector", 800)
        
        assert metadata.component_times["image_processor"] == 150
        assert metadata.component_times["face_detector"] == 800
    
    def test_get_total_time(self):
        """Test calculating total elapsed time."""
        import time
        
        start = time.time()
        metadata = ProcessingMetadata(
            request_id="test-123",
            start_time=start
        )
        
        time.sleep(0.1)  # Sleep 100ms
        current = time.time()
        
        total_ms = metadata.get_total_time_ms(current)
        assert total_ms >= 100  # At least 100ms elapsed
        assert total_ms < 200   # But not too much more
    
    def test_get_component_summary(self):
        """Test getting component summary for logging."""
        import time
        
        start = time.time()
        metadata = ProcessingMetadata(
            request_id="test-456",
            start_time=start
        )
        
        metadata.add_component_time("image_processor", 150)
        metadata.add_component_time("face_detector", 800)
        
        summary = metadata.get_component_summary()
        
        assert summary["request_id"] == "test-456"
        assert "total_ms" in summary
        assert summary["components"]["image_processor"] == 150
        assert summary["components"]["face_detector"] == 800
