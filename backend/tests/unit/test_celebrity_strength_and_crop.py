"""
Unit tests for celebrity match strength bands, null shape consistency, and shared face crop function.
Covers Batch 2 requirements:
- Strength bands: 0.54 (weak, name=None), 0.55 (moderate), 0.69 (moderate), 0.70 (strong)
- Null shape: object when matcher ran, None only when matcher is unavailable
- Shared face crop function (extract_face_crop)
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from app.components.celebrity_matcher import CelebrityMatcher, CelebrityMatchInternal
from app.utils.face_crop import extract_face_crop
from app.models.internal import ImageData
from app.config import settings


class TestCelebrityStrengthBands:
    """Test the strength band thresholds at 0.54, 0.55, 0.69, 0.70."""

    @pytest.fixture
    def mock_matcher(self):
        matcher = CelebrityMatcher(db_path="/dummy/path")
        matcher.celebrity_embeddings = {
            "Shah_Rukh_Khan": {
                "embedding": np.ones(512),
                "dataset_source": "Bollywood",
                "thumbnail_path": "Shah_Rukh_Khan/0.jpg",
                "display_name": "Shah Rukh Khan"
            }
        }
        matcher.is_available = True
        return matcher

    def test_strength_band_0_54_weak(self, mock_matcher):
        """At similarity 0.54 (< 0.55), strength is weak, name is None, message is 'No close resemblance found'."""
        mock_matcher.rank_celebrities = Mock(return_value=[("Shah_Rukh_Khan", 0.54)])
        result = mock_matcher._find_best_match(np.zeros(512))

        assert result.strength == "weak"
        assert result.name is None
        assert result.message == "No close resemblance found"
        assert result.similarity == 0.54
        assert result.confidence == 0.54
        assert len(result.top_matches) == 1
        assert result.top_matches[0]["strength"] == "weak"

    def test_strength_band_0_55_moderate(self, mock_matcher):
        """At similarity 0.55 (>= 0.55 and < 0.70), strength is moderate and name is present."""
        mock_matcher.rank_celebrities = Mock(return_value=[("Shah_Rukh_Khan", 0.55)])
        result = mock_matcher._find_best_match(np.zeros(512))

        assert result.strength == "moderate"
        assert result.name == "Shah Rukh Khan"
        assert result.message is None
        assert result.similarity == 0.55
        assert result.confidence == 0.55
        assert len(result.top_matches) == 1
        assert result.top_matches[0]["strength"] == "moderate"

    def test_strength_band_0_69_moderate(self, mock_matcher):
        """At similarity 0.69 (< 0.70), strength is moderate and name is present."""
        mock_matcher.rank_celebrities = Mock(return_value=[("Shah_Rukh_Khan", 0.69)])
        result = mock_matcher._find_best_match(np.zeros(512))

        assert result.strength == "moderate"
        assert result.name == "Shah Rukh Khan"
        assert result.message is None
        assert result.similarity == 0.69
        assert result.confidence == 0.69
        assert result.top_matches[0]["strength"] == "moderate"

    def test_strength_band_0_70_strong(self, mock_matcher):
        """At similarity 0.70 (>= 0.70), strength is strong and name is present."""
        mock_matcher.rank_celebrities = Mock(return_value=[("Shah_Rukh_Khan", 0.70)])
        result = mock_matcher._find_best_match(np.zeros(512))

        assert result.strength == "strong"
        assert result.name == "Shah Rukh Khan"
        assert result.message is None
        assert result.similarity == 0.70
        assert result.confidence == 0.70
        assert result.top_matches[0]["strength"] == "strong"


class TestSharedFaceCrop:
    """Test the shared extract_face_crop utility."""

    def test_crop_with_landmarks(self):
        """extract_face_crop crops to bounding box plus margin, resizes to (160, 160) and normalizes to [-1, 1]."""
        # Create 200x200 RGB image
        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        # Mock 4 landmarks forming a box around [50, 50] to [150, 150]
        landmarks = np.array([
            [50, 50],
            [150, 50],
            [150, 150],
            [50, 150]
        ], dtype=np.float32)

        crop, info = extract_face_crop(img, landmarks=landmarks, margin=0.10, return_info=True)

        assert crop is not None
        assert crop.shape == (160, 160, 3)
        assert crop.dtype == np.float32
        assert -1.0 <= crop.min() and crop.max() <= 1.0
        assert info["face_detected"] is True
        assert info["face_count"] == 1
        # Bounding box with 10% margin on 100px width/height: min=40, max=160
        assert info["box"] == (40, 40, 160, 160)

    def test_crop_fallback_128px(self):
        """For image <= 128px with no face detected, allows fallback to whole image."""
        img_small = np.full((64, 64, 3), 100, dtype=np.uint8)

        # Mock detector returning no face
        with patch("app.utils.face_crop.get_crop_detector") as mock_det:
            mock_res = Mock(face_landmarks=[])
            mock_det.return_value.detect.return_value = mock_res

            crop, info = extract_face_crop(img_small, allow_fallback_128=True, return_info=True)

            assert crop is not None
            assert crop.shape == (160, 160, 3)
            assert info["used_fallback"] is True

    def test_crop_rejects_multi_face(self):
        """If multiple faces are detected, returns None (skips multi-face image)."""
        img = np.full((300, 300, 3), 100, dtype=np.uint8)

        with patch("app.utils.face_crop.get_crop_detector") as mock_det:
            # 2 faces detected
            mock_res = Mock(face_landmarks=[[Mock(x=0.2, y=0.2)], [Mock(x=0.7, y=0.7)]])
            mock_det.return_value.detect.return_value = mock_res

            crop, info = extract_face_crop(img, landmarks=None, return_info=True)

            assert crop is None
            assert info["face_count"] == 2


class TestNullShapeConsistency:
    """Test consistent null shape: object when matcher ran, None only when unavailable."""

    @pytest.mark.asyncio
    async def test_celebrity_match_object_shape_on_weak_match(self):
        """When matcher runs and match is weak, celebrity_match is a full object (not None), with name=None."""
        from app.main import app
        from httpx import AsyncClient, ASGITransport
        import io
        from PIL import Image

        # Mock celebrity match internal returning weak resemblance
        weak_internal = CelebrityMatchInternal(
            name=None,
            similarity=0.32,
            strength="weak",
            confidence=0.32,
            dataset_source=None,
            thumbnail_path=None,
            message="No close resemblance found",
            top_matches=[
                {
                    "rank": 1,
                    "name": "Shah Rukh Khan",
                    "similarity": 0.32,
                    "strength": "weak",
                    "confidence": 0.32,
                    "dataset_source": "Bollywood",
                    "thumbnail_path": "Shah_Rukh_Khan/0.jpg"
                }
            ]
        )

        with patch("app.main.get_celebrity_matcher") as mock_get_matcher:
            mock_matcher = Mock()
            mock_matcher.is_available = True
            mock_matcher.find_match = AsyncMock(return_value=weak_internal)
            mock_get_matcher.return_value = mock_matcher

            img = Image.new("RGB", (320, 320), color=(128, 128, 128))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")

            with patch("app.main.get_face_detector") as mock_detector, \
                 patch("app.main.symmetry_classifier") as mock_sym, \
                 patch("app.main.golden_ratio_scorer") as mock_scorer, \
                 patch("app.main.overlay_generator") as mock_overlay, \
                 patch("app.main.percentile_engine") as mock_percentile:
                
                mock_detector.return_value.detect_faces.return_value = Mock(points=np.zeros((468, 3)))
                mock_sym.classify.return_value = Mock(value="oval")
                mock_scorer.score_all.return_value = Mock(overall_score=75.0, features=[
                    Mock(feature=f"f{i}", measured_ratio=1.618, ideal_ratio=1.618, score=100.0, deviation=0.0)
                    for i in range(8)
                ])
                mock_overlay.generate.return_value = Mock(
                    landmarks_2d=np.zeros((468, 2)),
                    midline_x=160.0,
                    horizontal_thirds_y=[40.0, 80.0, 120.0, 160.0],
                    phi_spiral_center=(160, 160)
                )
                mock_percentile.score_to_percentile.return_value = 70

                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    files = {"file": ("test.jpg", buf.getvalue(), "image/jpeg")}
                    data = {"include_celebrity_match": "true"}
                    resp = await ac.post("/api/analyze", files=files, data=data)

                assert resp.status_code == 200
                res = resp.json()["results"]
                celeb = res["celebrity_match"]

                # Consistent object shape: celebrity_match is NOT null
                assert celeb is not None
                assert isinstance(celeb, dict)
                assert celeb["name"] is None
                assert celeb["similarity"] == 0.32
                assert celeb["strength"] == "weak"
                assert celeb["message"] == "No close resemblance found"
                assert len(celeb["top_matches"]) == 1
                assert celeb["top_matches"][0]["strength"] == "weak"
