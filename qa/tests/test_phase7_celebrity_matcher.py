import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from app.components.celebrity_matcher import CelebrityMatcher, CelebrityMatchInternal
from app.models.internal import ImageData


class TestCelebrityThresholdBehavior:
    """Behavior tests with mocked similarity (Phase 7.7)."""

    @pytest.fixture
    def matcher(self):
        m = CelebrityMatcher(db_path="/nonexistent/path")
        m.celebrity_embeddings = {
            "Emma_Stone": {
                "embedding": np.ones(512),
                "dataset_source": "Pins Face Recognition",
                "thumbnail_path": "thumbnails/emma.jpg"
            },
            "Priyanka_Chopra": {
                "embedding": np.zeros(512),
                "dataset_source": "Bollywood Celeb Dataset",
                "thumbnail_path": "thumbnails/priyanka.jpg"
            }
        }
        m.is_available = True
        return m

    def test_threshold_boundary_unrounded(self, matcher):
        """
        Verify boundary tests at 0.55:
        0.5499 -> no match
        0.5500 -> match
        0.5500001 -> match
        Assert comparison uses unrounded value (H11).
        """
        threshold = 0.55

        # 0.5499 is below 0.55 -> should NOT match
        res_below = matcher.get_best_match([("Emma_Stone", 0.5499)], threshold=threshold)
        assert res_below["name"] is None
        assert res_below["message"] == "No close match found in our dataset"

        # 0.5500 is equal to 0.55 -> SHOULD match
        res_exact = matcher.get_best_match([("Emma_Stone", 0.5500)], threshold=threshold)
        assert res_exact["name"] == "Emma Stone"
        assert res_exact["confidence_score"] == 0.55

        # 0.5500001 is above 0.55 -> SHOULD match
        res_above = matcher.get_best_match([("Emma_Stone", 0.5500001)], threshold=threshold)
        assert res_above["name"] == "Emma Stone"
        assert res_above["confidence_score"] == 0.55

    def test_below_threshold_contract(self, matcher):
        """
        Verify contract shape when below threshold (H9):
        {'name': None, 'message': 'No close match found in our dataset'}
        """
        res = matcher.get_best_match([("Emma_Stone", 0.32)], threshold=0.55)
        assert res == {
            "name": None,
            "message": "No close match found in our dataset"
        }

    def test_returned_name_and_similarity_belong_to_top_candidate(self, matcher):
        """
        Verify that returned candidate corresponds strictly to top ranked item.
        """
        ranked = [
            ("Priyanka_Chopra", 0.82),
            ("Emma_Stone", 0.65)
        ]
        res = matcher.get_best_match(ranked, threshold=0.55)
        assert res["name"] == "Priyanka Chopra"
        assert res["confidence_score"] == 0.82

    @pytest.mark.xfail(strict=True, reason="FM-BUG-010: _find_best_match has no threshold check, always returning a match with clamped score 0.60-0.96")
    def test_find_best_match_enforces_threshold_on_low_similarity(self, matcher):
        """
        Production _find_best_match should return name=None when similarity is very low (e.g. 0.15).
        Currently, it clamps (0.50 + 0.15 = 0.65) and returns a false match.
        """
        # Mock rank_celebrities to return a very low similarity
        matcher.rank_celebrities = Mock(return_value=[("Emma_Stone", 0.15)])
        
        result = matcher._find_best_match(np.zeros(512))
        
        # Expected behavior: low similarity face should return no match
        assert result.name is None, f"Expected None but got {result.name} with confidence {result.confidence}"
        assert result.message == "No close match found in our dataset"


class TestAPIProgressiveDegradation:
    """Test API level graceful degradation and parameter handling."""

    @pytest.mark.asyncio
    async def test_celebrity_module_exception_still_returns_200(self):
        """
        When celebrity matching raises an exception, the API must degrade gracefully
        and return HTTP 200 with golden ratio score intact.
        """
        from app.main import app
        from httpx import AsyncClient, ASGITransport

        # Mock celebrity_matcher.find_match to raise Exception
        with patch("app.main.get_celebrity_matcher") as mock_get_matcher:
            mock_matcher = MagicMock()
            mock_matcher.is_available = True
            mock_matcher.find_match.side_effect = RuntimeError("FaceNet inference failed")
            mock_get_matcher.return_value = mock_matcher

            # Generate synthetic test image
            import io
            from PIL import Image
            img = Image.new("RGB", (400, 400), color=(128, 128, 128))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            buf.seek(0)

            # We can mock face detection & scoring to isolate celebrity exception
            with patch("app.main.get_face_detector") as mock_detector, \
                 patch("app.main.symmetry_classifier") as mock_sym, \
                 patch("app.main.golden_ratio_scorer") as mock_scorer, \
                 patch("app.main.overlay_generator") as mock_overlay, \
                 patch("app.main.percentile_engine") as mock_percentile:
                
                mock_detector.return_value.detect_faces.return_value = MagicMock(points=[(0, 0, 0)]*468)
                mock_sym.classify.return_value = MagicMock(value="oval")
                mock_features = [
                    MagicMock(feature=f"Feature {i}", measured_ratio=1.618, ideal_ratio=1.618, score=100.0, deviation=0.0)
                    for i in range(8)
                ]
                mock_res = MagicMock(overall_score=75.0, features=mock_features)
                mock_scorer.score_all.return_value = mock_res
                mock_ov = MagicMock(landmarks_2d=np.zeros((468, 2)), midline_x=200.0, horizontal_thirds_y=[50.0, 100.0, 150.0, 200.0], phi_spiral_center=(200, 200))
                mock_overlay.generate.return_value = mock_ov
                mock_percentile.score_to_percentile.return_value = 65.0

                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    files = {"file": ("test.jpg", buf.getvalue(), "image/jpeg")}
                    data = {"include_celebrity_match": "true"}
                    response = await ac.post("/api/analyze", files=files, data=data)

                assert response.status_code == 200
                res_json = response.json()
                assert res_json["status"] == "success"
                assert res_json["results"]["golden_ratio_score"] == 75.0
                assert res_json["results"]["celebrity_match"] is None

    @pytest.mark.asyncio
    async def test_include_celebrity_false_skips_matching(self):
        """
        When include_celebrity_match=false, celebrity matching should not be called.
        """
        from app.main import app
        from httpx import AsyncClient, ASGITransport

        with patch("app.main.get_celebrity_matcher") as mock_get_matcher:
            mock_matcher = MagicMock()
            mock_get_matcher.return_value = mock_matcher

            import io
            from PIL import Image
            img = Image.new("RGB", (400, 400), color=(128, 128, 128))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            buf.seek(0)

            with patch("app.main.get_face_detector") as mock_detector, \
                 patch("app.main.symmetry_classifier") as mock_sym, \
                 patch("app.main.golden_ratio_scorer") as mock_scorer, \
                 patch("app.main.overlay_generator") as mock_overlay, \
                 patch("app.main.percentile_engine") as mock_percentile:
                
                mock_detector.return_value.detect_faces.return_value = MagicMock(points=[(0, 0, 0)]*468)
                mock_sym.classify.return_value = MagicMock(value="oval")
                mock_features = [
                    MagicMock(feature=f"Feature {i}", measured_ratio=1.618, ideal_ratio=1.618, score=100.0, deviation=0.0)
                    for i in range(8)
                ]
                mock_res = MagicMock(overall_score=75.0, features=mock_features)
                mock_scorer.score_all.return_value = mock_res
                mock_ov = MagicMock(landmarks_2d=np.zeros((468, 2)), midline_x=200.0, horizontal_thirds_y=[50.0, 100.0, 150.0, 200.0], phi_spiral_center=(200, 200))
                mock_overlay.generate.return_value = mock_ov
                mock_percentile.score_to_percentile.return_value = 65.0

                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    files = {"file": ("test.jpg", buf.getvalue(), "image/jpeg")}
                    data = {"include_celebrity_match": "false"}
                    response = await ac.post("/api/analyze", files=files, data=data)

                assert response.status_code == 200
                assert mock_matcher.find_match.call_count == 0
                res_json = response.json()
                assert res_json["results"]["celebrity_match"] is None
