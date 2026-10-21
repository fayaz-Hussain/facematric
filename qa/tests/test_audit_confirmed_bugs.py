"""Regression tests for confirmed bugs from the 2026-10-22 repository audit."""
import inspect
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.components.celebrity_matcher import CelebrityMatcher
from app.components.face_detector import FaceDetector
from app.config import settings
from app.models.internal import ImageData
from app.models.responses import GoldenRatioGuides, OverlayData


class TestThumbnailUrlUsedForMatch:
    def test_find_best_match_uses_thumbnail_url_when_path_missing(self):
        matcher = CelebrityMatcher(db_path="/nonexistent/path.pkl")
        matcher.is_available = True
        matcher.celebrity_embeddings = {
            "Test_Celeb": {
                "embedding": np.ones(512, dtype=np.float32),
                "dataset_source": "Western",
                "thumbnail_url": "Test_Celeb/0.jpg",
                "display_name": "Test Celeb",
            }
        }
        matcher.rank_celebrities = lambda *_a, **_k: [("Test_Celeb", 0.80)]
        result = matcher._find_best_match(np.ones(512, dtype=np.float32))
        assert result.thumbnail_path == "Test_Celeb/0.jpg"
        assert result.top_matches[0]["thumbnail_path"] == "Test_Celeb/0.jpg"


class TestFileSizeHonorsSettings:
    def test_validate_and_read_file_uses_settings_max_image_size_mb(self):
        src = (ROOT / "backend" / "app" / "main.py").read_text(encoding="utf-8")
        assert "max_file_size_bytes = settings.max_image_size_mb * 1024 * 1024" in src
        assert "max_file_size_bytes = 10 * 1024 * 1024" not in src


class TestMultipleFacesCanBeDetected:
    @patch("app.components.face_detector.vision.FaceLandmarker.create_from_options")
    def test_detector_requests_at_least_two_faces(self, mock_create):
        mock_create.return_value = Mock()
        detector = FaceDetector(model_path="dummy_path.task")
        assert detector.num_faces_limit >= 2
        assert settings.mediapipe_max_faces == 1


class TestOverlayProcessedDimensions:
    def test_overlay_data_accepts_processed_image_size(self):
        overlay = OverlayData(
            landmarks=[[320.0, 240.0]],
            golden_ratio_guides=GoldenRatioGuides(
                vertical_midline=320.0,
                horizontal_thirds=[50.0, 120.0, 240.0, 360.0],
                phi_spiral_center=[320.0, 180.0],
            ),
            image_width=640,
            image_height=480,
        )
        display_w = 800
        wrong = 320.0 * (display_w / 4000.0)
        right = 320.0 * (display_w / overlay.image_width)
        assert right == pytest.approx(400.0)
        assert abs(right - wrong) > 50

    def test_frontend_uses_overlay_image_width(self):
        js = (ROOT / "frontend" / "src" / "js" / "results.js").read_text(encoding="utf-8")
        assert "overlayData.image_width" in js
        assert "const scaleX = displayWidth / naturalWidth" not in js

    def test_api_gateway_passes_landmark_image_size(self):
        src = (ROOT / "backend" / "app" / "main.py").read_text(encoding="utf-8")
        assert "image_width=landmarks.image_width" in src
        assert "image_height=landmarks.image_height" in src


class TestFacenet512No128Fallback:
    def test_generate_embedding_does_not_call_128d_facenet(self):
        src = inspect.getsource(CelebrityMatcher._generate_embedding)
        assert 'model_name="Facenet"' not in src
        assert "Facenet512" in inspect.getsource(CelebrityMatcher._initialize_facenet)

    def test_generate_embedding_raises_without_predict(self):
        matcher = CelebrityMatcher(db_path="/nonexistent/path.pkl")
        matcher.facenet_model = object()
        image = ImageData(array=np.zeros((160, 160, 3), dtype=np.uint8), width=160, height=160, original_format="JPEG")
        with patch("app.utils.face_crop.extract_face_crop", return_value=np.zeros((160, 160, 3), dtype=np.float32)):
            with pytest.raises(RuntimeError, match="FaceNet512"):
                matcher._generate_embedding(image)


class TestFrontendXss:
    def test_analyzing_error_modal_does_not_interpolate_message_in_html(self):
        js = (ROOT / "frontend" / "src" / "js" / "analyzing.js").read_text(encoding="utf-8")
        assert "${message}" not in js
        assert "${title}" not in js
        assert "titleEl.textContent = title" in js
        assert "messageEl.textContent = message" in js

    def test_results_does_not_interpolate_untrusted_names_in_innerhtml(self):
        js = (ROOT / "frontend" / "src" / "js" / "results.js").read_text(encoding="utf-8")
        assert "${match.name}" not in js
        assert "${feature.feature}" not in js
        assert "${message}" not in js
        assert "msgEl.textContent = message" in js
