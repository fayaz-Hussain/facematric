"""
Phase 8 Automated API Integration Tests:
- POST /api/analyze contract compliance (multipart and base64)
- Error code coverage (400s and 408 timeout)
- Concurrent request execution
- Privacy & temporary file hygiene
"""

import sys
import os
import io
import time
import base64
import asyncio
import tempfile
import logging
import pytest
from PIL import Image
import httpx
from starlette.testclient import TestClient

PROJECT_ROOT = os.path.abspath(".")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

from app.main import app
from app.config import settings
import app.components.face_detector as fd_module

TEST_IMAGE_PATH = os.path.join(PROJECT_ROOT, "qa", "data", "frontal_set", "frontal_00_pins_Adriana Lima_Adriana Lima0_0.jpg")


@pytest.fixture(scope="module", autouse=True)
def setup_models():
    settings.mediapipe_model_path = os.path.abspath("backend/data/face_landmarker.task")
    settings.celebrity_db_path = os.path.abspath("backend/data/celebrity_embeddings.pkl")
    fd_module.face_detector = None


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestApiAnalyzeContract:
    """
    Contract verification for /api/analyze endpoint.
    """

    def test_multipart_upload_contract(self, client):
        assert os.path.exists(TEST_IMAGE_PATH), f"Test image missing: {TEST_IMAGE_PATH}"
        with open(TEST_IMAGE_PATH, "rb") as f:
            resp = client.post(
                "/api/analyze?include_celebrity_match=false&include_overlay_data=true",
                files={"file": ("test.jpg", f, "image/jpeg")}
            )

        assert resp.status_code == 200, f"Unexpected error: {resp.text}"
        data = resp.json()

        # Contract assertions
        assert data.get("status") == "success"
        assert isinstance(data.get("processing_time_ms"), int)
        assert "results" in data
        res = data["results"]

        # Scores
        assert 0.0 <= res["golden_ratio_score"] <= 100.0
        assert 0 <= res["percentile_rank"] <= 100

        # Classification
        assert res["face_symmetry_type"] in ["oval", "rectangle", "round", "square", "heart", "diamond"]

        # 8 Breakdown features
        breakdown = res.get("feature_breakdown", [])
        assert len(breakdown) >= 8, f"Expected at least 8 features, got {len(breakdown)}"
        for feat in breakdown:
            assert "feature" in feat
            assert 0.0 <= feat["score"] <= 100.0
            assert "measured_ratio" in feat

        # Overlay data
        overlay = res.get("overlay_data", {})
        assert overlay is not None, "overlay_data must be present"
        landmarks = overlay.get("landmarks", [])
        assert len(landmarks) == 468, f"Expected 468 landmarks, got {len(landmarks)}"
        assert len(landmarks[0]) == 2  # [x, y] format

        guides = overlay.get("golden_ratio_guides")
        assert guides is not None

    @pytest.mark.xfail(
        strict=True,
        reason="FM-BUG-028: Base64 in POST body fails because image_base64 parameter in main.py is typed as Query, not Body/Form."
    )
    def test_base64_in_post_body_contract(self, client):
        with open(TEST_IMAGE_PATH, "rb") as f:
            b64_str = base64.b64encode(f.read()).decode("utf-8")

        # Standard REST convention: base64 in JSON payload or form field
        resp = client.post(
            "/api/analyze",
            data={"image_base64": b64_str}
        )
        assert resp.status_code == 200
        assert resp.json().get("status") == "success"


class TestApiErrorCodes:
    """
    Verifies that all documented error codes are reachable with expected HTTP statuses.
    """

    def test_no_image_400(self, client):
        resp = client.post("/api/analyze")
        assert resp.status_code == 400
        data = resp.json()
        assert data.get("error_code") == "NO_IMAGE" or data.get("detail", {}).get("error_code") == "NO_IMAGE"

    def test_invalid_request_both_inputs_400(self, client):
        with open(TEST_IMAGE_PATH, "rb") as f:
            resp = client.post(
                "/api/analyze?image_base64=abc12345",
                files={"file": ("test.jpg", f, "image/jpeg")}
            )
        assert resp.status_code == 400
        data = resp.json()
        err = data.get("error_code") or data.get("detail", {}).get("error_code")
        assert err == "INVALID_REQUEST"

    def test_file_too_large_400(self, client):
        large_bytes = b"0" * (11 * 1024 * 1024)
        resp = client.post(
            "/api/analyze",
            files={"file": ("large.jpg", io.BytesIO(large_bytes), "image/jpeg")}
        )
        assert resp.status_code == 400
        data = resp.json()
        err = data.get("error_code") or data.get("detail", {}).get("error_code")
        assert err == "FILE_TOO_LARGE"

    def test_invalid_image_corrupt_400(self, client):
        corrupt = b"NOT_A_VALID_JPEG_HEADER_BYTES"
        resp = client.post(
            "/api/analyze",
            files={"file": ("corrupt.jpg", io.BytesIO(corrupt), "image/jpeg")}
        )
        assert resp.status_code == 400
        data = resp.json()
        err = data.get("error_code") or data.get("detail", {}).get("error_code")
        assert err == "INVALID_IMAGE"

    def test_poor_quality_small_dimensions_400(self, client):
        im = Image.new("RGB", (200, 200), color=(128, 128, 128))
        buf = io.BytesIO()
        im.save(buf, format="JPEG")
        resp = client.post(
            "/api/analyze",
            files={"file": ("small.jpg", buf.getvalue(), "image/jpeg")}
        )
        assert resp.status_code == 400
        data = resp.json()
        err = data.get("error_code") or data.get("detail", {}).get("error_code")
        assert err == "POOR_QUALITY"

    def test_no_face_detected_400(self, client):
        # Blank gray 400x400 image has no face
        im = Image.new("RGB", (400, 400), color=(100, 100, 100))
        buf = io.BytesIO()
        im.save(buf, format="JPEG")
        resp = client.post(
            "/api/analyze",
            files={"file": ("noface.jpg", buf.getvalue(), "image/jpeg")}
        )
        assert resp.status_code == 400
        data = resp.json()
        err = data.get("error_code") or data.get("detail", {}).get("error_code")
        assert err == "NO_FACE_DETECTED"

    def test_processing_timeout_408(self, client, monkeypatch):
        # Set processing_timeout to 0.001s to trigger timeout immediately
        orig_timeout = settings.processing_timeout
        monkeypatch.setattr(settings, "processing_timeout", 0.0001)

        with open(TEST_IMAGE_PATH, "rb") as f:
            resp = client.post(
                "/api/analyze",
                files={"file": ("test.jpg", f, "image/jpeg")}
            )

        assert resp.status_code == 408
        data = resp.json()
        # In main.py:221-229, TimeoutError raises HTTPException with detail dict
        detail = data.get("detail", {})
        assert detail.get("error_code") == "PROCESSING_TIMEOUT"
        assert detail.get("retry") is True


class TestConcurrentRequests:
    """
    Verifies that simultaneous API requests execute without cross-contamination.
    """

    @pytest.mark.asyncio
    async def test_two_simultaneous_requests(self):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            with open(TEST_IMAGE_PATH, "rb") as f:
                img_bytes = f.read()

            async def send_req(tag):
                return await ac.post(
                    "/api/analyze?include_celebrity_match=false",
                    files={"file": (f"test_{tag}.jpg", img_bytes, "image/jpeg")}
                )

            res1, res2 = await asyncio.gather(send_req("req1"), send_req("req2"))

            assert res1.status_code == 200
            assert res2.status_code == 200

            data1 = res1.json()
            data2 = res2.json()

            # Results must be identical for identical image
            assert data1["results"]["golden_ratio_score"] == data2["results"]["golden_ratio_score"]
            assert data1["status"] == "success"
            assert data2["status"] == "success"


class TestPrivacyAndTempFileHygiene:
    """
    Verifies that no uploaded image data or sensitive vectors leak into disk storage or log outputs.
    """

    def test_no_temporary_image_retention(self, client):
        temp_dir = tempfile.gettempdir()
        before_files = set(os.listdir(temp_dir))

        with open(TEST_IMAGE_PATH, "rb") as f:
            resp = client.post(
                "/api/analyze?include_celebrity_match=false",
                files={"file": ("privacy_test.jpg", f, "image/jpeg")}
            )
        assert resp.status_code == 200

        after_files = set(os.listdir(temp_dir))
        new_files = after_files - before_files

        # Ensure no JPEG, PNG, or image files remain in temp directory
        image_leaks = [f for f in new_files if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".heic"))]
        assert len(image_leaks) == 0, f"Temporary image files retained on disk: {image_leaks}"

    def test_logs_contain_no_base64_or_landmark_arrays(self, client):
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        logger = logging.getLogger()
        logger.addHandler(handler)

        try:
            with open(TEST_IMAGE_PATH, "rb") as f:
                client.post(
                    "/api/analyze?include_celebrity_match=false",
                    files={"file": ("log_test.jpg", f, "image/jpeg")}
                )
            log_output = log_stream.getvalue()

            # Ensure no huge base64 strings in logs
            assert "data:image/jpeg;base64," not in log_output
            # Ensure no 468-point raw landmark float arrays dumped in logs
            assert "landmark_467" not in log_output
            # Ensure no 512-d float vectors dumped
            assert "[0.0" not in log_output or "embedding" not in log_output
        finally:
            logger.removeHandler(handler)
