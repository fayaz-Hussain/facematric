"""
Phase 4: Quality Gates and Error Handling Test Suite.
Tests:
1. Clean frontal photos acceptance rate through /api/analyze (expected 100%)
2. Multi-format support: RGBA PNG, Grayscale JPEG, WebP, BMP, Animated GIF, EXIF-rotated
3. Input validation errors: corrupt file, empty file, under 300x300, extreme aspect ratio >3:1, file >10MB
4. Request format validation: both inputs provided, neither input provided
5. Known bugs marked with xfail(strict=True):
   - FM-BUG-006: HEIC support fails without pillow-heif registered in app
   - FM-BUG-027: Error response contract nesting under 'detail'
   - FM-BUG-028: Base64 in POST body ignored (treated as query param)
   - FM-BUG-017: Synthetic sunglasses pass through without OCCLUSION_DETECTED
"""

import sys
import os
import glob
import io
import base64
import pytest
from PIL import Image, ImageDraw
import pillow_heif
from starlette.testclient import TestClient

backend_path = os.path.abspath("backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from app.config import settings
settings.mediapipe_model_path = os.path.abspath("backend/data/face_landmarker.task")
settings.celebrity_db_path = os.path.abspath("backend/data/celebrity_embeddings.pkl")

from app.components import face_detector as fd_module
fd_module.face_detector = None

from app.main import app

FRONTAL_DIR = os.path.join("qa", "data", "frontal_set")


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def sample_face_bytes():
    files = sorted(glob.glob(os.path.join(FRONTAL_DIR, "*.*")))
    assert len(files) >= 20, f"Expected >=20 frontal images, found {len(files)}"
    with open(files[0], "rb") as f:
        return f.read()


class TestCleanPhotoAcceptance:
    """Requirement: All clean near-frontal photos should pass analysis with HTTP 200."""

    def test_clean_frontal_acceptance(self, client):
        files = sorted(glob.glob(os.path.join(FRONTAL_DIR, "*.*")))
        accepted = 0
        rejections = []
        for p in files:
            name = os.path.basename(p)
            with open(p, "rb") as f:
                resp = client.post("/api/analyze", files={"file": (name, f.read(), "image/jpeg")})
            if resp.status_code == 200:
                accepted += 1
            else:
                rejections.append((name, resp.status_code, resp.json()))

        assert len(rejections) == 0, f"Expected 0 rejections, got {len(rejections)}: {rejections}"
        assert accepted == len(files) == 30


class TestImageFormatHandling:
    """Requirement 1.1: Multi-format support (PNG, WebP, BMP, GIF, EXIF JPEG)."""

    def test_rgba_png_acceptance(self, client, sample_face_bytes):
        im = Image.open(io.BytesIO(sample_face_bytes))
        buf = io.BytesIO()
        im.convert("RGBA").save(buf, format="PNG")
        resp = client.post("/api/analyze", files={"file": ("face.png", buf.getvalue(), "image/png")})
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_grayscale_jpeg_acceptance(self, client, sample_face_bytes):
        im = Image.open(io.BytesIO(sample_face_bytes))
        buf = io.BytesIO()
        im.convert("L").save(buf, format="JPEG")
        resp = client.post("/api/analyze", files={"file": ("face_gray.jpg", buf.getvalue(), "image/jpeg")})
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_webp_acceptance(self, client, sample_face_bytes):
        im = Image.open(io.BytesIO(sample_face_bytes))
        buf = io.BytesIO()
        im.save(buf, format="WEBP")
        resp = client.post("/api/analyze", files={"file": ("face.webp", buf.getvalue(), "image/webp")})
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_bmp_acceptance(self, client, sample_face_bytes):
        im = Image.open(io.BytesIO(sample_face_bytes))
        buf = io.BytesIO()
        im.save(buf, format="BMP")
        resp = client.post("/api/analyze", files={"file": ("face.bmp", buf.getvalue(), "image/bmp")})
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_animated_gif_acceptance(self, client, sample_face_bytes):
        im = Image.open(io.BytesIO(sample_face_bytes))
        buf = io.BytesIO()
        im.save(buf, format="GIF", save_all=True, append_images=[im], loop=0)
        resp = client.post("/api/analyze", files={"file": ("face.gif", buf.getvalue(), "image/gif")})
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_exif_rotated_phone_photo(self, client, sample_face_bytes):
        im = Image.open(io.BytesIO(sample_face_bytes))
        sideways = im.transpose(Image.Transpose.ROTATE_270)
        exif = sideways.getexif()
        exif[0x0112] = 6  # 90 deg CW
        buf = io.BytesIO()
        sideways.save(buf, format="JPEG", exif=exif)
        resp = client.post("/api/analyze", files={"file": ("phone.jpg", buf.getvalue(), "image/jpeg")})
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    @pytest.mark.xfail(strict=True, reason="FM-BUG-006: pillow-heif is not registered in production backend; HEIC returns INVALID_IMAGE")
    def test_heic_acceptance(self, client, sample_face_bytes):
        im = Image.open(io.BytesIO(sample_face_bytes))
        heif_file = pillow_heif.from_pillow(im)
        buf = io.BytesIO()
        heif_file.save(buf)
        resp = client.post("/api/analyze", files={"file": ("photo.heic", buf.getvalue(), "image/heic")})
        assert resp.status_code == 200, f"Expected 200 for HEIC format, got {resp.status_code}: {resp.json()}"


class TestValidationAndQualityGates:
    """Requirements: Dimensions, file size, aspect ratio, and request validation."""

    def test_empty_file_rejected(self, client):
        resp = client.post("/api/analyze", files={"file": ("empty.jpg", b"", "image/jpeg")})
        assert resp.status_code == 400
        body = resp.json()
        assert body["error_code"] == "INVALID_IMAGE"
        assert body["status"] == "error"

    def test_file_too_large_rejected(self, client):
        large_bytes = b"JPEG_TEST" + b"\x00" * (11 * 1024 * 1024)
        resp = client.post("/api/analyze", files={"file": ("huge.jpg", large_bytes, "image/jpeg")})
        assert resp.status_code == 400
        body = resp.json()
        assert body["error_code"] == "FILE_TOO_LARGE"

    def test_under_300x300_rejected(self, client, sample_face_bytes):
        im = Image.open(io.BytesIO(sample_face_bytes)).resize((250, 250))
        buf = io.BytesIO()
        im.save(buf, format="JPEG")
        resp = client.post("/api/analyze", files={"file": ("small.jpg", buf.getvalue(), "image/jpeg")})
        assert resp.status_code == 400
        target = resp.json().get("detail", resp.json())
        assert target["error_code"] == "POOR_QUALITY"
        assert "300x300" in target["message"]

    def test_wider_than_3_1_rejected(self, client, sample_face_bytes):
        im = Image.open(io.BytesIO(sample_face_bytes))
        wide = Image.new("RGB", (1200, 300), color="black")
        wide.paste(im.resize((300, 300)), (450, 0))
        buf = io.BytesIO()
        wide.save(buf, format="JPEG")
        resp = client.post("/api/analyze", files={"file": ("wide.jpg", buf.getvalue(), "image/jpeg")})
        assert resp.status_code == 400
        target = resp.json().get("detail", resp.json())
        assert target["error_code"] == "POOR_QUALITY"
        assert "3.0:1" in target["message"]

    def test_both_file_and_base64_rejected(self, client, sample_face_bytes):
        b64_str = base64.b64encode(sample_face_bytes[:100]).decode("utf-8")
        resp = client.post(
            f"/api/analyze?image_base64={b64_str}",
            files={"file": ("test.jpg", sample_face_bytes, "image/jpeg")}
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error_code"] == "INVALID_REQUEST"

    def test_neither_file_nor_base64_rejected(self, client):
        resp = client.post("/api/analyze")
        assert resp.status_code == 400
        body = resp.json()
        assert body["error_code"] == "NO_IMAGE"


class TestBugRegressions:
    """Strict xfail regressions for Phase 4 logged defects."""

    @pytest.mark.xfail(strict=True, reason="FM-BUG-027: Errors from _process_analysis are nested under 'detail' key instead of top-level ErrorResponse schema")
    def test_contract_top_level_error_schema_on_corrupt_file(self, client):
        resp = client.post("/api/analyze", files={"file": ("corrupt.jpg", b"corrupt", "image/jpeg")})
        assert resp.status_code == 400
        body = resp.json()
        # Documented contract requires top-level keys
        assert "status" in body, f"Expected top-level 'status', got nested under 'detail': {body}"
        assert body["status"] == "error"
        assert "error_code" in body
        assert body["error_code"] == "INVALID_IMAGE"

    @pytest.mark.xfail(strict=True, reason="FM-BUG-028: Base64 in POST body is ignored because route parameter is not declared as Form/Body")
    def test_base64_in_post_body_accepted(self, client, sample_face_bytes):
        im = Image.open(io.BytesIO(sample_face_bytes)).resize((300, 360))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=30)
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        resp = client.post("/api/analyze", data={"image_base64": b64_str})
        assert resp.status_code == 200, f"Expected 200 for base64 in POST body, got {resp.status_code}: {resp.json()}"

    @pytest.mark.xfail(strict=True, reason="FM-BUG-017: Heuristic occlusion detection fails on sunglasses; MediaPipe hallucinates face mesh and returns HTTP 200")
    def test_sunglasses_rejected_with_occlusion_detected(self, client, sample_face_bytes):
        im = Image.open(io.BytesIO(sample_face_bytes))
        w, h = im.size
        draw = ImageDraw.Draw(im)
        draw.rectangle([int(w * 0.2), int(h * 0.35), int(w * 0.8), int(h * 0.55)], fill="black")
        buf = io.BytesIO()
        im.save(buf, format="JPEG")
        resp = client.post("/api/analyze", files={"file": ("sunglasses.jpg", buf.getvalue(), "image/jpeg")})
        assert resp.status_code == 400, f"Expected 400 OCCLUSION_DETECTED, got {resp.status_code} (False Accept)"
        target = resp.json().get("detail", resp.json())
        assert target["error_code"] == "OCCLUSION_DETECTED"
