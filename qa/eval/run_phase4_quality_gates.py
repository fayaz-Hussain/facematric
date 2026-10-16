"""
Phase 4: Quality Gates and Error Handling Evaluation Script.
Evaluates:
1. Acceptance on clean photos (30 frontal-set images) through /api/analyze
2. Occlusion detection handling on real & synthetic photos
3. Input handling edge cases:
   - Corrupt file
   - Empty body
   - Data-URI prefix
   - RGBA PNG
   - Grayscale
   - Animated GIF
   - WebP
   - BMP
   - EXIF-rotated phone photo
   - HEIC image (FM-BUG-006)
   - 20 MB+ file (>10MB limit)
   - Image under 300x300
   - Aspect ratio wider than 3:1
   - Simultaneous file and base64 inputs
   - Neither file nor base64 input
4. Contract verification across all error responses:
   - JSON structure: status, error_code, message, retry
"""

import sys
import os
import glob
import io
import base64
import json
import numpy as np
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
from app.utils.errors import ErrorCode

FRONTAL_DIR = os.path.join("qa", "data", "frontal_set")
OUTPUT_SUMMARY_JSON = os.path.join("qa", "reports", "04_quality_gates_summary.json")


def create_test_fixtures():
    """Create all required test fixture byte buffers."""
    fixtures = {}

    # Sample baseline face image for test cases
    sample_path = sorted(glob.glob(os.path.join(FRONTAL_DIR, "*.*")))[0]
    with open(sample_path, "rb") as f:
        baseline_face_bytes = f.read()
    fixtures["baseline_face_bytes"] = baseline_face_bytes
    fixtures["baseline_face_name"] = os.path.basename(sample_path)

    # 1. Corrupt file
    fixtures["corrupt_bytes"] = b"GIF89a\x00\x00\x00\x00\xff\x00\x00not an actual image"

    # 2. Empty bytes
    fixtures["empty_bytes"] = b""

    # 3. Compact base64 string for URL query parameter tests (<4KB)
    pil_face = Image.open(io.BytesIO(baseline_face_bytes))
    compact_im = pil_face.resize((300, 360))
    buf_c = io.BytesIO()
    compact_im.save(buf_c, format="JPEG", quality=20)
    raw_b64 = base64.b64encode(buf_c.getvalue()).decode("utf-8")
    fixtures["data_uri_b64"] = f"data:image/jpeg;base64,{raw_b64}"
    fixtures["raw_b64"] = raw_b64

    # 4. RGBA PNG (Real face with alpha channel)
    buf = io.BytesIO()
    pil_face = Image.open(io.BytesIO(baseline_face_bytes))
    pil_face.convert("RGBA").save(buf, format="PNG")
    fixtures["rgba_png_bytes"] = buf.getvalue()

    # 5. Grayscale JPEG (Real face in mode L)
    buf = io.BytesIO()
    pil_face.convert("L").save(buf, format="JPEG")
    fixtures["grayscale_jpeg_bytes"] = buf.getvalue()

    # 6. Animated GIF (Real face multi-frame)
    buf = io.BytesIO()
    pil_face.save(buf, format="GIF", save_all=True, append_images=[pil_face], loop=0)
    fixtures["animated_gif_bytes"] = buf.getvalue()

    # 7. WebP (Real face)
    buf = io.BytesIO()
    pil_face.save(buf, format="WEBP")
    fixtures["webp_bytes"] = buf.getvalue()

    # 8. BMP (Real face)
    buf = io.BytesIO()
    pil_face.save(buf, format="BMP")
    fixtures["bmp_bytes"] = buf.getvalue()

    # 9. EXIF-rotated photo (Simulated phone portrait: sensor captures ROTATE_270, EXIF=6)
    sideways_pixels = pil_face.transpose(Image.Transpose.ROTATE_270)
    exif = sideways_pixels.getexif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    sideways_pixels.save(buf, format="JPEG", exif=exif)
    fixtures["exif_rotated_bytes"] = buf.getvalue()

    # 10. HEIC
    heif_file = pillow_heif.from_pillow(pil_face)
    buf = io.BytesIO()
    heif_file.save(buf)
    fixtures["heic_bytes"] = buf.getvalue()

    # 11. 20 MB+ image
    large_pil = Image.new("RGB", (3500, 3500), color="purple")
    buf = io.BytesIO()
    large_pil.save(buf, format="JPEG", quality=100)
    raw_large = buf.getvalue()
    target_size = 21 * 1024 * 1024
    padded = raw_large + b"\x00" * max(0, target_size - len(raw_large))
    fixtures["large_20mb_bytes"] = padded

    # 12. Under 300x300 image (250x250)
    small_pil = pil_face.resize((250, 250), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    small_pil.save(buf, format="JPEG")
    fixtures["small_under_300_bytes"] = buf.getvalue()

    # 13. Wider than 3:1 (Aspect ratio 4:1, e.g. 1200x300)
    wide_pil = Image.new("RGB", (1200, 300), color="black")
    wide_pil.paste(pil_face.resize((300, 300)), (450, 0))
    buf = io.BytesIO()
    wide_pil.save(buf, format="JPEG")
    fixtures["wide_4_1_bytes"] = buf.getvalue()

    return fixtures


def verify_error_contract(resp_json, expected_code=None):
    """Verify that error response satisfies the PRD and API schema contract."""
    errors = []
def verify_error_contract(resp_json, expected_code=None):
    """Verify that error response satisfies the PRD and API schema contract."""
    errors = []
    if not isinstance(resp_json, dict):
        return False, ["Response body is not a JSON dictionary"], False, {}

    is_nested = "detail" in resp_json and isinstance(resp_json["detail"], dict)
    target = resp_json["detail"] if is_nested else resp_json

    if target.get("status") != "error":
        errors.append(f"Expected status='error', got {target.get('status')}")

    error_code = target.get("error_code")
    if not error_code or not isinstance(error_code, str):
        errors.append(f"Missing or invalid error_code: {error_code}")
    elif expected_code and error_code != expected_code:
        errors.append(f"Expected error_code={expected_code}, got {error_code}")

    message = target.get("message")
    if not message or not isinstance(message, str):
        errors.append(f"Missing or invalid message: {message}")

    retry = target.get("retry")
    if retry is None or not isinstance(retry, bool):
        errors.append(f"Missing or non-boolean retry field: {retry}")

    if is_nested:
        errors.append("Schema breach (FM-BUG-027): Error fields are nested under 'detail' instead of at the root level")

    return len(errors) == 0, errors, is_nested, target


def main():
    client = TestClient(app)
    fixtures = create_test_fixtures()

    summary = {
        "clean_photos": {
            "total": 0,
            "accepted": 0,
            "rejected": 0,
            "rejections": []
        },
        "occlusion_folders": {
            "sunglasses": "BLOCKED (Empty directory: no curated dataset)",
            "glasses": "BLOCKED (Empty directory: no curated dataset)",
            "mask": "BLOCKED (Empty directory: no curated dataset)",
            "hand": "BLOCKED (Empty directory: no curated dataset)",
            "hair": "BLOCKED (Empty directory: no curated dataset)"
        },
        "occlusion_spot_checks": [],
        "input_handling": {},
        "contract_verification": {}
    }

    # 1. Clean Photos Acceptance Test
    print("Testing clean frontal photos through /api/analyze...")
    clean_paths = sorted(glob.glob(os.path.join(FRONTAL_DIR, "*.*")))
    summary["clean_photos"]["total"] = len(clean_paths)

    for p in clean_paths:
        name = os.path.basename(p)
        with open(p, "rb") as f:
            content = f.read()
        resp = client.post("/api/analyze", files={"file": (name, content, "image/jpeg")})
        if resp.status_code == 200:
            summary["clean_photos"]["accepted"] += 1
        else:
            summary["clean_photos"]["rejected"] += 1
            body = resp.json()
            target = body.get("detail", body) if isinstance(body, dict) else {}
            summary["clean_photos"]["rejections"].append({
                "image": name,
                "status_code": resp.status_code,
                "error_code": target.get("error_code"),
                "message": target.get("message")
            })

    print(f"Clean photos: {summary['clean_photos']['accepted']}/{summary['clean_photos']['total']} accepted.")

    # 2. Occlusion Spot Checks
    print("\nRunning occlusion spot checks on clear glasses, sunglasses, and mask...")
    bill_gates_path = os.path.join(FRONTAL_DIR, "frontal_10_pins_Bill Gates_Bill Gates237_567.jpg")
    if os.path.exists(bill_gates_path):
        with open(bill_gates_path, "rb") as f:
            bg_bytes = f.read()
        resp = client.post("/api/analyze", files={"file": ("bill_gates.jpg", bg_bytes, "image/jpeg")})
        summary["occlusion_spot_checks"].append({
            "case": "Clear glasses (Bill Gates)",
            "status_code": resp.status_code,
            "outcome": "Accepted as valid (HTTP 200)" if resp.status_code == 200 else resp.json().get("error_code"),
            "note": "Allowed by relaxed z-depth threshold in OCCLUSION_FIX_SUMMARY (false accept if glasses strictly forbidden)"
        })

    pil_face = Image.open(io.BytesIO(fixtures["baseline_face_bytes"]))
    w, h = pil_face.size
    sg_pil = pil_face.copy()
    draw = ImageDraw.Draw(sg_pil)
    draw.rectangle([int(w * 0.2), int(h * 0.35), int(w * 0.8), int(h * 0.55)], fill="black")
    buf = io.BytesIO()
    sg_pil.save(buf, format="JPEG")
    resp_sg = client.post("/api/analyze", files={"file": ("sunglasses.jpg", buf.getvalue(), "image/jpeg")})
    body_sg = resp_sg.json() if resp_sg.status_code != 200 else {}
    target_sg = body_sg.get("detail", body_sg) if isinstance(body_sg, dict) else {}
    summary["occlusion_spot_checks"].append({
        "case": "Synthetic sunglasses (black eye box)",
        "status_code": resp_sg.status_code,
        "error_code": target_sg.get("error_code") if resp_sg.status_code != 200 else None,
        "outcome": "False Accept (HTTP 200)" if resp_sg.status_code == 200 else f"Rejected with {target_sg.get('error_code')}",
        "note": "MediaPipe predicted smooth z-depth through sunglasses; _detect_glasses failed to trigger (FM-BUG-017)"
    })

    mask_pil = pil_face.copy()
    draw = ImageDraw.Draw(mask_pil)
    draw.rectangle([int(w * 0.2), int(h * 0.60), int(w * 0.8), int(h * 0.95)], fill=(240, 240, 250))
    buf = io.BytesIO()
    mask_pil.save(buf, format="JPEG")
    resp_mask = client.post("/api/analyze", files={"file": ("mask.jpg", buf.getvalue(), "image/jpeg")})
    body_mask = resp_mask.json() if resp_mask.status_code != 200 else {}
    target_mask = body_mask.get("detail", body_mask) if isinstance(body_mask, dict) else {}
    summary["occlusion_spot_checks"].append({
        "case": "Synthetic surgical mask (lower face cover)",
        "status_code": resp_mask.status_code,
        "error_code": target_mask.get("error_code") if resp_mask.status_code != 200 else None,
        "outcome": "False Accept (HTTP 200)" if resp_mask.status_code == 200 else f"Rejected with {target_mask.get('error_code')}",
        "note": "MediaPipe predicted smooth z-depth through mask; _detect_mask failed to trigger (FM-BUG-017)"
    })

    # 3. Input Handling Matrix
    print("\nRunning comprehensive input handling test matrix...")
    test_cases = [
        ("corrupt_file", "/api/analyze", {"files": {"file": ("corrupt.jpg", fixtures["corrupt_bytes"], "image/jpeg")}}, 400, "INVALID_IMAGE"),
        ("empty_body", "/api/analyze", {"files": {"file": ("empty.jpg", fixtures["empty_bytes"], "image/jpeg")}}, 400, "INVALID_IMAGE"),
        ("data_uri_prefix_in_query", f"/api/analyze?image_base64={fixtures['data_uri_b64']}", {}, 400, "INVALID_IMAGE"),
        ("raw_base64_in_query", f"/api/analyze?image_base64={fixtures['raw_b64']}", {}, 200, None),
        ("raw_base64_in_post_body", "/api/analyze", {"data": {"image_base64": fixtures["raw_b64"]}}, 400, "NO_IMAGE"),
        ("rgba_png", "/api/analyze", {"files": {"file": ("test.png", fixtures["rgba_png_bytes"], "image/png")}}, 200, None),
        ("grayscale", "/api/analyze", {"files": {"file": ("gray.jpg", fixtures["grayscale_jpeg_bytes"], "image/jpeg")}}, 200, None),
        ("animated_gif", "/api/analyze", {"files": {"file": ("anim.gif", fixtures["animated_gif_bytes"], "image/gif")}}, 200, None),
        ("webp", "/api/analyze", {"files": {"file": ("image.webp", fixtures["webp_bytes"], "image/webp")}}, 200, None),
        ("bmp", "/api/analyze", {"files": {"file": ("image.bmp", fixtures["bmp_bytes"], "image/bmp")}}, 200, None),
        ("exif_rotated", "/api/analyze", {"files": {"file": ("exif.jpg", fixtures["exif_rotated_bytes"], "image/jpeg")}}, 200, None),
        ("heic_file", "/api/analyze", {"files": {"file": ("photo.heic", fixtures["heic_bytes"], "image/heic")}}, 400, "INVALID_IMAGE"),
        ("file_too_large_20mb", "/api/analyze", {"files": {"file": ("huge.jpg", fixtures["large_20mb_bytes"], "image/jpeg")}}, 400, "FILE_TOO_LARGE"),
        ("under_300x300", "/api/analyze", {"files": {"file": ("small.jpg", fixtures["small_under_300_bytes"], "image/jpeg")}}, 400, "POOR_QUALITY"),
        ("wider_than_3_1", "/api/analyze", {"files": {"file": ("wide.jpg", fixtures["wide_4_1_bytes"], "image/jpeg")}}, 400, "POOR_QUALITY"),
        ("both_file_and_base64", f"/api/analyze?image_base64={fixtures['raw_b64']}", {
            "files": {"file": ("sample.jpg", fixtures["baseline_face_bytes"], "image/jpeg")}
        }, 400, "INVALID_REQUEST"),
        ("neither_file_nor_base64", "/api/analyze", {}, 400, "NO_IMAGE")
    ]

    for tc_name, url, kwargs, expected_status, expected_error in test_cases:
        resp = client.post(url, **kwargs)
        body = resp.json()

        if isinstance(expected_status, list):
            status_match = (resp.status_code in expected_status)
        else:
            status_match = (resp.status_code == expected_status)

        target = body.get("detail", body) if isinstance(body, dict) else {}
        error_match = True
        if expected_error:
            error_match = (target.get("error_code") == expected_error)

        contract_valid = True
        contract_errors = []
        is_nested = False
        if resp.status_code != 200:
            contract_valid, contract_errors, is_nested, target = verify_error_contract(body, expected_code=expected_error)

        summary["input_handling"][tc_name] = {
            "url": url,
            "status_code": resp.status_code,
            "expected_status": expected_status,
            "status_match": status_match,
            "error_code": target.get("error_code") if resp.status_code != 200 else None,
            "expected_error": expected_error,
            "error_match": error_match,
            "is_nested_under_detail": is_nested,
            "contract_valid": contract_valid,
            "contract_errors": contract_errors,
            "message": target.get("message") if resp.status_code != 200 else None,
            "retry": target.get("retry") if resp.status_code != 200 else None
        }

    # Save summary JSON
    os.makedirs(os.path.dirname(OUTPUT_SUMMARY_JSON), exist_ok=True)
    with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nPhase 4 evaluation complete. Summary saved to {OUTPUT_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
