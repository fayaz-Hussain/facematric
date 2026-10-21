"""Reproduce suspected production bugs without loading FaceNet or downloading data."""
import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.components.celebrity_matcher import CelebrityMatcher
from app.config import settings
from app.main import _validate_and_read_file
from app.components.face_detector import FaceDetector


def repro_thumbnail_url_ignored():
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
    print("BUG thumbnail: thumbnail_path=", repr(result.thumbnail_path))
    print("BUG thumbnail: top_matches[0] thumbnail_path=", repr(result.top_matches[0].get("thumbnail_path")))
    return result.thumbnail_path in (None, "")


def repro_file_size_hardcoded():
    src = inspect.getsource(_validate_and_read_file)
    hardcoded = "10 * 1024 * 1024" in src and "settings.max_image_size_mb" not in src
    print("BUG file size hardcoded in _validate_and_read_file:", hardcoded)
    print("settings.max_image_size_mb:", settings.max_image_size_mb)
    return hardcoded


def repro_num_faces_cannot_see_groups():
    src = inspect.getsource(FaceDetector._initialize_detector)
    uses_setting_one = "num_faces=settings.mediapipe_max_faces" in src
    print("BUG num_faces uses settings.mediapipe_max_faces:", uses_setting_one, "value=", settings.mediapipe_max_faces)
    return uses_setting_one and settings.mediapipe_max_faces == 1


def repro_overlay_scale():
    # Processed max dim 640 vs original 4000
    natural_w, natural_h = 4000, 3000
    max_dim = settings.max_image_dimension
    scale = min(max_dim / natural_w, max_dim / natural_h)
    processed_w = int(natural_w * scale)
    processed_h = int(natural_h * scale)
    landmark_x = processed_w * 0.5
    display_w = 800
    wrong_scale = display_w / natural_w
    right_scale = display_w / processed_w
    wrong_px = landmark_x * wrong_scale
    right_px = landmark_x * right_scale
    print(f"BUG overlay: processed={processed_w}x{processed_h} landmark_x={landmark_x:.1f}")
    print(f"BUG overlay: frontend-using-natural maps to {wrong_px:.1f}px; correct {right_px:.1f}px")
    return abs(wrong_px - right_px) > 50


def repro_facenet_fallback_dim():
    src = inspect.getsource(CelebrityMatcher._generate_embedding)
    wrong = 'model_name="Facenet"' in src and "Facenet512" not in src.split("else:")[-1]
    print("BUG Facenet 128-d fallback present:", 'model_name="Facenet"' in src)
    return 'model_name="Facenet"' in src


def repro_xss_innerhtml():
    analyzing = (ROOT / "frontend" / "src" / "js" / "analyzing.js").read_text(encoding="utf-8")
    results = (ROOT / "frontend" / "src" / "js" / "results.js").read_text(encoding="utf-8")
    a = "${message}" in analyzing and "innerHTML" in analyzing
    r = "${message}" in results and "innerHTML" in results
    print("BUG XSS analyzing.js interpolates message in innerHTML:", a)
    print("BUG XSS results.js interpolates message in innerHTML:", r)
    print("BUG XSS results.js interpolates match.name in innerHTML:", "${match.name}" in results)
    return a or r or "${match.name}" in results


if __name__ == "__main__":
    findings = {
        "thumbnail_url_ignored": repro_thumbnail_url_ignored(),
        "file_size_hardcoded": repro_file_size_hardcoded(),
        "num_faces_cannot_see_groups": repro_num_faces_cannot_see_groups(),
        "overlay_scale_mismatch": repro_overlay_scale(),
        "facenet_fallback_dim": repro_facenet_fallback_dim(),
        "xss_innerhtml": repro_xss_innerhtml(),
    }
    print("\nCONFIRMED:" if any(findings.values()) else "\nnone")
    for k, v in findings.items():
        print(f"  {k}: {'CONFIRMED' if v else 'not reproduced'}")
