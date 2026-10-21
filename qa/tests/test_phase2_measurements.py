import sys
import os
import glob
import pytest
import numpy as np
import cv2

backend_path = os.path.abspath("backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)
project_root = os.path.abspath(".")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.components.face_detector import FaceDetector
from app.components.golden_ratio_scorer import GoldenRatioScorer
from app.models.internal import ImageData, Landmarks
from app.utils.errors import ValidationError
from qa.eval.oracle import GoldenRatioOracle, PHI_CONSTANT

FRONTAL_DIR = os.path.join(os.path.abspath("."), "qa", "data", "frontal_set")
LANDMARK_TASK = os.path.join(os.path.abspath("."), "backend", "data", "face_landmarker.task")


@pytest.fixture(scope="module")
def detector():
    return FaceDetector(LANDMARK_TASK)


@pytest.fixture(scope="module")
def scorer():
    return GoldenRatioScorer()


@pytest.fixture(scope="module")
def frontal_images():
    files = sorted(glob.glob(os.path.join(FRONTAL_DIR, "*.*")))
    assert len(files) >= 20, f"Expected at least 20 frontal images, found {len(files)}"
    return files


def test_independent_oracle_parity(detector, scorer, frontal_images):
    """
    Assert that all 8 measured ratios from production functions match
    the independent oracle to 1e-6 relative tolerance across all frontal images.
    """
    tested_count = 0
    for img_path in frontal_images:
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_data = ImageData(array=rgb, width=rgb.shape[1], height=rgb.shape[0], original_format="JPEG")
        try:
            landmarks = detector.detect_faces(img_data)
        except ValidationError:
            continue
        tested_count += 1
        pts = landmarks.points

        # Production raw measurement methods
        r1_prod = scorer._measure_face_length_to_width(pts)
        r2_l, r2_r = scorer._measure_cheekbone_to_jaw_bilateral(pts)
        r2_prod = (r2_l + r2_r) / 2.0
        r3_l, r3_r = scorer._measure_eye_spacing_to_width_bilateral(pts)
        r3_prod = (r3_l + r3_r) / 2.0
        r4_prod = scorer._measure_nose_to_mouth_width(pts)
        r5_prod = scorer._measure_mouth_to_face_width(pts)
        r6_prod = scorer._measure_vertical_thirds(pts)
        r7_l, r7_r = scorer._measure_eye_to_brow_distance_bilateral(pts)
        r7_prod = (r7_l + r7_r) / 2.0
        r8_prod = scorer._measure_chin_to_nose_width(pts)

        prod_ratios = {
            "r1": r1_prod, "r2": r2_prod, "r3": r3_prod, "r4": r4_prod,
            "r5": r5_prod, "r6": r6_prod, "r7": r7_prod, "r8": r8_prod
        }

        # Independent Oracle
        oracle_ratios = GoldenRatioOracle.calculate_production_ratios(pts)

        for k in ["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8"]:
            pval = prod_ratios[k]
            oval = oracle_ratios[k]
            rel_diff = abs(pval - oval) / max(pval, 1e-9)
            assert rel_diff <= 1e-6, f"Oracle mismatch on {k} for {img_path}: prod={pval}, oracle={oval}, diff={rel_diff}"

    assert tested_count >= 20, f"Expected at least 20 passing frontal images, tested {tested_count}"


def test_non_square_aspect_ratio_invariance(detector, scorer, frontal_images):
    """
    H2: Pad image to square (1:1) without scaling the face; overall score must match
    within tolerance (<= 1.0 point delta).
    """
    img_path = frontal_images[0]
    bgr = cv2.imread(img_path)
    h, w, _ = bgr.shape
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Base score
    img_orig = ImageData(array=rgb, width=w, height=h, original_format="JPEG")
    lms_orig = detector.detect_faces(img_orig)
    score_orig = scorer.score_all(lms_orig).overall_score

    # Pad to square
    max_side = max(w, h)
    pad_top = (max_side - h) // 2
    pad_bottom = max_side - h - pad_top
    pad_left = (max_side - w) // 2
    pad_right = max_side - w - pad_left
    square_rgb = cv2.copyMakeBorder(rgb, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    img_square = ImageData(array=square_rgb, width=max_side, height=max_side, original_format="JPEG")
    lms_square = detector.detect_faces(img_square)
    score_square = scorer.score_all(lms_square).overall_score

    delta = abs(score_square - score_orig)
    assert delta <= 1.0, f"Score changed by {delta:.2f} points when padded to square (original={score_orig}, square={score_square})"


def test_synthetic_unilateral_eye_distortion_splits_ratio_3(scorer):
    """
    Verify that distorting only the left eye triggers the bilateral asymmetry for Ratio 3,
    maintaining exactly 8 features while reporting ratio_left, ratio_right, and asymmetry_pct (FM-BUG-009).
    """
    pts = np.zeros((468, 3), dtype=np.float32)
    pts[10] = [200.0, 50.0, 0.0]     # Top
    pts[152] = [200.0, 373.6, 0.0]   # Bottom
    pts[234] = [100.0, 200.0, 0.0]   # Face left
    pts[454] = [300.0, 200.0, 0.0]   # Face right
    pts[172] = [138.2, 300.0, 0.0]
    pts[397] = [261.8, 300.0, 0.0]
    pts[133] = [160.0, 180.0, 0.0]   # Left inner
    pts[362] = [240.0, 180.0, 0.0]   # Right inner
    pts[33] = [110.56, 180.0, 0.0]   # Left outer
    pts[263] = [289.44, 180.0, 0.0]  # Right outer
    pts[159] = [135.0, 175.0, 0.0]
    pts[145] = [135.0, 185.0, 0.0]
    pts[70] = [135.0, 158.82, 0.0]
    pts[6] = [200.0, 173.6, 0.0]
    pts[219] = [180.0, 230.0, 0.0]
    pts[439] = [220.0, 230.0, 0.0]
    pts[61] = [187.64, 270.0, 0.0]
    pts[291] = [212.36, 270.0, 0.0]

    # Symmetric face -> exactly 8 features
    lms_sym = Landmarks(points=pts, confidence=1.0, image_width=400, image_height=400)
    res_sym = scorer.score_all(lms_sym)
    assert len(res_sym.features) == 8

    # Distort left eye outer landmark 33 outward
    distorted_pts = pts.copy()
    distorted_pts[33][0] -= 25.0
    lms_dist = Landmarks(points=distorted_pts, confidence=1.0, image_width=400, image_height=400)
    res_dist = scorer.score_all(lms_dist)

    assert len(res_dist.features) == 8
    r3_feat = next(fs for fs in res_dist.features if fs.feature == "Eye Spacing ÷ Eye Width")
    assert r3_feat.asymmetry_pct is not None
    assert r3_feat.asymmetry_pct > 10.0
    assert r3_feat.ratio_left != r3_feat.ratio_right


def test_ratio_2_asymmetry_detection_on_asymmetric_jaw(scorer):
    """
    Ratio 2 should detect jaw asymmetry when the right jaw is wider than the left jaw.
    FM-BUG-022: Fixed with midline perpendicular distance measurement.
    """
    pts = np.zeros((468, 3), dtype=np.float32)
    pts[10] = [200.0, 50.0, 0.0]
    pts[152] = [200.0, 373.6, 0.0]
    pts[234] = [100.0, 200.0, 0.0]
    pts[454] = [300.0, 200.0, 0.0]
    # Asymmetric jaw: left jaw at 150 (dist to center = 50), right jaw at 290 (dist to center = 90)
    # A 40px jaw discrepancy should trigger a bilateral split
    pts[172] = [150.0, 300.0, 0.0]
    pts[397] = [290.0, 300.0, 0.0]
    pts[133] = [160.0, 180.0, 0.0]
    pts[362] = [240.0, 180.0, 0.0]
    pts[33] = [110.56, 180.0, 0.0]
    pts[263] = [289.44, 180.0, 0.0]
    pts[159] = [135.0, 175.0, 0.0]
    pts[145] = [135.0, 185.0, 0.0]
    pts[70] = [135.0, 158.82, 0.0]
    pts[6] = [200.0, 173.6, 0.0]
    pts[219] = [180.0, 230.0, 0.0]
    pts[439] = [220.0, 230.0, 0.0]
    pts[61] = [187.64, 270.0, 0.0]
    pts[291] = [212.36, 270.0, 0.0]

    r2_left, r2_right = scorer._measure_cheekbone_to_jaw_bilateral(pts)
    # Expect asymmetry between left and right jaw ratios
    assert abs(r2_left - r2_right) > 0.10, f"Expected asymmetry > 0.10, but got r2_left={r2_left}, r2_right={r2_right}"


def test_ratio_7_asymmetry_detection_on_asymmetric_eyebrows(scorer):
    """
    Ratio 7 should detect eyebrow height asymmetry when right eyebrow is arched higher.
    FM-BUG-007: Fixed with bilateral mirror landmarks (70/159/145 vs 300/386/374).
    """
    pts = np.zeros((468, 3), dtype=np.float32)
    pts[159] = [135.0, 175.0, 0.0]   # Left eye top
    pts[145] = [135.0, 185.0, 0.0]   # Left eye bottom (height = 10)
    pts[70] = [135.0, 158.82, 0.0]   # Left brow (distance = 16.18)
    pts[386] = [265.0, 175.0, 0.0]   # Right eye top
    pts[374] = [265.0, 185.0, 0.0]   # Right eye bottom (height = 10)
    pts[300] = [265.0, 140.0, 0.0]   # Right brow raised significantly (distance = 35.0)

    r7_left, r7_right = scorer._measure_eye_to_brow_distance_bilateral(pts)
    assert abs(r7_left - r7_right) > 0.10, f"Expected asymmetry between eyebrows, but got r7_left={r7_left}, r7_right={r7_right}"
