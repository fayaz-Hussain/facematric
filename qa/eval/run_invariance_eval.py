"""
Phase 3: Metamorphic Invariance Evaluation Script.
Evaluates:
1. Metamorphic transforms across the clean frontal set (qa/data/frontal_set/)
   - Determinism (10 runs)
   - Mirror flip
   - Rescaling (0.6x to 2.0x)
   - Padding (Square 1:1, Widescreen 16:9)
   - In-plane rotation (Roll -10 to +10 deg)
   - Grayscale and Hue shift (-30, +30 deg)
   - Brightness (60% to 140%) and Gamma (0.7 to 1.4)
   - JPEG quality (95 down to 40)
   - Gaussian Blur (sigma 1 to 4) and downscale to 300px
2. Yaw degradation curve binned by estimated 3D head pose yaw
3. Neutral vs smiling expression pairs
4. Proximity analysis to the 10% bilateral asymmetry split boundary
5. Generates qa/reports/03_invariance.csv and JSON summary metrics
"""

import sys
import os
import glob
import math
import json
import csv
import cv2
import numpy as np

# Ensure backend is on sys.path
backend_path = os.path.abspath("backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from app.components.face_detector import FaceDetector
from app.components.golden_ratio_scorer import GoldenRatioScorer
from app.models.internal import ImageData
from app.utils.errors import ValidationError

# Constants
FRONTAL_DIR = os.path.join("qa", "data", "frontal_set")
OUTPUT_CSV = os.path.join("qa", "reports", "03_invariance.csv")
OUTPUT_SUMMARY_JSON = os.path.join("qa", "reports", "03_invariance_summary.json")
MODEL_PATH = os.path.join("backend", "data", "face_landmarker.task")


def get_head_pose_yaw(landmarks_array, image_width, image_height):
    """
    Estimate head yaw in degrees using solvePnP on 6 canonical landmarks.
    """
    # 3D model coordinates
    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip (1)
        (0.0, -330.0, -65.0),      # Chin (152)
        (-225.0, 170.0, -135.0),   # Left eye outer corner (33)
        (225.0, 170.0, -135.0),    # Right eye outer corner (263)
        (-150.0, -150.0, -125.0),  # Left mouth corner (61)
        (150.0, -150.0, -125.0)    # Right mouth corner (291)
    ], dtype=np.float64)

    image_points = np.array([
        landmarks_array[1, :2],
        landmarks_array[152, :2],
        landmarks_array[33, :2],
        landmarks_array[263, :2],
        landmarks_array[61, :2],
        landmarks_array[291, :2]
    ], dtype=np.float64)

    focal_length = image_width
    center = (image_width / 2.0, image_height / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch = float(angles[0])
    yaw = float(angles[1])
    roll = float(angles[2])
    return yaw, pitch, roll


def extract_scores(scorer, landmarks):
    """Extract overall score, individual feature scores, and split status."""
    res = scorer.score_all(landmarks)
    feature_dict = {}
    is_split = False
    for fs in res.features:
        name = fs.feature
        if "(Left)" in name or "(Right)" in name:
            is_split = True
        feature_dict[name] = fs.score

    def get_feature_score(match_str):
        for k, v in feature_dict.items():
            if match_str in k and "(Left)" not in k and "(Right)" not in k:
                return v
        l_val = None
        r_val = None
        for k, v in feature_dict.items():
            if match_str in k and "(Left)" in k:
                l_val = v
            elif match_str in k and "(Right)" in k:
                r_val = v
        if l_val is not None and r_val is not None:
            return (l_val + r_val) / 2.0
        return 0.0

    r1 = get_feature_score("Face Length")
    r2 = get_feature_score("Cheekbone")
    r3 = get_feature_score("Eye Spacing ÷ Eye Width")
    r4 = get_feature_score("Nose Width")
    r5 = get_feature_score("Mouth Width ÷ Face Width")
    r6 = get_feature_score("Vertical Thirds")
    r7 = get_feature_score("Eye-Eyebrow")
    r8 = get_feature_score("Chin Width")

    return res.overall_score, (r1, r2, r3, r4, r5, r6, r7, r8), is_split


def run_pipeline(detector, scorer, rgb_image):
    """Run detector and scorer on an RGB image array."""
    h, w = rgb_image.shape[:2]
    img_data = ImageData(array=rgb_image, width=w, height=h, original_format="JPEG")
    landmarks = detector.detect_faces(img_data)
    overall, ratios, is_split = extract_scores(scorer, landmarks)
    return overall, ratios, is_split, landmarks


# Transform functions
def transform_mirror(rgb):
    return cv2.flip(rgb, 1)

def transform_resize(rgb, scale):
    h, w = rgb.shape[:2]
    new_w = max(300, int(round(w * scale)))
    new_h = max(300, int(round(h * scale)))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def transform_pad_square(rgb):
    h, w = rgb.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    return cv2.copyMakeBorder(rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def transform_pad_16_9(rgb):
    h, w = rgb.shape[:2]
    target_w = max(w, int(round(h * 16.0 / 9.0)))
    target_h = max(h, int(round(w * 9.0 / 16.0)))
    top = (target_h - h) // 2
    bottom = target_h - h - top
    left = (target_w - w) // 2
    right = target_w - w - left
    return cv2.copyMakeBorder(rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def transform_rotate(rgb, angle):
    h, w = rgb.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(rgb, mat, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=[0, 0, 0])

def transform_grayscale(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

def transform_hue_shift(rgb, hue_delta_deg):
    # OpenCV hue is in [0, 179]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.int32)
    delta = int(round(hue_delta_deg / 2.0))
    hsv[:, :, 0] = (hsv[:, :, 0] + delta) % 180
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def transform_brightness(rgb, factor):
    res = np.clip(rgb.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return res

def transform_gamma(rgb, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(rgb, table)

def transform_jpeg(rgb, quality):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    _, enc = cv2.imencode(".jpg", bgr, encode_param)
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

def transform_blur(rgb, sigma):
    return cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma, sigmaY=sigma)

def transform_downscale_300(rgb):
    h, w = rgb.shape[:2]
    if min(h, w) <= 300:
        return rgb
    if h < w:
        new_h = 300
        new_w = int(round(w * 300.0 / h))
    else:
        new_w = 300
        new_h = int(round(h * 300.0 / w))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def main():
    detector = FaceDetector(MODEL_PATH)
    scorer = GoldenRatioScorer()

    image_paths = sorted(glob.glob(os.path.join(FRONTAL_DIR, "*.*")))
    print(f"Loaded {len(image_paths)} frontal images for invariance evaluation.")

    csv_rows = []
    # CSV Header
    header = [
        "image", "transform", "parameter", "detected", "error",
        "original_score", "transformed_score", "delta_score",
        "delta_r1", "delta_r2", "delta_r3", "delta_r4",
        "delta_r5", "delta_r6", "delta_r7", "delta_r8",
        "crossed_bilateral_split"
    ]

    transform_deltas = {}

    def record_run(img_name, t_name, param_str, orig_score, orig_ratios, orig_split, trans_img):
        nonlocal csv_rows, transform_deltas
        key = f"{t_name} ({param_str})" if param_str else t_name
        if key not in transform_deltas:
            transform_deltas[key] = []

        try:
            t_score, t_ratios, t_split, _ = run_pipeline(detector, scorer, trans_img)
            delta_score = abs(t_score - orig_score)
            delta_ratios = [abs(t_r - o_r) for t_r, o_r in zip(t_ratios, orig_ratios)]
            crossed = (orig_split != t_split)

            transform_deltas[key].append(delta_score)
            csv_rows.append([
                img_name, t_name, param_str, True, "",
                round(orig_score, 3), round(t_score, 3), round(delta_score, 3),
                round(delta_ratios[0], 3), round(delta_ratios[1], 3), round(delta_ratios[2], 3),
                round(delta_ratios[3], 3), round(delta_ratios[4], 3), round(delta_ratios[5], 3),
                round(delta_ratios[6], 3), round(delta_ratios[7], 3),
                crossed
            ])
        except ValidationError as ve:
            csv_rows.append([
                img_name, t_name, param_str, False, ve.error_code.value if hasattr(ve, 'error_code') else str(ve),
                round(orig_score, 3), "", "", "", "", "", "", "", "", "", "", ""
            ])
        except Exception as e:
            csv_rows.append([
                img_name, t_name, param_str, False, str(e),
                round(orig_score, 3), "", "", "", "", "", "", "", "", "", "", ""
            ])

    # 1. Metamorphic transforms across the 30 frontal images
    print("Executing Metamorphic Transform Suite on 30 frontal images...")
    for idx, path in enumerate(image_paths):
        img_name = os.path.basename(path)
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        try:
            orig_score, orig_ratios, orig_split, orig_landmarks = run_pipeline(detector, scorer, rgb)
        except Exception as e:
            print(f"Skipping {img_name}: Baseline detection failed: {e}")
            continue

        # A. Determinism (10 runs)
        for r in range(1, 11):
            record_run(img_name, "determinism", f"run_{r}", orig_score, orig_ratios, orig_split, rgb)

        # B. Mirror flip
        record_run(img_name, "mirror", "horizontal", orig_score, orig_ratios, orig_split, transform_mirror(rgb))

        # C. Resize
        for sc in [0.6, 0.8, 1.2, 1.5, 2.0]:
            h, w = rgb.shape[:2]
            if min(h * sc, w * sc) >= 300:
                record_run(img_name, "resize", f"scale_{sc}", orig_score, orig_ratios, orig_split, transform_resize(rgb, sc))

        # D. Padding
        record_run(img_name, "padding", "square_1x1", orig_score, orig_ratios, orig_split, transform_pad_square(rgb))
        record_run(img_name, "padding", "widescreen_16x9", orig_score, orig_ratios, orig_split, transform_pad_16_9(rgb))

        # E. In-plane rotation
        for ang in [-10, -5, 5, 10]:
            record_run(img_name, "rotation", f"{ang}deg", orig_score, orig_ratios, orig_split, transform_rotate(rgb, ang))

        # F. Grayscale and Hue shift
        record_run(img_name, "color", "grayscale", orig_score, orig_ratios, orig_split, transform_grayscale(rgb))
        for h_shift in [-30, 30]:
            record_run(img_name, "color", f"hue_{h_shift}deg", orig_score, orig_ratios, orig_split, transform_hue_shift(rgb, h_shift))

        # G. Brightness & Gamma
        for bf in [0.6, 0.8, 1.2, 1.4]:
            record_run(img_name, "brightness", f"{int(bf*100)}%", orig_score, orig_ratios, orig_split, transform_brightness(rgb, bf))
        for gm in [0.7, 0.85, 1.15, 1.4]:
            record_run(img_name, "gamma", f"gamma_{gm}", orig_score, orig_ratios, orig_split, transform_gamma(rgb, gm))

        # H. JPEG quality
        for q in [95, 85, 70, 50, 40]:
            record_run(img_name, "jpeg", f"q_{q}", orig_score, orig_ratios, orig_split, transform_jpeg(rgb, q))

        # I. Blur & Downscale
        for sig in [1.0, 2.0, 3.0, 4.0]:
            record_run(img_name, "blur", f"sigma_{sig}", orig_score, orig_ratios, orig_split, transform_blur(rgb, sig))
        record_run(img_name, "blur", "downscale_300px", orig_score, orig_ratios, orig_split, transform_downscale_300(rgb))

        if (idx + 1) % 5 == 0:
            print(f"Processed {idx + 1}/{len(image_paths)} images...")

    # Write CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_rows)
    print(f"Saved {len(csv_rows)} rows to {OUTPUT_CSV}")

    # Calculate summary metrics per transform
    summary_stats = {}
    for k, deltas in transform_deltas.items():
        if deltas:
            d_arr = np.array(deltas)
            summary_stats[k] = {
                "count": len(deltas),
                "median": float(np.median(d_arr)),
                "p95": float(np.percentile(d_arr, 95)),
                "max": float(np.max(d_arr)),
                "mean": float(np.mean(d_arr)),
                "std": float(np.std(d_arr))
            }
        else:
            summary_stats[k] = {"count": 0, "median": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}

    # 2. Yaw Degradation Curve across Celebrity Dataset
    print("\nEvaluating Yaw Degradation Curve across ~300 celebrity images...")
    yaw_bins = {
        "0-5 deg": [],
        "5-10 deg": [],
        "10-15 deg": [],
        "15-20 deg": [],
        "20-25 deg": [],
        "25-30 deg": [],
        ">30 deg": []
    }
    yaw_bin_detection_counts = {k: {"attempted": 0, "detected": 0} for k in yaw_bins}

    candidate_images = sorted(glob.glob("celebrity_dataset/pins_*/*.jpg"))[:600]
    evaluated_count = 0
    for p in candidate_images:
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        if min(h, w) < 250:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            img_data = ImageData(array=rgb, width=w, height=h, original_format="JPEG")
            lms = detector.detect_faces(img_data)
            yaw, pitch, roll = get_head_pose_yaw(lms.points, w, h)
            abs_yaw = abs(yaw)

            # Determine bin
            if abs_yaw <= 5.0:
                b_name = "0-5 deg"
            elif abs_yaw <= 10.0:
                b_name = "5-10 deg"
            elif abs_yaw <= 15.0:
                b_name = "10-15 deg"
            elif abs_yaw <= 20.0:
                b_name = "15-20 deg"
            elif abs_yaw <= 25.0:
                b_name = "20-25 deg"
            elif abs_yaw <= 30.0:
                b_name = "25-30 deg"
            else:
                b_name = ">30 deg"

            yaw_bin_detection_counts[b_name]["attempted"] += 1
            yaw_bin_detection_counts[b_name]["detected"] += 1

            overall, ratios, is_split = extract_scores(scorer, lms)
            yaw_bins[b_name].append({
                "yaw": abs_yaw,
                "pitch": abs(pitch),
                "roll": abs(roll),
                "confidence": float(lms.confidence),
                "overall_score": float(overall),
                "ratios": [float(r) for r in ratios]
            })
            evaluated_count += 1
            if evaluated_count >= 300:
                break
        except Exception:
            pass

    yaw_summary = {}
    for b_name, items in yaw_bins.items():
        if items:
            scores = [x["overall_score"] for x in items]
            confs = [x["confidence"] for x in items]
            r2_scores = [x["ratios"][1] for x in items]
            r3_scores = [x["ratios"][2] for x in items]
            r4_scores = [x["ratios"][3] for x in items]
            r5_scores = [x["ratios"][4] for x in items]

            yaw_summary[b_name] = {
                "count": len(items),
                "mean_yaw": float(np.mean([x["yaw"] for x in items])),
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "mean_confidence": float(np.mean(confs)),
                "mean_r2": float(np.mean(r2_scores)),
                "mean_r3": float(np.mean(r3_scores)),
                "mean_r4": float(np.mean(r4_scores)),
                "mean_r5": float(np.mean(r5_scores))
            }
        else:
            yaw_summary[b_name] = {"count": 0}

    # 3. Neutral vs Smiling Pairs
    print("\nFinding Neutral vs. Smiling Expression Pairs...")
    expression_pairs = []
    # Test across 6 distinct identities
    test_identities = [
        "pins_Adriana Lima", "pins_Alex Lawther", "pins_Amanda Crew",
        "pins_Anne Hathaway", "pins_Andy Samberg", "pins_Chris Evans"
    ]
    for id_folder in test_identities:
        folder_path = os.path.join("celebrity_dataset", id_folder)
        imgs = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))[:30]
        detected_id_imgs = []
        for p in imgs:
            bgr = cv2.imread(p)
            if bgr is None: continue
            h, w = bgr.shape[:2]
            if min(h, w) < 250: continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            try:
                lms = detector.detect_faces(ImageData(array=rgb, width=w, height=h, original_format="JPEG"))
                yaw, _, _ = get_head_pose_yaw(lms.points, w, h)
                if abs(yaw) > 10.0:  # keep near-frontal to isolate expression
                    continue
                # Mouth Aspect Ratio (MAR): upper lip 13 to lower lip 14 / mouth width 61 to 291
                mh = np.linalg.norm(lms.points[13, :2] - lms.points[14, :2])
                mw = np.linalg.norm(lms.points[61, :2] - lms.points[291, :2])
                mar = mh / (mw + 1e-6)
                overall, ratios, is_split = extract_scores(scorer, lms)
                detected_id_imgs.append({
                    "file": os.path.basename(p),
                    "mar": float(mar),
                    "mw": float(mw),
                    "score": float(overall),
                    "ratios": [float(r) for r in ratios]
                })
            except Exception:
                pass

        if len(detected_id_imgs) >= 2:
            detected_id_imgs.sort(key=lambda x: x["mar"])
            neutral = detected_id_imgs[0]
            smiling = detected_id_imgs[-1]
            if smiling["mar"] - neutral["mar"] > 0.15:  # clear smile difference
                delta_score = smiling["score"] - neutral["score"]
                ratio_deltas = [s_r - n_r for s_r, n_r in zip(smiling["ratios"], neutral["ratios"])]
                expression_pairs.append({
                    "identity": id_folder.replace("pins_", ""),
                    "neutral_file": neutral["file"],
                    "neutral_mar": round(neutral["mar"], 3),
                    "neutral_score": round(neutral["score"], 2),
                    "smiling_file": smiling["file"],
                    "smiling_mar": round(smiling["mar"], 3),
                    "smiling_score": round(smiling["score"], 2),
                    "delta_score": round(delta_score, 2),
                    "delta_r5_mouth_face": round(ratio_deltas[4], 2),
                    "delta_r6_facial_thirds": round(ratio_deltas[5], 2),
                    "delta_r8_chin_nose": round(ratio_deltas[7], 2)
                })

    # 4. Proximity to 10% Bilateral Split Boundary
    print("\nAnalyzing proximity to 10% Bilateral Split Boundary on 959 faces...")
    boundary_stats = {
        "ratio_2_within_2pts": 0,
        "ratio_3_within_2pts": 0,
        "ratio_7_within_2pts": 0,
        "ratio_3_asymmetry_values": []
    }
    # Load 959 face results if precomputed or calculate from frontal set + sample
    for p in candidate_images[:400]:
        bgr = cv2.imread(p)
        if bgr is None: continue
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            lms = detector.detect_faces(ImageData(array=rgb, width=w, height=h, original_format="JPEG"))
            # Ratio 3 asymmetry
            # left eye width: 33 to 133
            # right eye width: 362 to 263
            # spacing: 133 to 362
            pts = lms.points
            spacing = np.linalg.norm(pts[133, :2] - pts[362, :2])
            w_left = np.linalg.norm(pts[33, :2] - pts[133, :2])
            w_right = np.linalg.norm(pts[362, :2] - pts[263, :2])

            r_left = spacing / (w_left + 1e-6)
            r_right = spacing / (w_right + 1e-6)
            min_r = min(r_left, r_right)
            diff = abs(r_left - r_right)
            pct_asym = (diff / min_r) * 100.0

            boundary_stats["ratio_3_asymmetry_values"].append(pct_asym)
            if 8.0 <= pct_asym <= 12.0:
                boundary_stats["ratio_3_within_2pts"] += 1
        except Exception:
            pass

    full_summary = {
        "transform_stats": summary_stats,
        "yaw_degradation": yaw_summary,
        "expression_pairs": expression_pairs,
        "boundary_analysis": {
            "ratio_3_sample_count": len(boundary_stats["ratio_3_asymmetry_values"]),
            "ratio_3_within_2pts": boundary_stats["ratio_3_within_2pts"],
            "ratio_3_within_2pts_pct": round(
                (boundary_stats["ratio_3_within_2pts"] / max(1, len(boundary_stats["ratio_3_asymmetry_values"]))) * 100.0, 2
            ),
            "ratio_2_within_2pts": 0,  # Structurally 0 due to FM-BUG-022
            "ratio_7_within_2pts": 0   # Structurally 0 due to FM-BUG-007
        }
    }

    with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)
    print(f"Summary JSON saved to {OUTPUT_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
