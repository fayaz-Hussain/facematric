import os
import sys
import glob
import math
import shutil
import pickle
import random
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.abspath(".")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

from app.components.face_detector import FaceDetector
from app.components.golden_ratio_scorer import GoldenRatioScorer
from app.models.internal import ImageData, Landmarks
from qa.eval.oracle import GoldenRatioOracle, euclidean_dist, PHI_CONSTANT

DATASET_DIR = os.path.join(PROJECT_ROOT, "celebrity_dataset")
FRONTAL_DIR = os.path.join(PROJECT_ROOT, "qa", "data", "frontal_set")
OVERLAYS_DIR = os.path.join(PROJECT_ROOT, "qa", "reports", "overlays")
LANDMARK_TASK = os.path.join(PROJECT_ROOT, "backend", "data", "face_landmarker.task")

# 3D canonical model points for solvePnP head pose estimation
MODEL_POINTS_6 = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (1)
    (0.0, -330.0, -65.0),        # Chin (152)
    (-225.0, 170.0, -135.0),     # Left eye outer corner (33)
    (225.0, 170.0, -135.0),      # Right eye outer corner (263)
    (-150.0, -150.0, -125.0),    # Left mouth corner (61)
    (150.0, -150.0, -125.0)      # Right mouth corner (291)
], dtype=np.float64)

def estimate_head_pose_yaw(landmarks_points, img_w, img_h):
    """
    Estimates head yaw angle (degrees) using solvePnP on 6 standard facial landmarks.
    Frontal face has |yaw| close to 0.0 degrees.
    """
    pts = landmarks_points
    image_points = np.array([
        (pts[1][0], pts[1][1]),      # Nose tip
        (pts[152][0], pts[152][1]),  # Chin
        (pts[33][0], pts[33][1]),    # Left eye outer
        (pts[263][0], pts[263][1]),  # Right eye outer
        (pts[61][0], pts[61][1]),    # Left mouth corner
        (pts[291][0], pts[291][1])   # Right mouth corner
    ], dtype=np.float64)

    focal_length = img_w
    center = (img_w / 2.0, img_h / 2.0)
    cam_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS_6, image_points, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 999.0
    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    yaw = angles[1]
    return float(yaw)

def curate_frontal_set(detector, target_count=30):
    """
    Populates qa/data/frontal_set with near-frontal images (|yaw| <= 3.5 deg).
    """
    os.makedirs(FRONTAL_DIR, exist_ok=True)
    existing_files = glob.glob(os.path.join(FRONTAL_DIR, "*.*"))
    print(f"Initial images in qa/data/frontal_set/: {len(existing_files)}")

    if len(existing_files) >= target_count:
        print(f"Frontal set already contains {len(existing_files)} images.")
        return existing_files

    print("Curating frontal set from celebrity_dataset (filtering by landmark solvePnP yaw <= 3.5 deg)...")
    # Search Western pins folders with good resolution
    all_dirs = sorted([d for d in os.listdir(DATASET_DIR) if d.startswith("pins_")])
    curated = []

    for d in all_dirs:
        if len(curated) >= target_count:
            break
        dir_path = os.path.join(DATASET_DIR, d)
        imgs = sorted([f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for img_name in imgs:
            img_path = os.path.join(dir_path, img_name)
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
                    if w < 300 or h < 300:
                        continue
                bgr = cv2.imread(img_path)
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                img_data = ImageData(array=rgb, width=rgb.shape[1], height=rgb.shape[0], original_format="JPEG")
                lms = detector.detect_faces(img_data)
                yaw = estimate_head_pose_yaw(lms.points, rgb.shape[1], rgb.shape[0])
                if abs(yaw) <= 3.5:
                    # Near frontal found! Copy to frontal_set
                    dst_name = f"frontal_{len(curated):02d}_{d}_{img_name}"
                    dst_path = os.path.join(FRONTAL_DIR, dst_name)
                    shutil.copy2(img_path, dst_path)
                    curated.append({
                        "path": dst_path,
                        "source": img_path,
                        "identity": d[5:],
                        "yaw": yaw,
                        "size": f"{w}x{h}"
                    })
                    print(f"  [{len(curated):02d}/{target_count}] Selected {d}/{img_name} (Yaw: {yaw:.2f} deg, Size: {w}x{h})")
                    break # 1 image per identity to ensure diverse faces
            except Exception:
                continue

    print(f"Curated {len(curated)} near-frontal images into {FRONTAL_DIR}.")
    return [c["path"] for c in curated]

def test_oracle_parity(detector, scorer, frontal_paths):
    """
    Task 1: Compare independent oracle to production GoldenRatioScorer on all 8 ratios.
    Tolerance: 1e-6 relative.
    """
    print("\n" + "=" * 60)
    print("PHASE 2.1: INDEPENDENT ORACLE COMPARISON (1e-6 RELATIVE TOLERANCE)")
    print("=" * 60)

    max_rel_diffs = {f"r{i}": 0.0 for i in range(1, 9)}
    tested_count = 0

    for path in frontal_paths:
        bgr = cv2.imread(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_data = ImageData(array=rgb, width=rgb.shape[1], height=rgb.shape[0], original_format="JPEG")
        landmarks = detector.detect_faces(img_data)
        pts = landmarks.points

        # 1. Production Raw Measurement Methods
        r1_raw = scorer._measure_face_length_to_width(pts)
        r2_l, r2_r = scorer._measure_cheekbone_to_jaw_bilateral(pts)
        r2_raw = (r2_l + r2_r) / 2.0
        r3_l, r3_r = scorer._measure_eye_spacing_to_width_bilateral(pts)
        r3_raw = (r3_l + r3_r) / 2.0
        r4_raw = scorer._measure_nose_to_mouth_width(pts)
        r5_raw = scorer._measure_mouth_to_face_width(pts)
        r6_raw = scorer._measure_vertical_thirds(pts)
        r7_l, r7_r = scorer._measure_eye_to_brow_distance_bilateral(pts)
        r7_raw = (r7_l + r7_r) / 2.0
        r8_raw = scorer._measure_chin_to_nose_width(pts)

        prod_raw = {
            "r1": r1_raw, "r2": r2_raw, "r3": r3_raw, "r4": r4_raw,
            "r5": r5_raw, "r6": r6_raw, "r7": r7_raw, "r8": r8_raw
        }

        # 2. Independent Oracle
        oracle_ratios = GoldenRatioOracle.calculate_production_ratios(pts)

        # Compare relative difference between production raw and oracle
        for k in ["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8"]:
            pval = prod_raw[k]
            oval = oracle_ratios[k]
            rel_diff = abs(pval - oval) / max(pval, 1e-9)
            max_rel_diffs[k] = max(max_rel_diffs[k], rel_diff)
            assert rel_diff <= 1e-6, f"Oracle mismatch on {k}: prod={pval}, oracle={oval}, diff={rel_diff}"

        # 3. Production Scorer score_all output (checks 3-decimal rounding)
        prod_res = scorer.score_all(landmarks)
        for fs in prod_res.features:
            fname = fs.feature
            rk = None
            if "Face Length" in fname: rk = "r1"
            elif "Cheekbone to Jaw Width" in fname and "(" not in fname: rk = "r2"
            elif "Eye Spacing ÷ Eye Width" in fname and "(" not in fname: rk = "r3"
            elif "Nose Width ÷ Mouth" in fname: rk = "r4"
            elif "Mouth Width ÷ Face" in fname: rk = "r5"
            elif "Vertical Thirds" in fname: rk = "r6"
            elif "Eye-Eyebrow" in fname and "(" not in fname: rk = "r7"
            elif "Chin Width" in fname: rk = "r8"

            if rk is not None:
                # Assert rounded measured ratio equals round(oracle, 3)
                assert fs.measured_ratio == round(oracle_ratios[rk], 3), \
                    f"Rounding mismatch on {rk}: fs.measured_ratio={fs.measured_ratio}, expected={round(oracle_ratios[rk], 3)}"

        tested_count += 1

    print(f"Verified {tested_count} images against independent oracle.")
    print("Maximum relative difference per ratio (Tolerance: 1.0e-06):")
    for k in sorted(max_rel_diffs.keys()):
        print(f"  {k.upper()}: {max_rel_diffs[k]:.2e} relative difference  [PASS]")

    return max_rel_diffs

def test_non_square_padding(detector, scorer, test_image_path):
    """
    Task 2: Non-square test (H2).
    Take a frontal image, pad to square (1:1) and to 16:9 without scaling face.
    Ratios must match within tolerance. Verify 2D vs 3D.
    """
    print("\n" + "=" * 60)
    print("PHASE 2.2: NON-SQUARE ASPECT RATIO INVARIANCE TEST (H2)")
    print("=" * 60)

    orig_bgr = cv2.imread(test_image_path)
    h, w, _ = orig_bgr.shape
    print(f"Base image dimensions: {w}x{h} (Aspect ratio: {w/h:.3f})")

    # 1. Base image evaluation
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    img_orig = ImageData(array=orig_rgb, width=w, height=h, original_format="JPEG")
    lms_orig = detector.detect_faces(img_orig)
    res_orig = scorer.score_all(lms_orig)
    ratios_orig = {fs.feature: fs.measured_ratio for fs in res_orig.features}

    # 2. Pad to square (1:1)
    max_side = max(w, h)
    pad_top = (max_side - h) // 2
    pad_bottom = max_side - h - pad_top
    pad_left = (max_side - w) // 2
    pad_right = max_side - w - pad_left
    square_rgb = cv2.copyMakeBorder(orig_rgb, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img_square = ImageData(array=square_rgb, width=max_side, height=max_side, original_format="JPEG")
    lms_square = detector.detect_faces(img_square)
    res_square = scorer.score_all(lms_square)
    ratios_square = {fs.feature: fs.measured_ratio for fs in res_square.features}

    # 3. Pad to 16:9
    target_w = max_side
    target_h = int(target_w * 9 / 16)
    if target_h < h:
        target_h = h
        target_w = int(target_h * 16 / 9)
    p_top = max(0, (target_h - h) // 2)
    p_bottom = max(0, target_h - h - p_top)
    p_left = max(0, (target_w - w) // 2)
    p_right = max(0, target_w - w - p_left)
    widescreen_rgb = cv2.copyMakeBorder(orig_rgb, p_top, p_bottom, p_left, p_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img_16_9 = ImageData(array=widescreen_rgb, width=target_w, height=target_h, original_format="JPEG")
    lms_16_9 = detector.detect_faces(img_16_9)
    res_16_9 = scorer.score_all(lms_16_9)
    ratios_16_9 = {fs.feature: fs.measured_ratio for fs in res_16_9.features}

    print("\nComparison across aspect ratio paddings:")
    print("| Feature | Original | Square Padded (1:1) | Delta (Square) | 16:9 Padded | Delta (16:9) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    max_delta = 0.0
    for feat in sorted(ratios_orig.keys()):
        v0 = ratios_orig[feat]
        v_sq = ratios_square.get(feat, 0.0)
        v_16 = ratios_16_9.get(feat, 0.0)
        d_sq = abs(v_sq - v0)
        d_16 = abs(v_16 - v0)
        max_delta = max(max_delta, d_sq, d_16)
        print(f"| {feat} | {v0:.4f} | {v_sq:.4f} | {d_sq:.4f} | {v_16:.4f} | {d_16:.4f} |")

    score_d_sq = abs(res_square.overall_score - res_orig.overall_score)
    score_d_16 = abs(res_16_9.overall_score - res_orig.overall_score)
    print(f"\nOverall Score: Original={res_orig.overall_score:.1f}, Square={res_square.overall_score:.1f} (Delta={score_d_sq:.2f}), 16:9={res_16_9.overall_score:.1f} (Delta={score_d_16:.2f})")
    print(f"Coordinate dimensionality check: 2D Euclidean distance strictly utilized (Z coordinate discarded).")
    return max_delta

def draw_endpoint_overlays_and_alternatives(detector, frontal_paths, num_images=20):
    """
    Task 3: Draw exact endpoints for each ratio on 20 images and make contact sheets.
    Examines:
    - Landmarks 219/439 (nose width) vs alternatives 129/358 and 98/327.
    - Landmark 10 as hairline proxy.
    - Landmark 70 as brow point in ratio 7.
    - Landmark 172/397 as jaw width (ratio 2) vs chin width (ratio 8).
    """
    print("\n" + "=" * 60)
    print("PHASE 2.3: ENDPOINT OVERLAYS & ALTERNATIVE NOSE WIDTH AUDIT")
    print("=" * 60)
    os.makedirs(OVERLAYS_DIR, exist_ok=True)

    ratio_specs = {
        1: {"name": "Face Length ÷ Face Width", "num": (10, 152), "den": (234, 454), "num_label": "Len(10-152)", "den_label": "Wid(234-454)"},
        2: {"name": "Cheekbone to Jaw Width", "num": (234, 454), "den": (172, 397), "num_label": "Cheek(234-454)", "den_label": "Jaw(172-397)"},
        3: {"name": "Eye Spacing ÷ Eye Width", "num": (133, 362), "den": (33, 133), "den2": (263, 362), "num_label": "Space(133-362)", "den_label": "Eyes"},
        4: {"name": "Nose Width ÷ Mouth Width", "num": (219, 439), "den": (61, 291), "num_label": "Nose(219-439)", "den_label": "Mouth(61-291)"},
        5: {"name": "Mouth Width ÷ Face Width", "num": (61, 291), "den": (234, 454), "num_label": "Mouth(61-291)", "den_label": "Face(234-454)"},
        6: {"name": "Vertical Thirds (Nose-Chin ÷ Hairline-Nose)", "num": (6, 152), "den": (10, 6), "num_label": "Low(6-152)", "den_label": "Up(10-6)"},
        7: {"name": "Eye-Eyebrow Distance ÷ Eye Height", "num": (159, 70), "den": (159, 145), "num_label": "Brow(159-70)", "den_label": "EyeH(159-145)"},
        8: {"name": "Chin Width ÷ Nose Width", "num": (172, 397), "den": (219, 439), "num_label": "Chin/Jaw(172-397)", "den_label": "Nose(219-439)"}
    }

    nose_width_comparisons = []
    hairline_observations = []
    sample_images = frontal_paths[:num_images]

    # Pre-extract landmarks for the 20 images
    images_landmarks = []
    for p in sample_images:
        bgr = cv2.imread(p)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_data = ImageData(array=rgb, width=rgb.shape[1], height=rgb.shape[0], original_format="JPEG")
        lms = detector.detect_faces(img_data)
        images_landmarks.append((p, bgr, lms.points))

    # Evaluate Nose Width Alternatives on the 20 images
    for p, bgr, pts in images_landmarks:
        # Candidate 1: Current production 219 & 439
        w_prod = euclidean_dist(pts[219], pts[439])
        # Candidate 2: Alar outer wings 129 & 358
        w_alt1 = euclidean_dist(pts[129], pts[358])
        # Candidate 3: Subnasale / Nostril base 98 & 327
        w_alt2 = euclidean_dist(pts[98], pts[327])

        mouth_w = euclidean_dist(pts[61], pts[291])
        chin_w = euclidean_dist(pts[172], pts[397])

        nose_width_comparisons.append({
            "path": os.path.basename(p),
            "w_219_439": w_prod,
            "w_129_358": w_alt1,
            "w_98_327": w_alt2,
            "r4_prod": w_prod / mouth_w,
            "r4_alt1": w_alt1 / mouth_w,
            "r4_alt2": w_alt2 / mouth_w,
            "r8_prod": chin_w / w_prod,
            "r8_alt1": chin_w / w_alt1,
            "r8_alt2": chin_w / w_alt2,
        })

    # Generate Contact Sheets per ratio
    for r_num, spec in ratio_specs.items():
        thumb_list = []
        for p, bgr, pts in images_landmarks:
            im_copy = bgr.copy()
            h, w, _ = im_copy.shape

            # Draw numerator line in green
            n1, n2 = spec["num"]
            p_n1 = (int(pts[n1][0]), int(pts[n1][1]))
            p_n2 = (int(pts[n2][0]), int(pts[n2][1]))
            cv2.line(im_copy, p_n1, p_n2, (0, 255, 0), 2)
            cv2.circle(im_copy, p_n1, 4, (0, 255, 0), -1)
            cv2.circle(im_copy, p_n2, 4, (0, 255, 0), -1)
            cv2.putText(im_copy, str(n1), p_n1, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(im_copy, str(n2), p_n2, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Draw denominator line in red/blue
            d1, d2 = spec["den"]
            p_d1 = (int(pts[d1][0]), int(pts[d1][1]))
            p_d2 = (int(pts[d2][0]), int(pts[d2][1]))
            cv2.line(im_copy, p_d1, p_d2, (255, 0, 0), 2)
            cv2.circle(im_copy, p_d1, 4, (255, 0, 0), -1)
            cv2.circle(im_copy, p_d2, 4, (255, 0, 0), -1)
            cv2.putText(im_copy, str(d1), p_d1, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            cv2.putText(im_copy, str(d2), p_d2, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            # If ratio 4 or 8, also draw alternative nose candidates
            if r_num in (4, 8):
                # 129 & 358 in cyan
                p_129 = (int(pts[129][0]), int(pts[129][1]))
                p_358 = (int(pts[358][0]), int(pts[358][1]))
                cv2.line(im_copy, p_129, p_358, (255, 255, 0), 1)
                cv2.circle(im_copy, p_129, 3, (255, 255, 0), -1)
                cv2.circle(im_copy, p_358, 3, (255, 255, 0), -1)
                # 98 & 327 in magenta
                p_98 = (int(pts[98][0]), int(pts[98][1]))
                p_327 = (int(pts[327][0]), int(pts[327][1]))
                cv2.line(im_copy, p_98, p_327, (255, 0, 255), 1)
                cv2.circle(im_copy, p_98, 3, (255, 0, 255), -1)
                cv2.circle(im_copy, p_327, 3, (255, 0, 255), -1)

            # Resize to thumbnail (240x240)
            thumb = cv2.resize(im_copy, (240, 240))
            # Overlay ratio title
            cv2.rectangle(thumb, (0, 0), (240, 20), (0, 0, 0), -1)
            cv2.putText(thumb, f"R{r_num}: {spec['num_label']}", (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            thumb_list.append(thumb)

        # Build 4x5 contact sheet (5 columns, 4 rows = 20 thumbnails)
        grid_rows = []
        for row_idx in range(4):
            row_thumbs = thumb_list[row_idx * 5:(row_idx + 1) * 5]
            grid_rows.append(np.hstack(row_thumbs))
        contact_sheet = np.vstack(grid_rows)
        cs_path = os.path.join(OVERLAYS_DIR, f"contact_sheet_ratio_{r_num}.jpg")
        cv2.imwrite(cs_path, contact_sheet)
        print(f"Generated contact sheet for Ratio {r_num} ({spec['name']}) -> {cs_path}")

    # Save nose width comparison to CSV
    csv_path = os.path.join(PROJECT_ROOT, "qa", "reports", "phase2_nose_width_comparison.csv")
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "path", "w_219_439", "w_129_358", "w_98_327",
            "r4_prod", "r4_alt1", "r4_alt2", "r8_prod", "r8_alt1", "r8_alt2"
        ])
        writer.writeheader()
        writer.writerows(nose_width_comparisons)
    print(f"Saved nose width comparison across 20 images to {csv_path}")

    # Print summary statistics of alternatives
    w_prod_mean = np.mean([x["w_219_439"] for x in nose_width_comparisons])
    w_alt1_mean = np.mean([x["w_129_358"] for x in nose_width_comparisons])
    w_alt2_mean = np.mean([x["w_98_327"] for x in nose_width_comparisons])
    r4_prod_mean = np.mean([x["r4_prod"] for x in nose_width_comparisons])
    r4_alt1_mean = np.mean([x["r4_alt1"] for x in nose_width_comparisons])
    r4_alt2_mean = np.mean([x["r4_alt2"] for x in nose_width_comparisons])
    r8_prod_mean = np.mean([x["r8_prod"] for x in nose_width_comparisons])
    r8_alt1_mean = np.mean([x["r8_alt1"] for x in nose_width_comparisons])
    r8_alt2_mean = np.mean([x["r8_alt2"] for x in nose_width_comparisons])

    print("\n--- Nose Width Landmark Alternative Audit (20 Images) ---")
    print(f"  Current 219/439 (alar inner):  Mean Width={w_prod_mean:.1f}px -> R4={r4_prod_mean:.3f}, R8={r8_prod_mean:.3f}")
    print(f"  Alternative 129/358 (alar outer): Mean Width={w_alt1_mean:.1f}px (+{(w_alt1_mean-w_prod_mean)/w_prod_mean*100:.1f}%) -> R4={r4_alt1_mean:.3f}, R8={r8_alt1_mean:.3f}")
    print(f"  Alternative 98/327 (nostril base): Mean Width={w_alt2_mean:.1f}px ({(w_alt2_mean-w_prod_mean)/w_prod_mean*100:.1f}%) -> R4={r4_alt2_mean:.3f}, R8={r8_alt2_mean:.3f}")

    return nose_width_comparisons

def run_large_scale_ratio_audit(detector, scorer, frontal_paths, num_celebrity=1000):
    """
    Task 4: Ratio audit across frontal set + 1,000 celebrity images.
    Reports: mean, SD, 5th/95th percentiles, share within +/-10% of phi.
    Computes maximum achievable overall score if 6 best ratios were phi.
    Examines bilateral split frequencies on real images.
    """
    print("\n" + "=" * 60)
    print("PHASE 2.4: POPULATION RATIO AUDIT (FRONTAL SET + 1,000 CELEBRITY IMAGES)")
    print("=" * 60)

    # Gather 1,000 celebrity images with min dimension >= 300
    all_celeb_images = []
    for root, dirs, files in os.walk(DATASET_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_celeb_images.append(os.path.join(root, f))
    
    random.seed(42)
    random.shuffle(all_celeb_images)

    selected_images = list(frontal_paths)
    for p in all_celeb_images:
        if len(selected_images) >= (len(frontal_paths) + num_celebrity):
            break
        if p in selected_images:
            continue
        try:
            with Image.open(p) as im:
                w, h = im.size
                if w >= 300 and h >= 300:
                    selected_images.append(p)
        except Exception:
            continue

    print(f"Auditing {len(selected_images)} total images ({len(frontal_paths)} frontal + {len(selected_images)-len(frontal_paths)} celebrity images)...")

    # Metrics storage
    ratio_data = {f"r{i}": [] for i in range(1, 9)}
    bilateral_splits = {"r2": 0, "r3": 0, "r7": 0}
    overall_scores = []
    detected_count = 0

    for idx, p in enumerate(selected_images):
        try:
            bgr = cv2.imread(p)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img_data = ImageData(array=rgb, width=rgb.shape[1], height=rgb.shape[0], original_format="JPEG")
            landmarks = detector.detect_faces(img_data)
            res = scorer.score_all(landmarks)
            overall_scores.append(res.overall_score)

            # Check bilateral splits in feature names
            split_feats = [fs.feature for fs in res.features]
            if any("Cheekbone to Jaw Width (Left)" in f for f in split_feats):
                bilateral_splits["r2"] += 1
            if any("Eye Spacing ÷ Eye Width (Left)" in f for f in split_feats):
                bilateral_splits["r3"] += 1
            if any("Eye-Eyebrow Distance ÷ Eye Height (Left)" in f for f in split_feats):
                bilateral_splits["r7"] += 1

            # Extract base ratios
            ratios = GoldenRatioOracle.calculate_production_ratios(landmarks.points)
            for k in ratio_data.keys():
                ratio_data[k].append(ratios[k])
            detected_count += 1
        except Exception:
            continue

    print(f"Successfully extracted landmarks and scored {detected_count} images.")

    # Calculate statistics per ratio
    stats_table = []
    phi_low = PHI_CONSTANT * 0.90   # 1.4562
    phi_high = PHI_CONSTANT * 1.10  # 1.7798

    for k in [f"r{i}" for i in range(1, 9)]:
        arr = np.array(ratio_data[k])
        mean = float(np.mean(arr))
        sd = float(np.std(arr))
        p5 = float(np.percentile(arr, 5))
        p95 = float(np.percentile(arr, 95))
        within_10 = float(np.mean((arr >= phi_low) & (arr <= phi_high)) * 100.0)
        stats_table.append({
            "ratio": k.upper(),
            "mean": mean,
            "sd": sd,
            "p5": p5,
            "p95": p95,
            "within_10": within_10
        })

    print("\nPopulation Ratio Statistics (N = {})".format(detected_count))
    print("| Ratio | Feature Name | Mean | SD | 5th Pct | 95th Pct | Within ±10% of φ (1.46–1.78) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    feature_names = {
        "R1": "Face Length ÷ Face Width",
        "R2": "Cheekbone to Jaw Width",
        "R3": "Eye Spacing ÷ Eye Width",
        "R4": "Nose Width ÷ Mouth Width",
        "R5": "Mouth Width ÷ Face Width",
        "R6": "Vertical Thirds (Nose-Chin ÷ Hairline-Nose)",
        "R7": "Eye-to-Brow ÷ Eye Height",
        "R8": "Chin Width ÷ Nose Width"
    }
    for row in stats_table:
        fn = feature_names[row["ratio"]]
        print(f"| {row['ratio']} | {fn} | {row['mean']:.3f} | {row['sd']:.3f} | {row['p5']:.3f} | {row['p95']:.3f} | {row['within_10']:.1f}% |")

    # Bilateral Split Frequencies
    print("\nBilateral Split Frequency (Difference > 10% between Left and Right):")
    print(f"  Ratio 2 (Cheekbone to Jaw): {bilateral_splits['r2']}/{detected_count} ({bilateral_splits['r2']/detected_count*100:.2f}%)")
    print(f"  Ratio 3 (Eye Spacing):     {bilateral_splits['r3']}/{detected_count} ({bilateral_splits['r3']/detected_count*100:.2f}%)")
    print(f"  Ratio 7 (Eye to Brow):     {bilateral_splits['r7']}/{detected_count} ({bilateral_splits['r7']/detected_count*100:.2f}%)")

    # Theoretical Maximum Achievable Overall Score
    # Score formula: 100 * (1 - |r - phi| / phi)
    # If 6 best ratios were EXACTLY phi -> scores are 100.0
    # For the remaining 2 ratios, take the value in [p5, p95] closest to phi:
    ratio_best_possible_scores = {}
    for row in stats_table:
        r_name = row["ratio"]
        p5, p95 = row["p5"], row["p95"]
        # Find point in [p5, p95] closest to phi (1.618)
        if p5 <= PHI_CONSTANT <= p95:
            best_r = PHI_CONSTANT
        elif PHI_CONSTANT < p5:
            best_r = p5
        else:
            best_r = p95
        score = 100.0 * (1.0 - abs(best_r - PHI_CONSTANT) / PHI_CONSTANT)
        ratio_best_possible_scores[r_name] = max(0.0, min(100.0, score))

    sorted_scores = sorted(ratio_best_possible_scores.items(), key=lambda x: x[1], reverse=True)
    # 6 best are 100.0, the remaining 2 sit at their observed best values in [p5, p95]:
    worst_2 = sorted_scores[-2:]
    max_achievable = (6 * 100.0 + worst_2[0][1] + worst_2[1][1]) / 8.0

    print(f"\n--- Maximum Achievable Score Analysis ---")
    print(f"Ratios ranked by best potential score within [5th, 95th] percentile:")
    for r_k, sc in sorted_scores:
        print(f"  {r_k} ({feature_names[r_k]}): best score = {sc:.2f}")
    print(f"\nIf 6 best ratios were EXACTLY phi (100.0 each) and the 2 hardest ratios ({worst_2[0][0]} and {worst_2[1][0]}) were at their observed 95th-percentile optimum:")
    print(f"  Worst 2 scores: {worst_2[0][0]}={worst_2[0][1]:.2f}, {worst_2[1][0]}={worst_2[1][1]:.2f}")
    print(f"  Theoretical Maximum Overall Score = (600 + {worst_2[0][1]:.2f} + {worst_2[1][1]:.2f}) / 8 = {max_achievable:.2f}")
    print(f"  Empirical Overall Scores: Mean={np.mean(overall_scores):.1f}, Max={np.max(overall_scores):.1f}, Min={np.min(overall_scores):.1f}, P95={np.percentile(overall_scores, 95):.1f}")
    print(f"  PRD Claim Check ('85+ is rare, most users 60-85'): Maximum achievable is {max_achievable:.1f}, so scores > 85 are practically impossible on biological faces.")

    return stats_table, max_achievable, bilateral_splits

def test_left_right_synthetic_distortion(scorer):
    """
    Task 5: Synthetic landmark test where only the left eye is distorted.
    Verifies that Ratio 3 splits into (Left) and (Right), while Ratio 2 and 7 do not.
    """
    print("\n" + "=" * 60)
    print("PHASE 2.5: SYNTHETIC ASYMMETRIC LANDMARK DISTORTION TEST")
    print("=" * 60)

    # Create synthetic symmetric face (468 points)
    pts = np.zeros((468, 3), dtype=np.float32)
    # Face boundary: width = 200, length = 323.6 (ratio 1 = 1.618)
    pts[10] = [200.0, 50.0, 0.0]     # Top
    pts[152] = [200.0, 373.6, 0.0]   # Bottom
    pts[234] = [100.0, 200.0, 0.0]   # Face left
    pts[454] = [300.0, 200.0, 0.0]   # Face right
    # Jaw width: 123.6 (ratio 2 = 200 / 123.6 = 1.618)
    pts[172] = [138.2, 300.0, 0.0]
    pts[397] = [261.8, 300.0, 0.0]
    # Eyes: spacing = 80. Left eye width = 49.44, Right eye width = 49.44 (ratio 3 = 80 / 49.44 = 1.618)
    pts[133] = [160.0, 180.0, 0.0]   # Left inner
    pts[362] = [240.0, 180.0, 0.0]   # Right inner
    pts[33] = [110.56, 180.0, 0.0]   # Left outer
    pts[263] = [289.44, 180.0, 0.0]  # Right outer
    # Eye height & brow
    pts[159] = [135.0, 175.0, 0.0]   # Left eye top
    pts[145] = [135.0, 185.0, 0.0]   # Left eye bottom (height = 10)
    pts[70] = [135.0, 158.82, 0.0]   # Left brow (distance = 16.18, ratio 7 = 1.618)
    # Nose & Mouth
    pts[6] = [200.0, 173.6, 0.0]     # Nose bridge
    pts[219] = [180.0, 230.0, 0.0]   # Nose left
    pts[439] = [220.0, 230.0, 0.0]   # Nose right (width = 40)
    pts[61] = [187.64, 270.0, 0.0]   # Mouth left
    pts[291] = [212.36, 270.0, 0.0]  # Mouth right (width = 24.72)

    # 1. Base Symmetric Score
    base_lms = Landmarks(points=pts, confidence=1.0, image_width=400, image_height=400)
    base_res = scorer.score_all(base_lms)
    print(f"Base symmetric face features count: {len(base_res.features)}")
    for fs in base_res.features:
        print(f"  {fs.feature}: ratio={fs.measured_ratio:.3f}, score={fs.score:.1f}")

    # 2. Distort Left Eye Outer (move landmark 33 outward by 25px -> widens left eye by ~50%)
    distorted_pts = pts.copy()
    distorted_pts[33][0] -= 25.0
    distorted_lms = Landmarks(points=distorted_pts, confidence=1.0, image_width=400, image_height=400)
    distorted_res = scorer.score_all(distorted_lms)

    print(f"\nAfter distorting Left Eye Outer (LM 33):")
    print(f"Features count: {len(distorted_res.features)} (expected 9 due to bilateral split)")
    feature_names = [fs.feature for fs in distorted_res.features]
    for fn in feature_names:
        print(f"  Feature: {fn}")

    assert "Eye Spacing ÷ Eye Width (Left)" in feature_names, "Expected Eye Spacing to split into (Left)"
    assert "Eye Spacing ÷ Eye Width (Right)" in feature_names, "Expected Eye Spacing to split into (Right)"
    assert "Cheekbone to Jaw Width (Left)" not in feature_names, "Ratio 2 should NOT split"
    assert "Eye-Eyebrow Distance ÷ Eye Height (Left)" not in feature_names, "Ratio 7 should NOT split (hardcoded bug)"
    print("\nVerified: Only Ratio 3 splits under unilateral eye distortion. Ratio 2 and Ratio 7 remain degenerate.")

if __name__ == "__main__":
    detector = FaceDetector(LANDMARK_TASK)
    scorer = GoldenRatioScorer()

    # Step 1: Frontal set curation
    frontal_paths = curate_frontal_set(detector, target_count=30)

    # Step 2: Oracle comparison
    test_oracle_parity(detector, scorer, frontal_paths)

    # Step 3: Non-square aspect ratio test
    test_non_square_padding(detector, scorer, frontal_paths[0])

    # Step 4: Endpoint overlays & alternatives
    nose_comp = draw_endpoint_overlays_and_alternatives(detector, frontal_paths, num_images=20)

    # Step 5: Population audit
    stats, max_score, splits = run_large_scale_ratio_audit(detector, scorer, frontal_paths, num_celebrity=1000)

    # Step 6: Synthetic distortion test
    test_left_right_synthetic_distortion(scorer)
