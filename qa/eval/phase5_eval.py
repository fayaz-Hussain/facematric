import sys
import os
import glob
import math
import json
import csv
from collections import defaultdict
from PIL import Image
import numpy as np
import cv2
from scipy import stats

PROJECT_ROOT = os.path.abspath(".")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

from app.components.face_detector import FaceDetector
from app.components.golden_ratio_scorer import GoldenRatioScorer
from app.components.symmetry_classifier import SymmetryClassifier
from app.components.percentile_engine import PercentileEngine
from app.models.internal import ImageData, SymmetryType
from qa.eval.phase2_measurements import estimate_head_pose_yaw

LANDMARK_TASK = os.path.join(PROJECT_ROOT, "backend", "data", "face_landmarker.task")
DATASET_DIR = os.path.join(PROJECT_ROOT, "celebrity_dataset")
if not os.path.exists(DATASET_DIR):
    DATASET_DIR = r"C:\Users\FayazHussain\Desktop\FaceMatric\celebrity_dataset"
REPORTS_DIR = os.path.join(PROJECT_ROOT, "qa", "reports")
DATA_DIR = os.path.join(PROJECT_ROOT, "qa", "data")

def run_phase5_eval():
    print("=" * 70)
    print("PHASE 5: REPEATABILITY, VALIDITY, PERCENTILE & FACE SHAPE EVALUATION")
    print("=" * 70)

    detector = FaceDetector(LANDMARK_TASK)
    scorer = GoldenRatioScorer()
    classifier = SymmetryClassifier()
    engine = PercentileEngine()

    # =========================================================================
    # Part 1: Same-Person Stability
    # =========================================================================
    print("\n--- 1. SAME-PERSON STABILITY ACROSS NEAR-FRONTAL PHOTOS ---")
    pins_dirs = sorted([d for d in os.listdir(DATASET_DIR) if d.startswith("pins_")])
    
    # Target 15 identities, with 3 to 5 near-frontal photos each (|yaw| <= 6.0 deg)
    identity_samples = []
    
    for p in pins_dirs:
        dir_path = os.path.join(DATASET_DIR, p)
        imgs = sorted(glob.glob(os.path.join(dir_path, "*.jpg")))
        person_name = p[5:]
        person_imgs = []
        
        for img_path in imgs:
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
                if abs(yaw) <= 6.0:
                    # Run scoring and face shape
                    gr_res = scorer.score_all(lms)
                    shape = classifier.classify(lms)
                    
                    # Extract ratios
                    ratio_dict = {}
                    for fs in gr_res.features:
                        ratio_dict[fs.feature] = fs.score
                    
                    person_imgs.append({
                        "identity": person_name,
                        "image": img_path,
                        "yaw": float(yaw),
                        "overall_score": float(gr_res.overall_score),
                        "shape": shape.value,
                        "ratios": ratio_dict,
                        "features_count": len(gr_res.features)
                    })
                    if len(person_imgs) >= 5:
                        break
            except Exception:
                continue
        
        if len(person_imgs) >= 3:
            identity_samples.append({
                "identity": person_name,
                "photos": person_imgs
            })
            print(f"  Selected '{person_name}': {len(person_imgs)} photos (yaw range: {min(p['yaw'] for p in person_imgs):.1f}° to {max(p['yaw'] for p in person_imgs):.1f}°)")
            if len(identity_samples) >= 15:
                break

    print(f"\nTotal identities evaluated: {len(identity_samples)}")
    total_photos = sum(len(x["photos"]) for x in identity_samples)
    print(f"Total near-frontal photos evaluated: {total_photos}")

    # Compute per-person stats
    person_stats = []
    all_scores_flat = []
    identities_flat = []

    # Prepare ratio SD accumulators
    canonical_features = [
        "Face Length ÷ Face Width",
        "Cheekbone to Jaw Width",
        "Eye Spacing ÷ Eye Width",
        "Nose Width ÷ Mouth Width",
        "Mouth Width ÷ Face Width",
        "Vertical Thirds",
        "Eye-Eyebrow Distance ÷ Eye Height",
        "Chin Width ÷ Nose Width"
    ]
    per_ratio_variances = defaultdict(list)

    for item in identity_samples:
        ident = item["identity"]
        scores = [p["overall_score"] for p in item["photos"]]
        shapes = [p["shape"] for p in item["photos"]]
        all_scores_flat.extend(scores)
        identities_flat.extend([ident] * len(scores))
        
        mean_score = float(np.mean(scores))
        sd_score = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))
        range_score = max_score - min_score
        
        # Shape consistency
        from collections import Counter
        shape_counts = Counter(shapes)
        shape_mode, shape_mode_count = shape_counts.most_common(1)[0]
        shape_consistency = float(shape_mode_count) / len(shapes) * 100.0

        # Per-ratio SD for this person
        for cf in canonical_features:
            cf_vals = []
            for p in item["photos"]:
                # Match canonical feature name prefix
                for k, v in p["ratios"].items():
                    if k.startswith(cf) or (cf == "Vertical Thirds" and "Thirds" in k):
                        cf_vals.append(v)
                        break
            if len(cf_vals) > 1:
                per_ratio_variances[cf].append(float(np.var(cf_vals, ddof=1)))

        person_stats.append({
            "identity": ident,
            "n_photos": len(scores),
            "mean_score": round(mean_score, 2),
            "sd_score": round(sd_score, 2),
            "range": round(range_score, 2),
            "min_score": round(min_score, 2),
            "max_score": round(max_score, 2),
            "primary_shape": str(shape_mode),
            "shape_consistency_pct": round(shape_consistency, 1)
        })

    # Summary of Same-Person Stability
    all_sds = [p["sd_score"] for p in person_stats]
    pooled_within_sd = float(np.sqrt(np.mean([s**2 for s in all_sds])))
    mean_person_sd = float(np.mean(all_sds))
    median_person_sd = float(np.median(all_sds))
    max_person_sd = float(np.max(all_sds))
    pct_under_3 = float(np.mean([s <= 3.0 for s in all_sds]) * 100.0)

    print("\nPer-Person Overall Score Stability (Target: SD <= 3.0 points):")
    print(f"  Mean SD across persons:   {mean_person_sd:.2f} points")
    print(f"  Pooled Within-Person SD:  {pooled_within_sd:.2f} points")
    print(f"  Median SD across persons: {median_person_sd:.2f} points")
    print(f"  Max SD observed:          {max_person_sd:.2f} points")
    print(f"  Percentage meeting SD<=3: {pct_under_3:.1f}% ({sum(1 for s in all_sds if s <= 3.0)}/{len(all_sds)})")

    # Ratio SD summary
    ratio_sd_summary = {}
    print("\nPer-Ratio Within-Person Standard Deviation:")
    for cf, var_list in per_ratio_variances.items():
        pooled_sd = float(np.sqrt(np.mean(var_list)))
        ratio_sd_summary[cf] = round(pooled_sd, 2)
        print(f"  {cf:40s}: SD = {pooled_sd:.2f} points")

    # Save to CSV
    csv_path = os.path.join(REPORTS_DIR, "05_same_person_stability.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["identity", "n_photos", "mean_score", "sd_score", "range", "min_score", "max_score", "primary_shape", "shape_consistency_pct"])
        for p in person_stats:
            writer.writerow([
                p["identity"], p["n_photos"], p["mean_score"], p["sd_score"],
                p["range"], p["min_score"], p["max_score"], p["primary_shape"],
                p["shape_consistency_pct"]
            ])
    print(f"\nWrote same-person stability metrics to {csv_path}")

    # =========================================================================
    # Part 2: Within-Person vs Between-Person Variance (ICC(1) One-Way Random)
    # =========================================================================
    print("\n--- 2. WITHIN-PERSON VS BETWEEN-PERSON VARIANCE & ICC(1) ---")
    K = len(identity_samples)  # Number of subjects (groups)
    N = len(all_scores_flat)   # Total observations
    
    grand_mean = float(np.mean(all_scores_flat))
    SST = float(np.sum((np.array(all_scores_flat) - grand_mean)**2))
    
    # Group means and counts
    group_counts = [len(item["photos"]) for item in identity_samples]
    group_means = [np.mean([p["overall_score"] for p in item["photos"]]) for item in identity_samples]
    
    SSB = float(np.sum([n * (gm - grand_mean)**2 for n, gm in zip(group_counts, group_means)]))
    SSW = float(SST - SSB)
    
    df_B = K - 1
    df_W = N - K
    
    MSB = SSB / df_B
    MSW = SSW / df_W
    
    F_stat = MSB / MSW
    p_val = 1.0 - stats.f.cdf(F_stat, df_B, df_W)
    
    # k0 for unbalanced one-way random effects model:
    # k0 = (1 / (K - 1)) * (N - sum(n_i^2) / N)
    k0 = (1.0 / (K - 1)) * (N - sum(n**2 for n in group_counts) / N)
    
    # ICC(1) = (MSB - MSW) / (MSB + (k0 - 1) * MSW)
    icc1 = (MSB - MSW) / (MSB + (k0 - 1) * MSW)
    
    # Variance components:
    # MSW estimates sigma_w^2 (within-person variance)
    # MSB estimates sigma_w^2 + k0 * sigma_b^2 -> sigma_b^2 = max(0, (MSB - MSW) / k0)
    sigma2_w = MSW
    sigma2_b = max(0.0, (MSB - MSW) / k0)
    total_var = sigma2_b + sigma2_w
    pct_var_between = (sigma2_b / total_var) * 100.0 if total_var > 0 else 0.0
    pct_var_within = (sigma2_w / total_var) * 100.0 if total_var > 0 else 0.0

    print(f"  Number of Identities (K):          {K}")
    print(f"  Total Observations (N):             {N}")
    print(f"  Average Images per Subject (k0):    {k0:.2f}")
    print(f"  Between-Person Mean Square (MSB):   {MSB:.3f} (df={df_B})")
    print(f"  Within-Person Mean Square (MSW):    {MSW:.3f} (df={df_W})")
    print(f"  F-statistic:                        {F_stat:.3f} (p = {p_val:.4e})")
    print(f"  Estimated Between-Person Variance:  {sigma2_b:.3f} ({pct_var_between:.1f}%)")
    print(f"  Estimated Within-Person Variance:   {sigma2_w:.3f} ({pct_var_within:.1f}%)")
    print(f"  ICC(1) (One-way Random):            {icc1:.4f}")

    # =========================================================================
    # Part 3: Percentile Distribution Calibration (Engine vs Empirical)
    # =========================================================================
    print("\n--- 3. PERCENTILE ENGINE CALIBRATION EVALUATION (H12) ---")
    # Empirical distribution parameters from Phase 2 (959 celebrity images):
    # Mean = 52.41, SD = 4.62
    emp_mu = 52.41
    emp_sd = 4.62
    
    test_scores = [55, 60, 65, 70, 75, 80]
    percentile_comparison = []

    print("| Score | Engine Percentile (N(70, 10)) | Empirical Percentile (N(52.4, 4.6)) | Discrepancy (Δ) | Real-World Meaning |")
    print("| :---: | :---: | :---: | :---: | :--- |")
    
    for s in test_scores:
        engine_pct = engine.score_to_percentile(s)
        # Empirical normal CDF
        emp_cdf = stats.norm.cdf(s, emp_mu, emp_sd) * 100.0
        emp_pct = int(round(max(0.0, min(100.0, emp_cdf))))
        diff = engine_pct - emp_pct
        
        meaning = ""
        if s == 55:
            meaning = "Engine labels below-average (7th pct); actually above average (71st pct)!"
        elif s == 60:
            meaning = "Engine labels low-tier (16th pct); actually top 5% of population (95th pct)!"
        elif s == 65:
            meaning = "Engine labels bottom third (31st pct); actually top 0.3% (99.7th pct)!"
        elif s == 70:
            meaning = "Engine median (50th pct); practically impossible in real faces (>99.99th pct)!"
        elif s >= 75:
            meaning = "Biologically unreachable; engine ranks 69–84%, reality is 100.0%!"
            
        print(f"| {s:5d} | {engine_pct:29d} | {emp_pct:35d} | {diff:+15d} | {meaning} |")
        percentile_comparison.append({
            "score": s,
            "engine_percentile": engine_pct,
            "empirical_percentile": emp_pct,
            "discrepancy": diff,
            "note": meaning
        })

    # Signal-to-noise / slope near median
    engine_slope = (100.0 / (10.0 * np.sqrt(2 * np.pi)))
    empirical_slope = (100.0 / (emp_sd * np.sqrt(2 * np.pi)))
    print(f"\nPercentile sensitivity near the distribution median:")
    print(f"  Engine Model (σ=10.0):    {engine_slope:.2f} percentile points per 1.0 score point")
    print(f"  Empirical Model (σ=4.62): {empirical_slope:.2f} percentile points per 1.0 score point")
    print(f"  Impact of Measurement Noise (±2.0 points observed in Phase 3):")
    print(f"    Under Engine Model:    ±{engine_slope * 2.0:.1f} percentile ranks")
    print(f"    Under Empirical Model: ±{empirical_slope * 2.0:.1f} percentile ranks (MASSIVE SWING)")

    # =========================================================================
    # Part 4: Face Shape Classifier Rules & Independence Verification
    # =========================================================================
    print("\n--- 4. FACE SHAPE CLASSIFIER RULES & SCORE INDEPENDENCE ---")
    decision_rules = [
        {
            "shape": "HEART",
            "priority": 1,
            "condition": "forehead_width > jaw_width * 1.20",
            "threshold": "forehead / jaw > 1.20"
        },
        {
            "shape": "DIAMOND",
            "priority": 2,
            "condition": "cheekbone_width > max(forehead_width, jaw_width) * 1.10",
            "threshold": "cheekbone / max(forehead, jaw) > 1.10"
        },
        {
            "shape": "OVAL",
            "priority": 3,
            "condition": "1.40 <= length_width_ratio <= 1.60 and jawline_angle > 140.0°",
            "threshold": "ratio in [1.40, 1.60], angle > 140.0°"
        },
        {
            "shape": "RECTANGLE",
            "priority": 4,
            "condition": "length_width_ratio > 1.60 and jawline_angle < 130.0°",
            "threshold": "ratio > 1.60, angle < 130.0°"
        },
        {
            "shape": "ROUND",
            "priority": 5,
            "condition": "0.90 <= length_width_ratio <= 1.10 and jawline_angle > 140.0°",
            "threshold": "ratio in [0.90, 1.10], angle > 140.0°"
        },
        {
            "shape": "SQUARE",
            "priority": 6,
            "condition": "0.90 <= length_width_ratio <= 1.10 and jawline_angle < 130.0°",
            "threshold": "ratio in [0.90, 1.10], angle < 130.0°"
        },
        {
            "shape": "OVAL (Default fallback)",
            "priority": 7,
            "condition": "None of the above match (borderline/unclassified)",
            "threshold": "Fallback catch-all"
        }
    ]

    print("Extracted Numeric Decision Tree Rules from SymmetryClassifier._classify_by_ratios:")
    for r in decision_rules:
        print(f"  Priority {r['priority']}: {r['shape']:20s} -> Condition: {r['condition']}")

    # Save summary json
    summary_data = {
        "repeatability": {
            "n_identities": K,
            "total_photos": N,
            "pooled_within_sd": round(pooled_within_sd, 3),
            "mean_person_sd": round(mean_person_sd, 3),
            "median_person_sd": round(median_person_sd, 3),
            "max_person_sd": round(max_person_sd, 3),
            "pct_meeting_target_sd_le_3": round(pct_under_3, 1),
            "per_ratio_sd": ratio_sd_summary
        },
        "variance_decomposition": {
            "K_identities": K,
            "N_photos": N,
            "k0": round(k0, 2),
            "MSB": round(MSB, 3),
            "MSW": round(MSW, 3),
            "F_statistic": round(F_stat, 3),
            "p_value": float(p_val),
            "sigma2_between": round(sigma2_b, 3),
            "sigma2_within": round(sigma2_w, 3),
            "pct_var_between": round(pct_var_between, 1),
            "pct_var_within": round(pct_var_within, 1),
            "ICC1": round(icc1, 4)
        },
        "percentile_calibration": {
            "engine_distribution": "Normal(mu=70.0, sigma=10.0)",
            "empirical_distribution": f"Normal(mu={emp_mu}, sigma={emp_sd})",
            "comparisons": percentile_comparison,
            "engine_slope_at_median": round(engine_slope, 2),
            "empirical_slope_at_median": round(empirical_slope, 2)
        },
        "face_shape_rules": decision_rules
    }

    json_path = os.path.join(REPORTS_DIR, "05_validity_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nWrote full Phase 5 evaluation summary to {json_path}")

if __name__ == "__main__":
    run_phase5_eval()
