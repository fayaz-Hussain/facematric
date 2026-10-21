"""
build_score_reference.py

Generates backend/data/score_reference.json from the FairFace dataset.
Implements Batch 7 Part 4 requirements:
- FairFace (CC BY 4.0), config 1.25
- Stratified adult sampling across 14 race x gender cells (700 images per cell = 9,800 total)
- Production quality gates (yaw <= 12 deg, roll <= 10 deg, occlusion, min dimension 300px bicubic)
- Production GoldenRatioScorer with stabilized ratios R7 and R8
- Equal race x gender cell weighting for pooled quantiles
- Saves 101 quantiles, metadata, and within-person SD (1.26)
"""

import os
import sys
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure backend directory is in path and working dir
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(str(BACKEND_DIR))

from datasets import load_dataset
from app.components.face_detector import FaceDetector
from app.components.golden_ratio_scorer import GoldenRatioScorer
from app.models.internal import ImageData
from app.utils.errors import ValidationError


def run():
    print("=" * 70)
    print("BUILD SCORE REFERENCE: FAIRFACE STRATIFIED POPULATION REFERENCE")
    print("=" * 70)
    start_time = time.time()
    
    SEED = 42
    TARGET_PER_CELL = 700
    WITHIN_PERSON_SD = 1.26
    SCORER_COMMIT = "3a1cb063e6dd742d65e21e98187d256e14863e36"
    
    random.seed(SEED)
    np.random.seed(SEED)
    
    target_dir = str(REPO_ROOT / "qa" / "data" / "fairface")
    print(f"Loading FairFace dataset (config 1.25) from {target_dir}...")
    ds = load_dataset("HuggingFaceM4/FairFace", "1.25", cache_dir=target_dir, split="train")
    print(f"Loaded {len(ds)} total examples in train split.")
    
    races = ds.features["race"].names
    genders = ds.features["gender"].names
    ages = ds.features["age"].names
    print(f"Races ({len(races)}): {races}")
    print(f"Genders ({len(genders)}): {genders}")
    print(f"Ages ({len(ages)}): {ages}")
    
    adult_age_names = {"20-29", "30-39", "40-49", "50-59"}
    adult_age_ids = {i for i, name in enumerate(ages) if name in adult_age_names}
    print(f"Filtering for adults only: {adult_age_names} (IDs: {adult_age_ids})")
    
    # Fast metadata pass
    t0 = time.time()
    metadata_ds = ds.select_columns(["race", "gender", "age"])
    cell_indices = defaultdict(list)
    for idx in range(len(metadata_ds)):
        row = metadata_ds[idx]
        age_val = row["age"]
        if age_val in adult_age_ids or age_val in adult_age_names:
            r = row["race"]
            g = row["gender"]
            r_name = races[r] if isinstance(r, int) else r
            g_name = genders[g] if isinstance(g, int) else g
            cell_indices[(r_name, g_name)].append(idx)
    
    print(f"Grouped adult indices across {len(cell_indices)} cells in {time.time() - t0:.2f}s.")
    
    # Stratified sampling of TARGET_PER_CELL per cell
    sampled_indices = []
    cell_sampled_counts = {}
    for cell, idx_list in sorted(cell_indices.items()):
        if len(idx_list) < TARGET_PER_CELL:
            raise ValueError(f"Cell {cell} has only {len(idx_list)} images, less than target {TARGET_PER_CELL}")
        chosen = random.sample(idx_list, TARGET_PER_CELL)
        cell_sampled_counts[f"{cell[0]}_{cell[1]}"] = len(chosen)
        for idx in chosen:
            sampled_indices.append((idx, cell[0], cell[1]))
    
    print(f"Total sampled: {len(sampled_indices)} images ({TARGET_PER_CELL} per cell x 14 cells).")
    
    model_path = str(BACKEND_DIR / "data" / "face_landmarker.task")
    detector = FaceDetector(model_path=model_path)
    scorer = GoldenRatioScorer()
    
    # Process images through production pipeline
    print("\nProcessing images through production pipeline (ImageProcessor/FaceDetector/PoseGate/Scorer)...")
    accepted_by_cell = defaultdict(list)
    rejection_reasons_by_cell = defaultdict(lambda: Counter())
    all_rejections = Counter()
    
    t_proc = time.time()
    for count, (idx, race_name, gender_name) in enumerate(sampled_indices, 1):
        item = ds[idx]
        pil_img = item["image"]
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
            
        w, h = pil_img.size
        # Upscale to at least 300 px on the short side (bicubic)
        scale = 300.0 / min(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        pil_img = pil_img.resize((new_w, new_h), Image.BICUBIC)
        
        img_arr = np.array(pil_img, dtype=np.uint8)
        img_data = ImageData(array=img_arr, width=new_w, height=new_h, original_format="JPEG")
        
        cell_key = (race_name, gender_name)
        try:
            landmarks = detector.detect_faces(img_data)
            res = scorer.score_all(landmarks)
            score = float(res.overall_score)
            accepted_by_cell[cell_key].append(score)
        except Exception as e:
            msg = str(e)
            reason = "other"
            if "Head pose exceeds threshold" in msg or "pose" in msg.lower():
                reason = "pose_limit_exceeded"
            elif "occlusion" in msg.lower() or "eye aspect ratio" in msg.lower():
                reason = "occlusion_detected"
            elif "couldn't detect a face" in msg or "no face" in msg.lower():
                reason = "no_face_detected"
            elif "Multiple faces" in msg:
                reason = "multiple_faces"
            elif "quality is too low" in msg:
                reason = "low_confidence"
            rejection_reasons_by_cell[cell_key][reason] += 1
            all_rejections[reason] += 1
            
        if count % 1000 == 0 or count == len(sampled_indices):
            elapsed = time.time() - t_proc
            rate = count / elapsed
            accepted_so_far = sum(len(v) for v in accepted_by_cell.values())
            print(f"Processed {count}/{len(sampled_indices)} ({count/len(sampled_indices)*100:.1f}%) - {rate:.1f} img/s - Accepted: {accepted_so_far}")

    # Summary and reporting
    print("\n" + "=" * 70)
    print("ACCEPTANCE RATE REPORT BY RACE AND GENDER")
    print("=" * 70)
    print(f"{'Race':<18} | {'Gender':<8} | {'Sampled':<7} | {'Accepted':<8} | {'Rate':<7} | Primary Rejection")
    print("-" * 75)
    
    acceptance_by_cell_dict = {}
    for cell, scores in sorted(accepted_by_cell.items()):
        n_acc = len(scores)
        rate = (n_acc / TARGET_PER_CELL) * 100.0
        top_rejection = rejection_reasons_by_cell[cell].most_common(1)
        top_rej_str = f"{top_rejection[0][0]} ({top_rejection[0][1]})" if top_rejection else "none"
        print(f"{cell[0]:<18} | {cell[1]:<8} | {TARGET_PER_CELL:<7} | {n_acc:<8} | {rate:5.1f}% | {top_rej_str}")
        acceptance_by_cell_dict[f"{cell[0]}_{cell[1]}"] = {
            "race": cell[0],
            "gender": cell[1],
            "sampled": TARGET_PER_CELL,
            "accepted": n_acc,
            "acceptance_rate": round(rate, 2),
            "rejections": dict(rejection_reasons_by_cell[cell])
        }

    total_accepted = sum(len(v) for v in accepted_by_cell.values())
    total_sampled = len(sampled_indices)
    overall_acc_rate = (total_accepted / total_sampled) * 100.0
    print("-" * 75)
    print(f"TOTAL: {total_accepted}/{total_sampled} accepted ({overall_acc_rate:.2f}%)")
    print(f"Overall Rejection Breakdown: {dict(all_rejections)}")

    # Distribution analysis
    print("\n" + "=" * 70)
    print("SCORE DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    # By Race
    by_race = defaultdict(list)
    for (r, g), scores in accepted_by_cell.items():
        by_race[r].extend(scores)
        
    print("\nScore Distribution by Race:")
    print(f"{'Race':<18} | {'N':<6} | {'Mean':<7} | {'SD':<6} | {'Min':<6} | {'Max':<6}")
    print("-" * 60)
    dist_by_race = {}
    for r, scores in sorted(by_race.items()):
        m, s = float(np.mean(scores)), float(np.std(scores, ddof=1))
        print(f"{r:<18} | {len(scores):<6} | {m:6.2f}  | {s:5.2f} | {min(scores):5.1f}  | {max(scores):5.1f}")
        dist_by_race[r] = {"n": len(scores), "mean": round(m, 3), "std": round(s, 3)}

    # By Gender
    by_gender = defaultdict(list)
    for (r, g), scores in accepted_by_cell.items():
        by_gender[g].extend(scores)
        
    print("\nScore Distribution by Gender:")
    print(f"{'Gender':<18} | {'N':<6} | {'Mean':<7} | {'SD':<6} | {'Min':<6} | {'Max':<6}")
    print("-" * 60)
    dist_by_gender = {}
    for g, scores in sorted(by_gender.items()):
        m, s = float(np.mean(scores)), float(np.std(scores, ddof=1))
        print(f"{g:<18} | {len(scores):<6} | {m:6.2f}  | {s:5.2f} | {min(scores):5.1f}  | {max(scores):5.1f}")
        dist_by_gender[g] = {"n": len(scores), "mean": round(m, 3), "std": round(s, 3)}

    # Pooled distribution (equal race x gender cell weighting = 1/14 per cell)
    cell_means = [np.mean(scores) for scores in accepted_by_cell.values()]
    cell_stds = [np.std(scores, ddof=1) for scores in accepted_by_cell.values()]
    
    pooled_mean = float(np.mean(cell_means))
    # Pooled variance across equal-weighted cells: mean(var) + var(means)
    pooled_var = float(np.mean([s**2 for s in cell_stds]) + np.var(cell_means))
    pooled_std = float(np.sqrt(pooled_var))
    
    print(f"\nPooled Distribution (Equal Race x Gender Cell Weighting):")
    print(f"  Pooled Mean: {pooled_mean:.3f}")
    print(f"  Pooled SD:   {pooled_std:.3f}")
    
    # Check if any group mean differs by more than 1 pooled SD
    max_diff_race = max(abs(d["mean"] - pooled_mean) for d in dist_by_race.values())
    max_diff_gender = max(abs(d["mean"] - pooled_mean) for d in dist_by_gender.values())
    max_diff_cell = max(abs(np.mean(scores) - pooled_mean) for scores in accepted_by_cell.values())
    
    print(f"\nGroup Difference Audit (Threshold = 1 Pooled SD = {pooled_std:.3f}):")
    print(f"  Max Race Mean Difference:   {max_diff_race:.3f} ({max_diff_race / pooled_std:.2f} SD)")
    print(f"  Max Gender Mean Difference: {max_diff_gender:.3f} ({max_diff_gender / pooled_std:.2f} SD)")
    print(f"  Max Cell Mean Difference:   {max_diff_cell:.3f} ({max_diff_cell / pooled_std:.2f} SD)")
    if max(max_diff_race, max_diff_gender, max_diff_cell) > pooled_std:
        print("  WARNING: A group mean differs by more than one pooled SD!")
    else:
        print("  PASS: All group means are within one pooled SD of the population mean.")

    # Compute 101 quantiles (0 to 100) under equal race x gender cell weighting
    # Draw balanced resample of 10,000 scores per cell (140,000 total) to produce strictly monotonic empirical quantiles
    balanced_pool = []
    for cell, scores in accepted_by_cell.items():
        resampled = np.random.choice(scores, size=10000, replace=True)
        balanced_pool.extend(resampled)
    balanced_pool = np.array(balanced_pool)
    
    quantiles_101 = [float(round(q, 3)) for q in np.percentile(balanced_pool, np.linspace(0, 100, 101))]
    
    # Ensure strictly monotonic non-decreasing
    for i in range(1, len(quantiles_101)):
        if quantiles_101[i] < quantiles_101[i - 1]:
            quantiles_101[i] = quantiles_101[i - 1]

    # Reference data payload
    reference_data = {
        "dataset_name": "FairFace",
        "dataset_config": "1.25",
        "license": "CC BY 4.0",
        "citation": "Karkkainen, K., & Joo, J. (2021). FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation. WACV 2021.",
        "filters": "Adults (ages 20-59), frontal pose (|yaw| <= 12 deg, |roll| <= 10 deg), no severe occlusion, upscaled bicubic to >= 300px",
        "seed": SEED,
        "scorer_commit": SCORER_COMMIT,
        "within_person_sd": WITHIN_PERSON_SD,
        "n_sampled_total": total_sampled,
        "n_accepted_total": total_accepted,
        "overall_acceptance_rate": round(overall_acc_rate, 2),
        "pooled_mean": round(pooled_mean, 3),
        "pooled_std": round(pooled_std, 3),
        "quantiles_0_to_100": quantiles_101,
        "acceptance_by_cell": acceptance_by_cell_dict,
        "distribution_by_race": dist_by_race,
        "distribution_by_gender": dist_by_gender
    }
    
    output_path = BACKEND_DIR / "data" / "score_reference.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(reference_data, f, indent=2)
    print(f"\nSaved score reference to {output_path}")

    # Diagnostics for Part 4i
    print("\n" + "=" * 70)
    print("PART 4i: DIAGNOSTIC PERCENTILE EVALUATION")
    print("=" * 70)
    
    # Percentile lookup function using empirical CDF
    def score_to_pct(s):
        # np.searchsorted or interpolated
        return float(np.interp(s, quantiles_101, np.linspace(0, 100, 101)))
    
    celeb_mean_959 = 52.41
    celeb_pct = score_to_pct(celeb_mean_959)
    print(f"1. 959 Celebrity Photos Mean Score (52.41):")
    print(f"   Percentile in new FairFace distribution: {celeb_pct:.2f}th percentile")
    
    mean_score = pooled_mean
    m_minus_1sd = pooled_mean - pooled_std
    m_plus_1sd = pooled_mean + pooled_std
    
    pct_m_minus = score_to_pct(m_minus_1sd)
    pct_mean = score_to_pct(mean_score)
    pct_m_plus = score_to_pct(m_plus_1sd)
    
    print(f"\n2. Reference Distribution Key Anchors:")
    print(f"   Mean - 1 SD ({m_minus_1sd:.2f}): {pct_m_minus:.2f}th percentile")
    print(f"   Mean        ({mean_score:.2f}): {pct_mean:.2f}th percentile")
    print(f"   Mean + 1 SD ({m_plus_1sd:.2f}): {pct_m_plus:.2f}th percentile")
    
    median_score = quantiles_101[50]
    p_med_minus_half = score_to_pct(median_score - 0.5)
    p_med_plus_half = score_to_pct(median_score + 0.5)
    slope_at_median = p_med_plus_half - p_med_minus_half
    print(f"\n3. Sensitivity Near Median (Score = {median_score:.2f}):")
    print(f"   Slope: {slope_at_median:.2f} percentile points per 1.0 score point")
    
    total_elapsed = time.time() - start_time
    print(f"\nBuild score reference completed in {total_elapsed / 60:.2f} minutes.")


if __name__ == "__main__":
    run()
