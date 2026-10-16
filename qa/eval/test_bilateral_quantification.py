import sys
import os
import numpy as np
from PIL import Image as PILImage
import io

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure backend root is on sys.path
backend_path = os.path.abspath("backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from app.components.golden_ratio_scorer import GoldenRatioScorer
from app.components.face_detector import FaceDetector
from app.models.internal import Landmarks

scorer = GoldenRatioScorer()

def build_base_landmarks():
    points = np.zeros((468, 3), dtype=np.float32)
    # Face boundaries
    points[10] = [320.0, 60.0, 0.0]     # Top
    points[152] = [320.0, 340.0, 0.0]   # Bottom (length = 280)
    points[234] = [150.0, 200.0, 0.0]   # Left (width = 340)
    points[454] = [490.0, 200.0, 0.0]   # Right

    # Jaw
    points[172] = [180.0, 300.0, 0.0]   # Left jaw
    points[397] = [460.0, 300.0, 0.0]   # Right jaw

    # Eyes: baseline eye_spacing = dist(133, 362) = 390 - 250 = 140
    # Left eye width: dist(33, 133) = 250 - 200 = 50
    # Right eye width: dist(362, 263) = 440 - 390 = 50
    # Baseline ratio_left = 140/50 = 2.8, ratio_right = 140/50 = 2.8 (symmetric)
    points[33] = [200.0, 180.0, 0.0]    # Left outer
    points[133] = [250.0, 180.0, 0.0]   # Left inner
    points[362] = [390.0, 180.0, 0.0]   # Right inner
    points[263] = [440.0, 180.0, 0.0]   # Right outer
    points[159] = [225.0, 170.0, 0.0]   # Left top
    points[145] = [225.0, 190.0, 0.0]   # Left bottom
    points[386] = [415.0, 170.0, 0.0]   # Right top
    points[374] = [415.0, 190.0, 0.0]   # Right bottom

    # Eyebrows
    points[70] = [225.0, 150.0, 0.0]    # Left eyebrow bottom
    points[300] = [415.0, 150.0, 0.0]   # Right eyebrow bottom

    # Nose
    points[219] = [290.0, 220.0, 0.0]   # Left
    points[439] = [350.0, 220.0, 0.0]   # Right
    points[6] = [320.0, 190.0, 0.0]     # Bridge

    # Mouth
    points[61] = [270.0, 260.0, 0.0]    # Left
    points[291] = [370.0, 260.0, 0.0]   # Right
    return points

def run_bilateral_split_experiment():
    print("=== PART 1: BILATERAL SPLIT RULE QUANTIFICATION (RATIO 3) ===")
    
    # In GoldenRatioScorer, Ratio 3:
    # eye_spacing = dist(133, 362) = 140
    # left_eye_width = dist(33, 133)
    # right_eye_width = dist(263, 362)
    # ratio_left = eye_spacing / left_eye_width
    # ratio_right = eye_spacing / right_eye_width
    # ratio_avg = (ratio_left + ratio_right) / 2
    # asymmetry_pct = |ratio_left - ratio_right| / ratio_avg
    
    # Let average ratio be R_avg = 1.618 (phi!)
    # eye_spacing = 140 -> average eye width W_avg = 140 / 1.618 ≈ 86.526576
    # To get asymmetry_pct = delta (0.09, 0.10, 0.11):
    # ratio_left = R_avg * (1 - delta/2)
    # ratio_right = R_avg * (1 + delta/2)
    # left_width = eye_spacing / ratio_left
    # right_width = eye_spacing / ratio_right
    
    target_deltas = [0.090, 0.100, 0.110]
    base_pts = build_base_landmarks()
    
    # Set base eye spacing to 140 (x=250 to x=390)
    eye_spacing = 140.0
    r_avg = 1.618
    
    results = []
    
    for delta in target_deltas:
        pts = base_pts.copy()
        r_left = r_avg * (1.0 - delta / 2.0)
        r_right = r_avg * (1.0 + delta / 2.0)
        
        w_left = eye_spacing / r_left
        w_right = eye_spacing / r_right
        
        # Adjust left eye outer (33) and right eye outer (263)
        pts[33, 0] = pts[133, 0] - w_left
        pts[263, 0] = pts[362, 0] + w_right
        
        lm = Landmarks(points=pts, confidence=1.0, image_width=640, image_height=480)
        res = scorer.score_all(lm)
        
        # Verify measured asymmetry
        r3_left, r3_right = scorer._measure_eye_spacing_to_width_bilateral(pts)
        meas_asym = abs(r3_left - r3_right) / ((r3_left + r3_right) / 2.0)
        
        entry_count = len(res.features)
        results.append({
            "target_delta": delta * 100,
            "measured_asym": meas_asym * 100,
            "entry_count": entry_count,
            "overall_score": res.overall_score,
            "features": [(f.feature, f.score, f.measured_ratio) for f in res.features]
        })
        
        print(f"\nTarget Delta: {delta*100:.1f}%")
        print(f"  Measured Asymmetry: {meas_asym*100:.2f}%")
        print(f"  Feature Count in Breakdown: {entry_count}")
        print(f"  Overall Score: {res.overall_score:.2f}")
        for f in res.features:
            if "Eye Spacing" in f.feature:
                print(f"    -> {f.feature}: score={f.score}, ratio={f.measured_ratio}")
                
    # Compare 9% vs 10% vs 11%
    score_9 = results[0]["overall_score"]
    score_10 = results[1]["overall_score"]
    score_11 = results[2]["overall_score"]
    jump = score_11 - score_10
    print(f"\n=== DISCONTINUITY ANALYSIS ===")
    print(f"Score at 9%: {score_9:.2f} (8 entries)")
    print(f"Score at 10%: {score_10:.2f} (8 entries)")
    print(f"Score at 11%: {score_11:.2f} (9 entries)")
    print(f"Discontinuity jump across 10% threshold: {jump:+.2f} points ({jump/score_10*100:+.2f}%)")
    return results

def run_celebrity_feature_audit():
    print("\n=== PART 2: REAL CELEBRITY IMAGES FEATURE COUNT AUDIT ===")
    
    images = [
        "celebrity_dataset/pins_Adriana Lima/Adriana Lima0_0.jpg",
        "celebrity_dataset/pins_Adriana Lima/Adriana Lima101_3.jpg",
        "celebrity_dataset/pins_Adriana Lima/Adriana Lima102_4.jpg",
    ]
    
    model_path = os.path.abspath("backend/data/face_landmarker.task")
    detector = FaceDetector(model_path=model_path)
    
    celeb_results = []
    
    for img_path in images:
        full_path = os.path.abspath(img_path)
        img_pil = PILImage.open(full_path).convert("RGB")
        img_arr = np.array(img_pil)
        
        from app.models.internal import ImageData
        img_data = ImageData(array=img_arr, width=img_pil.width, height=img_pil.height, original_format="JPEG")
        lm = detector.detect_faces(img_data)
        res = scorer.score_all(lm)
        
        entry_count = len(res.features)
        split_entries = [f for f in res.features if "(Left)" in f.feature or "(Right)" in f.feature]
        
        print(f"\nImage: {os.path.basename(img_path)}")
        print(f"  Dimensions: {img_pil.size}")
        print(f"  Overall Score: {res.overall_score:.1f}")
        print(f"  Total Breakdown Entries Averaged: {entry_count}")
        if split_entries:
            print(f"  Split Features ({len(split_entries)} entries):")
            for sf in split_entries:
                print(f"    * {sf.feature}: score={sf.score}, ratio={sf.measured_ratio}")
        else:
            print("  No features split (all 8 symmetric <= 10% asymmetry)")
            
        celeb_results.append({
            "image": os.path.basename(img_path),
            "entry_count": entry_count,
            "overall_score": res.overall_score,
            "split_entries": split_entries
        })
        
    return celeb_results

if __name__ == "__main__":
    run_bilateral_split_experiment()
    run_celebrity_feature_audit()
