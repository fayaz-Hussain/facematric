import os
import sys
import csv
import json
import math
import time
import pickle
import random
import traceback
import numpy as np
import cv2

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.abspath(".")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

from app.utils.face_crop import extract_face_crop
from deepface.basemodels import Facenet512

PKL_PATH = os.path.join(PROJECT_ROOT, "backend", "data", "celebrity_embeddings.pkl")
SPLITS_CSV_V2 = os.path.join(PROJECT_ROOT, "qa", "reports", "splits_v2.csv")
PREDICTIONS_CSV = os.path.join(PROJECT_ROOT, "qa", "reports", "celebrity_predictions_v2.csv")
SUMMARY_JSON = os.path.join(PROJECT_ROOT, "qa", "reports", "celebrity_eval_v2_summary.json")

def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2.0 * n)
    m = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))

def main():
    try:
        print("=" * 70, flush=True)
        print("PHASE 4 CELEBRITY EVALUATION ON VALIDATION SPLIT (splits_v2.csv)", flush=True)
        print("=" * 70, flush=True)
        
        # 1. Load Database
        print(f"Loading celebrity database from {PKL_PATH}...", flush=True)
        with open(PKL_PATH, "rb") as f:
            db = pickle.load(f)
        print(f"Loaded {len(db)} identities from database.", flush=True)
        
        id_names = sorted(list(db.keys()))
        id_clean_names = [k.replace(" ", "_") for k in id_names]
        
        # Build normalized centroids matrix of shape (275, 512)
        centroids = []
        for k in id_names:
            emb = np.array(db[k]["embedding"], dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 1e-10:
                emb = emb / norm
            centroids.append(emb)
        centroids_mat = np.array(centroids, dtype=np.float32) # (275, 512)
        print(f"Centroids matrix shape: {centroids_mat.shape}", flush=True)

        # 2. Load Validation Set from splits_v2.csv
        print(f"Loading validation queries from {SPLITS_CSV_V2}...", flush=True)
        val_records = []
        test_count = 0
        ref_count = 0
        with open(SPLITS_CSV_V2, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["split"] == "validation":
                    val_records.append(r)
                elif r["split"] == "test":
                    test_count += 1
                elif r["split"] == "reference":
                    ref_count += 1
                    
        print(f"Loaded {len(val_records)} validation images across {len(set(r['identity'] for r in val_records))} identities.", flush=True)
        print(f"Verified held-out partitions: {ref_count} reference, {test_count} test (STRICTLY SEALED & UNTOUCHED).", flush=True)
        assert len(val_records) > 0, "No validation records found!"
        assert all(r["split"] == "validation" for r in val_records), "Non-validation record in validation list!"

        # 3. Load LFW Unknown Set
        print("\nLoading LFW Unknown Set (240 faces across 50 people)...", flush=True)
        from sklearn.datasets import fetch_lfw_people
        lfw = fetch_lfw_people(data_home="qa/data/unknown/lfw_home", min_faces_per_person=4, resize=None, color=True)
        
        excluded_lfw_people = {"Ben Affleck", "Bill Gates", "Elizabeth Dole", "Emma Watson", 
                               "Gwyneth Paltrow", "Keanu Reeves", "Leonardo DiCaprio", 
                               "Robert De Niro", "Tom Cruise"}
        valid_lfw_people = [i for i, name in enumerate(lfw.target_names) if name not in excluded_lfw_people]
        random.seed(42)
        sampled_50_people = random.sample(valid_lfw_people, 50)
        
        unknown_records = []
        for p_idx in sampled_50_people:
            p_name = lfw.target_names[p_idx]
            p_img_indices = [idx for idx in range(len(lfw.target)) if lfw.target[idx] == p_idx][:5]
            for p_img_idx in p_img_indices:
                img_rgb_float = lfw.images[p_img_idx]
                img_rgb = np.clip(img_rgb_float * 255.0, 0, 255).astype(np.uint8)
                unknown_records.append({
                    "filepath": f"LFW/{p_name}_{p_img_idx}",
                    "identity": "UNKNOWN",
                    "source": "LFW_Unknown",
                    "split": "unknown",
                    "rgb_image": img_rgb
                })
        print(f"Loaded {len(unknown_records)} unknown faces across {len(sampled_50_people)} people.", flush=True)

        # 4. Load Facenet512 Model
        print("\nLoading Facenet512 model...", flush=True)
        model = Facenet512.loadModel()
        print("Facenet512 model loaded successfully.", flush=True)

        # Helper function to crop and embed in streaming chunks
        def process_and_embed_streaming(records, is_unknown=False, chunk_size=256):
            n_records = len(records)
            all_embeddings = []
            valid_indices = []
            fallbacks = 0
            start_time = time.time()
            
            for b_start in range(0, n_records, chunk_size):
                b_end = min(b_start + chunk_size, n_records)
                chunk = records[b_start:b_end]
                chunk_crops = []
                chunk_indices = []
                
                for i, r in enumerate(chunk):
                    global_idx = b_start + i
                    if is_unknown:
                        img_rgb = r["rgb_image"]
                    else:
                        img_bgr = cv2.imread(r["filepath"])
                        if img_bgr is None:
                            continue
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        
                    crop = extract_face_crop(img_rgb, margin=0.10)
                    if crop is None:
                        resized = cv2.resize(img_rgb, (160, 160))
                        crop = (resized.astype(np.float32) / 127.5) - 1.0
                        fallbacks += 1
                        
                    chunk_crops.append(crop)
                    chunk_indices.append(global_idx)
                    
                if chunk_crops:
                    batch_arr = np.array(chunk_crops, dtype=np.float32)
                    preds = model.predict(batch_arr, batch_size=len(chunk_crops), verbose=0)
                    norms = np.linalg.norm(preds, axis=1, keepdims=True)
                    norms = np.maximum(norms, 1e-10)
                    norm_preds = preds / norms
                    all_embeddings.append(norm_preds)
                    valid_indices.extend(chunk_indices)
                    
                elapsed = time.time() - start_time
                processed = b_end
                rate = processed / max(elapsed, 0.1)
                remaining = (n_records - processed) / max(rate, 0.01)
                print(f"  Processed {processed}/{n_records} ({processed/n_records*100:.1f}%) | "
                      f"Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s | "
                      f"Fallbacks: {fallbacks}", flush=True)
                      
            norm_embs = np.vstack(all_embeddings)
            return norm_embs, valid_indices, fallbacks

        # Process validation records
        print("\nProcessing validation records in batches of 256...", flush=True)
        start_all = time.time()
        val_embs, val_valid_idx, val_fallbacks = process_and_embed_streaming(val_records, is_unknown=False, chunk_size=256)
        
        # Process unknown records
        print("\nProcessing unknown records in batches of 256...", flush=True)
        unk_embs, unk_valid_idx, unk_fallbacks = process_and_embed_streaming(unknown_records, is_unknown=True, chunk_size=256)

        # 5. Evaluate Validation Predictions
        print("\n=== Calculating Validation Set Ranking & Metrics ===", flush=True)
        val_sims = val_embs @ centroids_mat.T
        
        results_val = []
        top1_correct = 0
        top5_correct = 0
        
        val_strong = 0   # >= 0.55
        val_mod = 0      # 0.40 <= sim < 0.55
        val_weak = 0     # < 0.40
        
        for i, idx in enumerate(val_valid_idx):
            rec = val_records[idx]
            true_id = rec["identity"]
            scores = val_sims[i]
            
            top_indices = np.argsort(-scores)
            top1_idx = top_indices[0]
            top1_id = id_clean_names[top1_idx]
            top1_sim = float(scores[top1_idx])
            
            top5_ids = [id_clean_names[j] for j in top_indices[:5]]
            top5_sims = [float(scores[j]) for j in top_indices[:5]]
            
            is_top1 = (top1_id == true_id)
            is_top5 = (true_id in top5_ids)
            
            if is_top1:
                top1_correct += 1
            if is_top5:
                top5_correct += 1
                
            if top1_sim >= 0.55:
                strength = "strong"
                val_strong += 1
            elif top1_sim >= 0.40:
                strength = "moderate"
                val_mod += 1
            else:
                strength = "weak"
                val_weak += 1
                
            margin = float(scores[top_indices[0]] - scores[top_indices[1]])
            
            results_val.append({
                "filepath": rec["filepath"],
                "true_identity": true_id,
                "source": rec["source"],
                "split": "validation",
                "top1_identity": top1_id,
                "top1_similarity": f"{top1_sim:.4f}",
                "top2_identity": id_clean_names[top_indices[1]],
                "top2_similarity": f"{float(scores[top_indices[1]]):.4f}",
                "margin": f"{margin:.4f}",
                "strength": strength,
                "returned_at_055": "TRUE" if top1_sim >= 0.55 else "FALSE",
                "returned_at_040": "TRUE" if top1_sim >= 0.40 else "FALSE",
                "top1_correct": "TRUE" if is_top1 else "FALSE",
                "top5_correct": "TRUE" if is_top5 else "FALSE"
            })

        n_val = len(results_val)
        top1_rate = top1_correct / n_val
        t1_low, t1_high = wilson(top1_correct, n_val)
        
        top5_rate = top5_correct / n_val
        t5_low, t5_high = wilson(top5_correct, n_val)
        
        val_strong_rate = val_strong / n_val
        vs_low, vs_high = wilson(val_strong, n_val)
        
        val_mod_rate = val_mod / n_val
        vm_low, vm_high = wilson(val_mod, n_val)
        
        val_weak_rate = val_weak / n_val
        vw_low, vw_high = wilson(val_weak, n_val)
        
        ret_055 = [r for r in results_val if r["returned_at_055"] == "TRUE"]
        corr_055 = sum(1 for r in ret_055 if r["top1_correct"] == "TRUE")
        prec_055 = corr_055 / len(ret_055) if ret_055 else 0.0
        p055_low, p055_high = wilson(corr_055, len(ret_055))
        cov_055 = len(ret_055) / n_val

        # 6. Evaluate Unknown Set Predictions
        print("\n=== Calculating Unknown Set Metrics ===", flush=True)
        unk_sims = unk_embs @ centroids_mat.T
        
        results_unk = []
        unk_strong = 0   # False match at 0.55
        unk_mod = 0      # False match at 0.40
        unk_weak = 0     # Correctly rejected < 0.40
        
        for i, idx in enumerate(unk_valid_idx):
            rec = unknown_records[idx]
            scores = unk_sims[i]
            top_indices = np.argsort(-scores)
            top1_idx = top_indices[0]
            top1_id = id_clean_names[top1_idx]
            top1_sim = float(scores[top1_idx])
            margin = float(scores[top_indices[0]] - scores[top_indices[1]])
            
            if top1_sim >= 0.55:
                strength = "strong"
                unk_strong += 1
            elif top1_sim >= 0.40:
                strength = "moderate"
                unk_mod += 1
            else:
                strength = "weak"
                unk_weak += 1
                
            results_unk.append({
                "filepath": rec["filepath"],
                "true_identity": "UNKNOWN",
                "source": rec["source"],
                "split": "unknown",
                "top1_identity": top1_id,
                "top1_similarity": f"{top1_sim:.4f}",
                "top2_identity": id_clean_names[top_indices[1]],
                "top2_similarity": f"{float(scores[top_indices[1]]):.4f}",
                "margin": f"{margin:.4f}",
                "strength": strength,
                "returned_at_055": "TRUE" if top1_sim >= 0.55 else "FALSE",
                "returned_at_040": "TRUE" if top1_sim >= 0.40 else "FALSE",
                "top1_correct": "FALSE",
                "top5_correct": "FALSE"
            })

        n_unk = len(results_unk)
        unk_strong_rate = unk_strong / n_unk
        us_low, us_high = wilson(unk_strong, n_unk)
        
        unk_mod_rate = unk_mod / n_unk
        um_low, um_high = wilson(unk_mod, n_unk)
        
        unk_weak_rate = unk_weak / n_unk
        uw_low, uw_high = wilson(unk_weak, n_unk)
        
        unk_fm_040 = (unk_strong + unk_mod) / n_unk
        ufm40_low, ufm40_high = wilson(unk_strong + unk_mod, n_unk)

        # 7. Write Predictions CSV
        print(f"\nWriting predictions to {PREDICTIONS_CSV}...", flush=True)
        all_predictions = results_val + results_unk
        fieldnames = [
            "filepath", "true_identity", "source", "split",
            "top1_identity", "top1_similarity", "top2_identity", "top2_similarity",
            "margin", "strength", "returned_at_055", "returned_at_040",
            "top1_correct", "top5_correct"
        ]
        with open(PREDICTIONS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_predictions)

        # 8. Write Summary JSON
        summary = {
            "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_database_identities": len(db),
            "validation_queries_total": n_val,
            "validation_fallbacks_count": val_fallbacks,
            "validation_top1": {
                "correct": top1_correct,
                "total": n_val,
                "accuracy": top1_rate,
                "ci_95": [t1_low, t1_high]
            },
            "validation_top5": {
                "correct": top5_correct,
                "total": n_val,
                "accuracy": top5_rate,
                "ci_95": [t5_low, t5_high]
            },
            "validation_strength_breakdown": {
                "strong_ge_055": {
                    "count": val_strong,
                    "rate": val_strong_rate,
                    "ci_95": [vs_low, vs_high]
                },
                "moderate_040_to_055": {
                    "count": val_mod,
                    "rate": val_mod_rate,
                    "ci_95": [vm_low, vm_high]
                },
                "weak_lt_040": {
                    "count": val_weak,
                    "rate": val_weak_rate,
                    "ci_95": [vw_low, vw_high]
                }
            },
            "validation_operating_point_055": {
                "coverage": cov_055,
                "returned_count": len(ret_055),
                "precision": prec_055,
                "precision_ci_95": [p055_low, p055_high]
            },
            "unknown_set_total": n_unk,
            "unknown_fallbacks_count": unk_fallbacks,
            "unknown_strength_breakdown": {
                "strong_ge_055_false_matches": {
                    "count": unk_strong,
                    "rate": unk_strong_rate,
                    "ci_95": [us_low, us_high]
                },
                "moderate_040_to_055_false_matches": {
                    "count": unk_mod,
                    "rate": unk_mod_rate,
                    "ci_95": [um_low, um_high]
                },
                "weak_lt_040_correctly_rejected": {
                    "count": unk_weak,
                    "rate": unk_weak_rate,
                    "ci_95": [uw_low, uw_high]
                },
                "total_false_matches_at_040": {
                    "count": unk_strong + unk_mod,
                    "rate": unk_fm_040,
                    "ci_95": [ufm40_low, ufm40_high]
                }
            }
        }
        
        with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        total_elap = time.time() - start_all
        print(f"\nEvaluation complete in {total_elap:.1f}s!", flush=True)
        print("\n" + "=" * 70, flush=True)
        print("FINAL SUMMARY RESULTS (275 Identities - Validation Split)", flush=True)
        print("=" * 70, flush=True)
        print(f"Validation Queries (N={n_val}):", flush=True)
        print(f"  Top-1 Accuracy: {top1_correct}/{n_val} = {top1_rate*100:.2f}% (95% CI: [{t1_low*100:.2f}%, {t1_high*100:.2f}%])", flush=True)
        print(f"  Top-5 Accuracy: {top5_correct}/{n_val} = {top5_rate*100:.2f}% (95% CI: [{t5_low*100:.2f}%, {t5_high*100:.2f}%])", flush=True)
        print(f"  Strong (>= 0.55): {val_strong}/{n_val} = {val_strong_rate*100:.2f}% (95% CI: [{vs_low*100:.2f}%, {vs_high*100:.2f}%])", flush=True)
        print(f"  Moderate [0.40, 0.55): {val_mod}/{n_val} = {val_mod_rate*100:.2f}% (95% CI: [{vm_low*100:.2f}%, {vm_high*100:.2f}%])", flush=True)
        print(f"  Weak (< 0.40): {val_weak}/{n_val} = {val_weak_rate*100:.2f}% (95% CI: [{vw_low*100:.2f}%, {vw_high*100:.2f}%])", flush=True)
        print(f"  Precision at 0.55: {corr_055}/{len(ret_055)} = {prec_055*100:.2f}% (95% CI: [{p055_low*100:.2f}%, {p055_high*100:.2f}%])", flush=True)
        print(f"  Coverage at 0.55: {cov_055*100:.2f}% ({len(ret_055)}/{n_val})", flush=True)
        print(f"\nUnknown Faces (LFW, N={n_unk}):", flush=True)
        print(f"  Strong (>= 0.55) [False Match at 0.55]: {unk_strong}/{n_unk} = {unk_strong_rate*100:.2f}% (95% CI: [{us_low*100:.2f}%, {us_high*100:.2f}%])", flush=True)
        print(f"  Moderate [0.40, 0.55): {unk_mod}/{n_unk} = {unk_mod_rate*100:.2f}% (95% CI: [{um_low*100:.2f}%, {um_high*100:.2f}%])", flush=True)
        print(f"  Weak (< 0.40) [Correctly Rejected]: {unk_weak}/{n_unk} = {unk_weak_rate*100:.2f}% (95% CI: [{uw_low*100:.2f}%, {uw_high*100:.2f}%])", flush=True)
        print(f"  Combined False Match at 0.40: {unk_strong + unk_mod}/{n_unk} = {unk_fm_040*100:.2f}% (95% CI: [{ufm40_low*100:.2f}%, {ufm40_high*100:.2f}%])", flush=True)

    except Exception as e:
        print(f"\nCRITICAL ERROR: {type(e)}: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
