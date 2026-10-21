"""
Phase 7 Evaluation Script for Batch 7:
1c: Margin comparison (10%, 25%, 40%) on validation split
1d: Validation metrics (top-1, top-5) and Unknown LFW evaluation (top-1, top-5, share in strong/moderate/weak)
"""

import os
import sys
import csv
import json
import time
import pickle
import random
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(".")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

from app.utils.face_crop import extract_face_crop
from deepface.basemodels import Facenet512

DB_PATH = "backend/data/celebrity_embeddings.pkl"
SPLITS_CSV_V2 = r"..\FaceMatric\qa\reports\splits_v2.csv"
LFW_HOME = r"..\FaceMatric\qa\data\unknown\lfw_home"

def main():
    print("=" * 60)
    print("BATCH 7 - PART 1C & 1D CELEBRITY EVALUATION")
    print("=" * 60)

    # 1. Load Database Centroids
    with open(DB_PATH, "rb") as f:
        db = pickle.load(f)

    id_names = sorted(list(db.keys()))
    # Map identity name variants
    # Note: identity in splits_v2 is folder name without 'pins_' or folder name
    # Let's map folder_name and display_name to index
    id_map = {}
    for i, name in enumerate(id_names):
        id_map[name.lower()] = i
        id_map[name.lower().replace(" ", "_")] = i
        folder = db[name].get("folder_name", "")
        if folder:
            id_map[folder.lower()] = i
            clean_folder = folder.replace("pins_", "").lower()
            id_map[clean_folder] = i
            id_map[clean_folder.replace(" ", "_")] = i

    centroids = []
    for k in id_names:
        emb = np.array(db[k]["embedding"], dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 1e-10:
            emb = emb / norm
        centroids.append(emb)
    centroids_mat = np.array(centroids, dtype=np.float32) # (275, 512)
    print(f"Loaded {len(id_names)} centroids.")

    # 2. Load Validation Set from splits_v2.csv
    with open(SPLITS_CSV_V2, "r", encoding="utf-8") as f:
        val_rows = [r for r in csv.DictReader(f) if r["split"] == "validation"]

    print(f"Total validation images: {len(val_rows)}")

    # Stratified sample: 2 per identity (550 images) for clean margin comparison
    by_id = {}
    for r in val_rows:
        by_id.setdefault(r["identity"], []).append(r)

    random.seed(42)
    sample_val = []
    for ident, rows in sorted(by_id.items()):
        sample_val.extend(random.sample(rows, min(2, len(rows))))

    print(f"Sampled {len(sample_val)} validation images for margin evaluation across {len(by_id)} identities.")

    # Load FaceNet512 model
    print("Loading Facenet512 model...")
    model = Facenet512.loadModel()
    print("Model loaded.")

    # 3. Part 1c: Compare margins (10%, 25%, 40%)
    margins = [0.10, 0.25, 0.40]
    margin_results = {}

    for m in margins:
        print(f"\nEvaluating Margin {int(m*100)}%...")
        crops = []
        valid_records = []
        for r in sample_val:
            p = os.path.join(r"..\FaceMatric", r["filepath"])
            bgr = cv2.imread(p)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            crop = extract_face_crop(rgb, margin=m)
            if crop is None:
                resized = cv2.resize(rgb, (160, 160))
                crop = (resized.astype(np.float32) / 127.5) - 1.0
            crops.append(crop)
            valid_records.append(r)

        batch_arr = np.array(crops, dtype=np.float32)
        preds = model.predict(batch_arr, batch_size=32, verbose=0)
        # Normalize embeddings
        norms = np.linalg.norm(preds, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        embs = preds / norms

        # Cosine similarity matrix: (N, 275)
        sim_matrix = np.dot(embs, centroids_mat.T)

        top1_correct = 0
        top5_correct = 0
        total = len(valid_records)

        for i, r in enumerate(valid_records):
            true_id = r["identity"].lower().replace("pins_", "").replace(" ", "_")
            true_idx = id_map.get(true_id, id_map.get(r["identity"].lower()))
            ranked_indices = np.argsort(sim_matrix[i])[::-1]

            if true_idx is not None:
                if ranked_indices[0] == true_idx:
                    top1_correct += 1
                if true_idx in ranked_indices[:5]:
                    top5_correct += 1

        top1_acc = (top1_correct / total) * 100
        top5_acc = (top5_correct / total) * 100
        margin_results[m] = {"top1": top1_acc, "top5": top5_acc, "total": total}
        print(f"Margin {int(m*100)}%: Top-1 = {top1_acc:.2f}%, Top-5 = {top5_acc:.2f}% (N={total})")

    # 4. Part 1d: Full Validation Set & 240 Unknown LFW Faces
    # For Part 1d, evaluate unknown faces
    print("\n--- Part 1d: Evaluating 240 Unknown LFW Faces ---")
    from sklearn.datasets import fetch_lfw_people
    lfw = fetch_lfw_people(data_home=LFW_HOME, min_faces_per_person=4, resize=None, color=True)
    excluded_names = {"Ben Affleck", "Bill Gates", "Elizabeth Dole", "Emma Watson", 
                      "Gwyneth Paltrow", "Keanu Reeves", "Leonardo DiCaprio", 
                      "Robert De Niro", "Tom Cruise"}
    valid_lfw_indices = [i for i, name in enumerate(lfw.target_names) if name not in excluded_names]
    random.seed(42)
    sampled_50 = random.sample(valid_lfw_indices, 50)

    unknown_crops = []
    for p_idx in sampled_50:
        p_name = lfw.target_names[p_idx]
        img_indices = [idx for idx in range(len(lfw.target)) if lfw.target[idx] == p_idx][:5]
        for idx in img_indices:
            img_rgb_float = lfw.images[idx]
            img_rgb = np.clip(img_rgb_float * 255.0, 0, 255).astype(np.uint8)
            crop = extract_face_crop(img_rgb, margin=0.10)
            if crop is None:
                resized = cv2.resize(img_rgb, (160, 160))
                crop = (resized.astype(np.float32) / 127.5) - 1.0
            unknown_crops.append(crop)
            if len(unknown_crops) >= 240:
                break
        if len(unknown_crops) >= 240:
            break

    print(f"Loaded {len(unknown_crops)} unknown LFW crops.")
    unk_arr = np.array(unknown_crops, dtype=np.float32)
    unk_preds = model.predict(unk_arr, batch_size=32, verbose=0)
    unk_norms = np.linalg.norm(unk_preds, axis=1, keepdims=True)
    unk_norms[unk_norms < 1e-10] = 1.0
    unk_embs = unk_preds / unk_norms

    unk_sims = np.dot(unk_embs, centroids_mat.T)
    unk_top1_sims = np.max(unk_sims, axis=1)

    strong_count = np.sum(unk_top1_sims >= 0.70)
    moderate_count = np.sum((unk_top1_sims >= 0.55) & (unk_top1_sims < 0.70))
    weak_count = np.sum(unk_top1_sims < 0.55)
    n_unk = len(unk_top1_sims)

    print("\nUnknown LFW Faces Distribution (Thresholds: Strong >= 0.70, Moderate >= 0.55):")
    print(f"  Strong (>= 0.70): {strong_count}/{n_unk} ({strong_count/n_unk*100:.2f}%)")
    print(f"  Moderate (0.55 - 0.70): {moderate_count}/{n_unk} ({moderate_count/n_unk*100:.2f}%)")
    print(f"  Weak (< 0.55): {weak_count}/{n_unk} ({weak_count/n_unk*100:.2f}%)")

    # Output summary json
    summary = {
        "margin_comparison": {f"{int(m*100)}%": res for m, res in margin_results.items()},
        "unknown_lfw": {
            "total": n_unk,
            "strong": int(strong_count),
            "strong_pct": float(strong_count/n_unk*100),
            "moderate": int(moderate_count),
            "moderate_pct": float(moderate_count/n_unk*100),
            "weak": int(weak_count),
            "weak_pct": float(weak_count/n_unk*100)
        }
    }
    with open("qa/reports/phase7_celebrity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved summary to qa/reports/phase7_celebrity_summary.json")

if __name__ == "__main__":
    main()
