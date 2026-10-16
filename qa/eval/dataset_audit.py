import os
import sys
import glob
import pickle
import hashlib
import difflib
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imagehash
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.abspath(".")
DATASET_DIR = os.path.join(PROJECT_ROOT, "celebrity_dataset")
PKL_PATH = os.path.join(PROJECT_ROOT, "backend", "data", "celebrity_embeddings.pkl")
MODEL_PATH = os.path.join(PROJECT_ROOT, "backend", "data", "face_landmarker.task")
OVERLAYS_DIR = os.path.join(PROJECT_ROOT, "qa", "reports", "overlays")
os.makedirs(OVERLAYS_DIR, exist_ok=True)

def clean_celebrity_name(folder_name: str) -> str:
    name = folder_name
    if name.startswith("pins_"):
        name = name[5:]
    name = name.strip().replace(" ", "_")
    return name

def init_detector():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=5,
        min_face_detection_confidence=0.5
    )
    return vision.FaceLandmarker.create_from_options(options)

def audit_dataset():
    print("=" * 70)
    print("PHASE 6: CELEBRITY DATASET AUDIT")
    print("=" * 70)

    # 1. Directory Structure & Inventory
    all_entries = sorted(os.listdir(DATASET_DIR))
    western_dirs = [d for d in all_entries if os.path.isdir(os.path.join(DATASET_DIR, d)) and d.startswith("pins_")]
    bollywood_dirs = [d for d in all_entries if os.path.isdir(os.path.join(DATASET_DIR, d)) and not d.startswith("pins_")]
    total_dirs = len(western_dirs) + len(bollywood_dirs)

    print(f"Total Folders: {total_dirs} (Western: {len(western_dirs)}, Bollywood: {len(bollywood_dirs)})")

    # Image counts per identity
    identity_counts = {}
    fewer_than_20 = []
    
    for d in western_dirs:
        p = os.path.join(DATASET_DIR, d)
        imgs = [f for f in os.listdir(p) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        identity_counts[d] = {"source": "Western", "clean_name": clean_celebrity_name(d), "count": len(imgs), "folder": d}
        if len(imgs) < 20:
            fewer_than_20.append((d, "Western", len(imgs)))

    for d in bollywood_dirs:
        p = os.path.join(DATASET_DIR, d)
        imgs = [f for f in os.listdir(p) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        identity_counts[d] = {"source": "Bollywood", "clean_name": clean_celebrity_name(d), "count": len(imgs), "folder": d}
        if len(imgs) < 20:
            fewer_than_20.append((d, "Bollywood", len(imgs)))

    total_images = sum(v["count"] for v in identity_counts.values())
    western_images = sum(v["count"] for v in identity_counts.values() if v["source"] == "Western")
    bollywood_images = sum(v["count"] for v in identity_counts.values() if v["source"] == "Bollywood")

    print(f"Total Images: {total_images} (Western: {western_images}, Bollywood: {bollywood_images})")
    print(f"Identities with <20 images: {len(fewer_than_20)}")

    # 2. Name Issues: Leftovers, Spelling Variants, Duplicates across sources
    western_names = {clean_celebrity_name(d): d for d in western_dirs}
    bollywood_names = {clean_celebrity_name(d): d for d in bollywood_dirs}

    # Cross-source overlaps
    overlap_names = set(western_names.keys()).intersection(set(bollywood_names.keys()))
    print(f"Exact Cross-Source Overlaps: {overlap_names}")

    # Fuzzy name similarity / spelling variants
    all_names = list(western_names.keys()) + list(bollywood_names.keys())
    spelling_variants = []
    for i in range(len(all_names)):
        for j in range(i + 1, len(all_names)):
            n1 = all_names[i]
            n2 = all_names[j]
            # normalized
            n1_clean = n1.lower().replace("_", "").replace("-", "")
            n2_clean = n2.lower().replace("_", "").replace("-", "")
            sim = difflib.SequenceMatcher(None, n1_clean, n2_clean).ratio()
            if sim > 0.85 and n1 != n2:
                spelling_variants.append((n1, n2, sim))

    print(f"Spelling variants / Near-duplicates found: {len(spelling_variants)}")
    for sv in spelling_variants[:10]:
        print(f"  {sv[0]} <-> {sv[1]} (similarity={sv[2]:.2f})")

    # 3. Embeddings Database Audit (celebrity_embeddings.pkl)
    print("\n--- Auditing celebrity_embeddings.pkl ---")
    with open(PKL_PATH, "rb") as f:
        db = pickle.load(f)

    db_identities = len(db)
    db_western = sum(1 for v in db.values() if v.get("dataset_source") == "Western")
    db_bollywood = sum(1 for v in db.values() if v.get("dataset_source") == "Bollywood")
    print(f"Database identities: {db_identities} (Western: {db_western}, Bollywood: {db_bollywood})")

    # Vector checks
    nan_count = 0
    zero_count = 0
    wrong_dim_count = 0
    missing_thumbnails = []
    leftover_pins_in_db = []

    for name, data in db.items():
        if name.startswith("pins_"):
            leftover_pins_in_db.append(name)
        emb = data.get("embedding")
        if emb is None or emb.shape != (512,):
            wrong_dim_count += 1
        elif np.isnan(emb).any() or np.isinf(emb).any():
            nan_count += 1
        elif np.all(emb == 0):
            zero_count += 1
            
        t_path = data.get("thumbnail_path", "")
        # normalize path
        full_t_path = os.path.join(PROJECT_ROOT, t_path.replace("/", os.sep))
        if not os.path.exists(full_t_path):
            missing_thumbnails.append((name, t_path))

    print(f"Embedding dimension check: {wrong_dim_count} invalid dimensions (all 512-d: {wrong_dim_count == 0})")
    print(f"NaN or Inf vectors: {nan_count}")
    print(f"All-zero vectors: {zero_count}")
    print(f"Leftover 'pins_' in database keys: {len(leftover_pins_in_db)}")
    print(f"Missing thumbnail files: {len(missing_thumbnails)}")

    # 4. Selection Audit on Shipped Database (105 Western + 100 Bollywood, 10 images each = 2050 images)
    print("\n--- Auditing Shipped 205 Identities (First 10 Images Each) ---")
    selected_western = western_dirs[:105]
    selected_bollywood = bollywood_dirs[:100]
    selected_folders = [(d, "Western") for d in selected_western] + [(d, "Bollywood") for d in selected_bollywood]

    detector = init_detector()

    shipped_stats = {
        "total_images": 0,
        "corrupt_images": 0,
        "no_face_images": 0,
        "single_face_images": 0,
        "multi_face_images": 0,
        "exact_file_duplicates": 0,
        "identities_with_duplicates": 0,
        "identities_with_no_face": 0,
        "identities_single_photoset": 0
    }

    per_identity_audit = {}
    all_file_hashes = {}
    all_perceptual_hashes = {}

    for folder_name, source in tqdm(selected_folders, desc="Auditing Shipped Selection"):
        folder_path = os.path.join(DATASET_DIR, folder_name)
        images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        selected_10 = images[:10]
        
        id_info = {
            "folder": folder_name,
            "source": source,
            "images_available": len(images),
            "images_audited": len(selected_10),
            "no_face": 0,
            "single_face": 0,
            "multi_face": 0,
            "corrupt": 0,
            "near_duplicates": 0,
            "exact_duplicates": 0,
            "avg_pairwise_phash_dist": 0.0,
            "is_single_photoset": False
        }

        phashes = []
        dhashes = []
        file_hashes = []

        for img_name in selected_10:
            shipped_stats["total_images"] += 1
            img_path = os.path.join(folder_path, img_name)
            
            # File hash
            try:
                with open(img_path, "rb") as fp:
                    content = fp.read()
                    f_hash = hashlib.sha256(content).hexdigest()
                    if f_hash in all_file_hashes:
                        shipped_stats["exact_file_duplicates"] += 1
                        id_info["exact_duplicates"] += 1
                    all_file_hashes[f_hash] = img_path
                    file_hashes.append(f_hash)
            except Exception:
                shipped_stats["corrupt_images"] += 1
                id_info["corrupt"] += 1
                continue

            # Perceptual hash & PIL check
            try:
                with Image.open(img_path) as pil_img:
                    pil_img.verify()
                with Image.open(img_path) as pil_img:
                    ph = imagehash.phash(pil_img)
                    dh = imagehash.dhash(pil_img)
                    phashes.append(ph)
                    dhashes.append(dh)
            except Exception:
                shipped_stats["corrupt_images"] += 1
                id_info["corrupt"] += 1
                continue

            # MediaPipe Detection
            try:
                cv_img = cv2.imread(img_path)
                if cv_img is None:
                    shipped_stats["corrupt_images"] += 1
                    id_info["corrupt"] += 1
                    continue
                rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
                det_res = detector.detect(mp_image)
                face_count = len(det_res.face_landmarks)
                
                if face_count == 0:
                    shipped_stats["no_face_images"] += 1
                    id_info["no_face"] += 1
                elif face_count == 1:
                    shipped_stats["single_face_images"] += 1
                    id_info["single_face"] += 1
                else:
                    shipped_stats["multi_face_images"] += 1
                    id_info["multi_face"] += 1
            except Exception as e:
                shipped_stats["no_face_images"] += 1
                id_info["no_face"] += 1

        # Near-duplicate analysis among the 10 images
        # Pairwise Hamming distance
        pairwise_dists = []
        near_dup_pairs = 0
        for i in range(len(phashes)):
            for j in range(i + 1, len(phashes)):
                dist = phashes[i] - phashes[j]
                pairwise_dists.append(dist)
                if dist <= 6:  # Near-duplicate threshold in perceptual hash
                    near_dup_pairs += 1

        id_info["near_duplicates"] = near_dup_pairs
        if near_dup_pairs > 0:
            shipped_stats["identities_with_duplicates"] += 1
            
        if id_info["no_face"] > 0:
            shipped_stats["identities_with_no_face"] += 1

        if pairwise_dists:
            avg_dist = float(np.mean(pairwise_dists))
            id_info["avg_pairwise_phash_dist"] = avg_dist
            # If average distance is very small (< 12) or multiple near duplicates, it's a single photoshoot/burst
            if avg_dist < 12.0 or near_dup_pairs >= 3:
                id_info["is_single_photoset"] = True
                shipped_stats["identities_single_photoset"] += 1

        per_identity_audit[folder_name] = id_info

    # 5. Generate Contact Sheets
    print("\n--- Generating Contact Sheets (30 Images per Source) ---")
    random.seed(42)

    # 30 random Pins images
    pins_sample = []
    for d in selected_western:
        p = os.path.join(DATASET_DIR, d)
        imgs = [os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if imgs:
            pins_sample.append((random.choice(imgs), clean_celebrity_name(d)))
    random.shuffle(pins_sample)
    pins_30 = pins_sample[:30]

    # 30 random Bollywood images
    bolly_sample = []
    for d in selected_bollywood:
        p = os.path.join(DATASET_DIR, d)
        imgs = [os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if imgs:
            bolly_sample.append((random.choice(imgs), clean_celebrity_name(d)))
    random.shuffle(bolly_sample)
    bolly_30 = bolly_sample[:30]

    def create_contact_sheet(image_label_pairs, output_path, title):
        cols, rows = 6, 5
        thumb_w, thumb_h = 160, 160
        label_h = 30
        cell_w, cell_h = thumb_w + 10, thumb_h + label_h + 10
        sheet_w = cols * cell_w + 20
        sheet_h = rows * cell_h + 60

        sheet = Image.new("RGB", (sheet_w, sheet_h), color=(30, 30, 30))
        draw = ImageDraw.Draw(sheet)
        draw.text((20, 15), title, fill=(255, 255, 255))

        for idx, (img_path, label) in enumerate(image_label_pairs):
            r = idx // cols
            c = idx % cols
            x = 20 + c * cell_w
            y = 50 + r * cell_h

            try:
                with Image.open(img_path) as im:
                    im_rgb = im.convert("RGB")
                    im_rgb.thumbnail((thumb_w, thumb_h))
                    # Center thumbnail
                    tx = x + (thumb_w - im_rgb.width) // 2
                    ty = y + (thumb_h - im_rgb.height) // 2
                    sheet.paste(im_rgb, (tx, ty))
            except Exception as e:
                draw.rectangle([(x, y), (x + thumb_w, y + thumb_h)], fill=(100, 0, 0))

            # Draw label
            short_label = label[:18]
            draw.text((x + 2, y + thumb_h + 4), short_label, fill=(220, 220, 220))

        sheet.save(output_path, "JPEG", quality=90)
        print(f"Saved contact sheet: {output_path}")

    create_contact_sheet(pins_30, os.path.join(OVERLAYS_DIR, "pins_contact_sheet.jpg"), "Pins (Western) Sample Contact Sheet (30 Random Identities)")
    create_contact_sheet(bolly_30, os.path.join(OVERLAYS_DIR, "bollywood_contact_sheet.jpg"), "Bollywood Sample Contact Sheet (30 Random Identities)")

    # Return full summary dictionary
    summary = {
        "total_folders": total_dirs,
        "western_folders": len(western_dirs),
        "bollywood_folders": len(bollywood_dirs),
        "total_images": total_images,
        "western_images": western_images,
        "bollywood_images": bollywood_images,
        "fewer_than_20_count": len(fewer_than_20),
        "fewer_than_20_list": fewer_than_20,
        "overlap_names": list(overlap_names),
        "spelling_variants": spelling_variants,
        "db_identities": db_identities,
        "db_western": db_western,
        "db_bollywood": db_bollywood,
        "wrong_dim_count": wrong_dim_count,
        "nan_count": nan_count,
        "zero_count": zero_count,
        "leftover_pins_in_db": leftover_pins_in_db,
        "missing_thumbnails": missing_thumbnails,
        "shipped_stats": shipped_stats,
        "per_identity_audit": per_identity_audit
    }

    # Save summary as pickle or json for reporting
    with open("qa/eval/dataset_audit_results.pkl", "wb") as fp:
        pickle.dump(summary, fp)

    print("\nDataset audit completed and saved to qa/eval/dataset_audit_results.pkl")
    return summary

if __name__ == "__main__":
    audit_dataset()
