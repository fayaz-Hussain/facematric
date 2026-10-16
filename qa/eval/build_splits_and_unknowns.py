import os
import sys
import glob
import csv
import pickle
import difflib
import random
import numpy as np
import imagehash
from PIL import Image

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.abspath(".")
DATASET_DIR = os.path.join(PROJECT_ROOT, "celebrity_dataset")
PKL_PATH = os.path.join(PROJECT_ROOT, "backend", "data", "celebrity_embeddings.pkl")
SPLITS_CSV = os.path.join(PROJECT_ROOT, "qa", "reports", "splits.csv")

def clean_name(folder_name):
    name = folder_name
    if name.startswith("pins_"):
        name = name[5:]
    return name.strip().replace(" ", "_")

def build_splits():
    print("=== Building Reference / Validation / Test Splits ===")
    with open(PKL_PATH, "rb") as f:
        db = pickle.load(f)
    db_keys = set(db.keys())

    all_folders = sorted(os.listdir(DATASET_DIR))
    western_dirs = [d for d in all_folders if os.path.isdir(os.path.join(DATASET_DIR, d)) and d.startswith("pins_")][:105]
    bollywood_dirs = [d for d in all_folders if os.path.isdir(os.path.join(DATASET_DIR, d)) and not d.startswith("pins_")][:100]
    
    selected_folders = [(d, "Western") for d in western_dirs] + [(d, "Bollywood") for d in bollywood_dirs]

    random.seed(42)
    splits_records = []
    excluded_identities = []

    for folder_name, source in selected_folders:
        folder_path = os.path.join(DATASET_DIR, folder_name)
        images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        celeb_name = clean_name(folder_name)

        # First 10 images are the reference set
        ref_images = images[:10]
        rem_images = images[10:]

        if len(rem_images) < 6:
            excluded_identities.append((folder_name, len(rem_images)))
            continue

        for img in ref_images:
            splits_records.append({
                "filepath": os.path.join("celebrity_dataset", folder_name, img).replace("\\", "/"),
                "identity": celeb_name,
                "source": source,
                "split": "reference"
            })

        # To keep near-duplicates in the same split:
        # Compute phash for remaining images and group duplicates
        img_hashes = {}
        for img in rem_images:
            p = os.path.join(folder_path, img)
            try:
                with Image.open(p) as im:
                    img_hashes[img] = imagehash.phash(im)
            except Exception:
                img_hashes[img] = None

        # Connected component clustering of near-duplicates (Hamming dist <= 6)
        clusters = []
        visited = set()
        for i, img1 in enumerate(rem_images):
            if img1 in visited:
                continue
            cluster = [img1]
            visited.add(img1)
            h1 = img_hashes[img1]
            if h1 is not None:
                for j in range(i + 1, len(rem_images)):
                    img2 = rem_images[j]
                    if img2 in visited:
                        continue
                    h2 = img_hashes[img2]
                    if h2 is not None and (h1 - h2) <= 6:
                        cluster.append(img2)
                        visited.add(img2)
            clusters.append(cluster)

        # Shuffle clusters and partition 50/50 into validation and test
        random.shuffle(clusters)
        half = len(clusters) // 2
        val_clusters = clusters[:half]
        test_clusters = clusters[half:]

        for cl in val_clusters:
            for img in cl:
                splits_records.append({
                    "filepath": os.path.join("celebrity_dataset", folder_name, img).replace("\\", "/"),
                    "identity": celeb_name,
                    "source": source,
                    "split": "validation"
                })

        for cl in test_clusters:
            for img in cl:
                splits_records.append({
                    "filepath": os.path.join("celebrity_dataset", folder_name, img).replace("\\", "/"),
                    "identity": celeb_name,
                    "source": source,
                    "split": "test"
                })

    # Save to CSV
    os.makedirs(os.path.dirname(SPLITS_CSV), exist_ok=True)
    with open(SPLITS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "identity", "source", "split"])
        writer.writeheader()
        writer.writerows(splits_records)

    print(f"Total images split: {len(splits_records)}")
    ref_c = sum(1 for r in splits_records if r["split"] == "reference")
    val_c = sum(1 for r in splits_records if r["split"] == "validation")
    test_c = sum(1 for r in splits_records if r["split"] == "test")
    print(f"  Reference: {ref_c} images")
    print(f"  Validation: {val_c} images")
    print(f"  Test: {test_c} images")
    print(f"Excluded identities (<6 remaining): {len(excluded_identities)}")

    return splits_records, excluded_identities

def build_unknown_set():
    print("\n=== Building Unknown Set from LFW ===")
    from sklearn.datasets import fetch_lfw_people
    lfw = fetch_lfw_people(data_home="qa/data/unknown", min_faces_per_person=4, resize=None, color=True)
    
    with open(PKL_PATH, "rb") as f:
        db = pickle.load(f)
    celeb_names = set(db.keys())

    # Normalized comparison to detect collisions
    excluded_lfw_people = []
    valid_lfw_indices = []
    
    for idx, target_name in enumerate(lfw.target_names):
        norm_lfw = target_name.lower().replace(" ", "_").replace("-", "_")
        is_collision = False
        for cname in celeb_names:
            norm_c = cname.lower().replace("_", "")
            norm_l = norm_lfw.replace("_", "")
            sim = difflib.SequenceMatcher(None, norm_l, norm_c).ratio()
            if sim > 0.85:
                is_collision = True
                excluded_lfw_people.append((target_name, cname, sim))
                break
        if not is_collision:
            # Person has no collision with our 205 identities
            person_indices = np.where(lfw.target == idx)[0]
            valid_lfw_indices.extend(person_indices)

    print(f"LFW Excluded collisions ({len(excluded_lfw_people)} people):")
    for ex in excluded_lfw_people:
        print(f"  LFW '{ex[0]}' matches DB '{ex[1]}' (similarity={ex[2]:.2f})")

    # Select at least 200 faces across at least 50 distinct people
    random.seed(42)
    valid_people = set(lfw.target[i] for i in valid_lfw_indices)
    print(f"Clean LFW people available: {len(valid_people)} people, {len(valid_lfw_indices)} images")

    # Sample exactly 50 people, taking up to 5 images each -> ~220-250 images
    sampled_people = random.sample(sorted(list(valid_people)), 50)
    sampled_unknown_indices = []
    for p_idx in sampled_people:
        p_imgs = [i for i in valid_lfw_indices if lfw.target[i] == p_idx]
        sampled_unknown_indices.extend(p_imgs[:5])

    print(f"Sampled Unknown Set: {len(sampled_unknown_indices)} images across {len(sampled_people)} people.")
    return lfw, sampled_unknown_indices, excluded_lfw_people

if __name__ == "__main__":
    import numpy as np
    build_splits()
    build_unknown_set()
