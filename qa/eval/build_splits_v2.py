import os
import sys
import csv
import pickle
import random
import time
from PIL import Image
import imagehash

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.abspath(".")
PKL_PATH = os.path.join(PROJECT_ROOT, "backend", "data", "celebrity_embeddings.pkl")
SPLITS_CSV_OLD = os.path.join(PROJECT_ROOT, "qa", "reports", "splits.csv")
SPLITS_CSV_V2 = os.path.join(PROJECT_ROOT, "qa", "reports", "splits_v2.csv")
DATASET_DIR = os.path.join(PROJECT_ROOT, "celebrity_dataset")

def main():
    print("=== Building splits_v2.csv for 275 Identities ===")
    start_time = time.time()
    
    with open(PKL_PATH, "rb") as f:
        db = pickle.load(f)
        
    print(f"Loaded DB with {len(db)} identities.")
    
    random.seed(42)
    splits_records = []
    excluded_identities = []
    
    # Process sorted by identity key for determinism
    sorted_keys = sorted(db.keys())
    
    for idx, ident in enumerate(sorted_keys):
        info = db[ident]
        folder_name = info.get("folder_name")
        source = info.get("dataset_source", "Unknown")
        clean_id = ident.replace(" ", "_")
        
        folder_path = os.path.join(DATASET_DIR, folder_name)
        if not os.path.isdir(folder_path):
            print(f"ERROR: folder not found: {folder_path}")
            continue
            
        # Get reference files from database
        ref_files_db = info.get("reference_files", [])
        ref_filenames = set()
        for rf in ref_files_db:
            # rf is like 'folder/filename.jpg' or just 'filename.jpg'
            rf_base = os.path.basename(rf)
            ref_filenames.add(rf_base)
            splits_records.append({
                "filepath": os.path.join("celebrity_dataset", folder_name, rf_base).replace("\\", "/"),
                "identity": clean_id,
                "source": source,
                "split": "reference"
            })
            
        all_imgs = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        rem_images = [f for f in all_imgs if f not in ref_filenames]
        
        if len(rem_images) < 6:
            excluded_identities.append((ident, len(rem_images)))
            continue
            
        # Compute phash for remaining images
        img_hashes = {}
        for img in rem_images:
            p = os.path.join(folder_path, img)
            try:
                with Image.open(p) as im:
                    img_hashes[img] = imagehash.phash(im)
            except Exception:
                img_hashes[img] = None
                
        # Connected component clustering (Hamming distance <= 6)
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
            
        # Partition clusters 50/50 into validation and test
        random.shuffle(clusters)
        half = len(clusters) // 2
        val_clusters = clusters[:half]
        test_clusters = clusters[half:]
        
        for cl in val_clusters:
            for img in cl:
                splits_records.append({
                    "filepath": os.path.join("celebrity_dataset", folder_name, img).replace("\\", "/"),
                    "identity": clean_id,
                    "source": source,
                    "split": "validation"
                })
                
        for cl in test_clusters:
            for img in cl:
                splits_records.append({
                    "filepath": os.path.join("celebrity_dataset", folder_name, img).replace("\\", "/"),
                    "identity": clean_id,
                    "source": source,
                    "split": "test"
                })
                
        if (idx + 1) % 50 == 0 or (idx + 1) == len(sorted_keys):
            print(f"Processed {idx + 1}/{len(sorted_keys)} identities in {time.time() - start_time:.1f}s...")

    # Write splits_v2.csv
    os.makedirs(os.path.dirname(SPLITS_CSV_V2), exist_ok=True)
    with open(SPLITS_CSV_V2, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "identity", "source", "split"])
        writer.writeheader()
        writer.writerows(splits_records)
        
    total_time = time.time() - start_time
    print(f"\nDone in {total_time:.1f}s!")
    print(f"Total records written: {len(splits_records)}")
    ref_c = sum(1 for r in splits_records if r["split"] == "reference")
    val_c = sum(1 for r in splits_records if r["split"] == "validation")
    test_c = sum(1 for r in splits_records if r["split"] == "test")
    print(f"  Reference: {ref_c}")
    print(f"  Validation: {val_c}")
    print(f"  Test: {test_c}")
    print(f"Excluded identities (< 6 remaining): {len(excluded_identities)}")
    
    # Overlap / contamination analysis
    print("\n=== Checking Test-Set Contamination from splits.csv ===")
    new_ref_set = set(r["filepath"].lower() for r in splits_records if r["split"] == "reference")
    
    old_test = []
    old_val = []
    old_ref = []
    if os.path.exists(SPLITS_CSV_OLD):
        with open(SPLITS_CSV_OLD, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fp = row["filepath"].replace("\\", "/").lower()
                if row["split"] == "test":
                    old_test.append(fp)
                elif row["split"] == "validation":
                    old_val.append(fp)
                elif row["split"] == "reference":
                    old_ref.append(fp)
                    
        test_contam = [fp for fp in old_test if fp in new_ref_set]
        val_contam = [fp for fp in old_val if fp in new_ref_set]
        ref_contam = [fp for fp in old_ref if fp in new_ref_set]
        
        print(f"Old test images: {len(old_test)}")
        print(f"Old test images absorbed into new reference set (CONTAMINATION): {len(test_contam)} ({len(test_contam)/len(old_test)*100:.2f}%)")
        print(f"Old validation images: {len(old_val)}")
        print(f"Old validation images absorbed into new reference set: {len(val_contam)} ({len(val_contam)/len(old_val)*100:.2f}%)")
        print(f"Old reference images: {len(old_ref)}")
        print(f"Old reference images retained in new reference set: {len(ref_contam)} ({len(ref_contam)/len(old_ref)*100:.2f}%)")
    else:
        print(f"Old splits file not found at {SPLITS_CSV_OLD}")

if __name__ == "__main__":
    main()
