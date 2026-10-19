"""
Celebrity Database Preprocessing Script (Tasks 12.1, 12.2, 12.3 & Batch 2 Rebuild)

Rebuilds the celebrity embedding database:
- All 275 identity folders (105 Western + 170 Bollywood).
- Up to 20 reference images per identity, chosen evenly across the sorted file list.
- MediaPipe face crop with 10% margin using extract_face_crop.
- Skips images with no face or several faces.
- For images <= 128px where MediaPipe finds no face, falls back to whole image and logs count.
- Skips near-duplicates using perceptual hash distance <= 6 (imagehash.phash).
- Clean display names (Title Case, pins_ and underscores stripped, with exception mapping).
- Relative thumbnail paths (relative to celebrity_dataset/).
- Backs up old database to celebrity_embeddings.old.pkl.
"""

import os
import sys
import shutil
from pathlib import Path
import pickle
import cv2
from PIL import Image
import imagehash
import numpy as np
from tqdm import tqdm
import gc

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(backend_dir))

from app.utils.face_crop import extract_face_crop
from deepface.basemodels import Facenet512


NAME_EXCEPTIONS = {
    "alycia dabnem carey": "Alycia Debnam-Carey",
    "kiernen shipka": "Kiernan Shipka",
    "henry cavil": "Henry Cavill",
    "ellen page": "Elliot Page",
}


def get_display_name(folder_name: str) -> str:
    """Format folder name to standardized display name with exceptions."""
    name = folder_name
    if name.startswith("pins_"):
        name = name[5:]
    name = name.strip().replace("_", " ")
    key = name.lower()
    if key in NAME_EXCEPTIONS:
        return NAME_EXCEPTIONS[key]
    return name.title()


def find_dataset_dir() -> Path:
    """Locate celebrity_dataset directory across repo roots."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent.parent / "celebrity_dataset",
        script_dir.parent.parent.parent / "FaceMatric" / "celebrity_dataset",
        Path(r"C:\Users\FayazHussain\Desktop\FaceMatric\celebrity_dataset"),
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError(f"Could not find celebrity_dataset in {[str(c) for c in candidates]}")


def collect_identity_folders(dataset_path: Path):
    """
    Collect all 275 identity folders (105 Western + 170 Bollywood).
    """
    folders = sorted([f for f in dataset_path.iterdir() if f.is_dir()])
    western = [f for f in folders if f.name.startswith("pins_")]
    bollywood = [f for f in folders if not f.name.startswith("pins_")]
    
    print(f"Discovered {len(folders)} total folders: {len(western)} Western, {len(bollywood)} Bollywood.")
    identities = []
    for f in folders:
        source = "Western" if f.name.startswith("pins_") else "Bollywood"
        display_name = get_display_name(f.name)
        images = sorted(
            list(f.glob("*.jpg")) + list(f.glob("*.png")) + list(f.glob("*.jpeg")),
            key=lambda p: p.name
        )
        if images:
            identities.append((f, display_name, source, images))
        else:
            print(f"Warning: No images in folder {f.name}")
            
    return identities


def select_reference_images(
    images,
    max_refs=20,
    margin=0.10
):
    """
    Select up to max_refs reference images evenly across sorted files.
    Skips images with no face or >1 face.
    Allows <= 128px fallback when no face detected.
    Skips near-duplicates where phash distance <= 6.
    
    Returns:
        valid_crops, valid_paths, fallback_count
    """
    n_total = len(images)
    if n_total == 0:
        return [], [], 0
        
    # Build candidate index list distributed evenly across sorted list
    num_bins = min(max_refs, n_total)
    bin_size = n_total / float(num_bins)
    
    # Priority order of indices: start from bin centers
    candidate_indices = []
    for b in range(num_bins):
        center_idx = int(round((b + 0.5) * bin_size))
        center_idx = max(0, min(n_total - 1, center_idx))
        if center_idx not in candidate_indices:
            candidate_indices.append(center_idx)
            
    # Add all remaining indices in stride order for fallback topping up
    remaining = [i for i in range(n_total) if i not in candidate_indices]
    candidate_indices.extend(remaining)
    
    accepted_crops = []
    accepted_paths = []
    accepted_hashes = []
    fallback_count = 0
    
    for idx in candidate_indices:
        if len(accepted_crops) >= max_refs:
            break
            
        p = images[idx]
        try:
            img_bgr = cv2.imread(str(p))
            if img_bgr is None or img_bgr.size == 0:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Crop using shared face crop function
            crop, info = extract_face_crop(
                img_rgb,
                landmarks=None,
                margin=margin,
                allow_fallback_128=True,
                return_info=True
            )
            
            # Filter: skip if several faces detected
            if info["face_count"] > 1:
                continue
                
            # Filter: skip if no face and not fallback
            if crop is None:
                continue
                
            if info.get("used_fallback", False):
                fallback_count += 1
                
            # Filter: near-duplicate check with perceptual hash
            pil_img = Image.fromarray(img_rgb)
            img_hash = imagehash.phash(pil_img)
            
            is_near_dup = False
            for prev_hash in accepted_hashes:
                if (img_hash - prev_hash) <= 6:
                    is_near_dup = True
                    break
                    
            if is_near_dup:
                continue
                
            accepted_crops.append(crop)
            accepted_paths.append(p)
            accepted_hashes.append(img_hash)
            
        except Exception:
            continue
            
    return accepted_crops, accepted_paths, fallback_count


def process_identities(identities, model, dataset_path: Path):
    """Generate averaged embeddings for all identities."""
    database = {}
    failed = []
    total_fallbacks = 0
    total_refs_used = 0
    
    for folder, display_name, source, images in tqdm(identities, desc="Processing identities"):
        try:
            crops, paths, fallbacks = select_reference_images(
                images,
                max_refs=20,
                margin=0.10
            )
            total_fallbacks += fallbacks
            
            if not crops:
                print(f"Warning: No valid reference crops for {display_name} ({folder.name})")
                failed.append(display_name)
                continue
                
            total_refs_used += len(crops)
            
            # Predict FaceNet512 embeddings
            batch = np.array(crops, dtype=np.float32)
            raw_embs = model.predict(batch, batch_size=len(batch), verbose=0)
            
            # Unit normalize each embedding
            norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
            norm_embs = raw_embs / np.maximum(norms, 1e-10)
            
            # Average and unit normalize centroid
            avg_emb = np.mean(norm_embs, axis=0)
            avg_emb = avg_emb / np.maximum(np.linalg.norm(avg_emb), 1e-10)
            
            # Thumbnail path relative to celebrity_dataset/
            rel_thumbnail = paths[0].relative_to(dataset_path).as_posix()
            
            database[display_name] = {
                "embedding": avg_emb.astype(np.float32),
                "dataset_source": source,
                "thumbnail_url": rel_thumbnail,
                "thumbnail_path": rel_thumbnail,
                "image_count": len(crops),
                "display_name": display_name,
                "folder_name": folder.name,
                "reference_files": [p.relative_to(dataset_path).as_posix() for p in paths]
            }
            
        except Exception as e:
            print(f"Error processing {display_name}: {e}")
            failed.append(display_name)
            
        if len(database) % 25 == 0:
            gc.collect()
            
    print(f"\nCompleted processing:")
    print(f"  Total identities processed: {len(database)}")
    print(f"  Total reference images embedded: {total_refs_used}")
    print(f"  Images using <=128px fallback (no face detected): {total_fallbacks}")
    print(f"  Failed identities: {len(failed)}")
    
    return database, failed, total_fallbacks


def main():
    print("=" * 60)
    print("Celebrity Database Rebuild (All 275 Identities)")
    print("=" * 60)
    
    script_dir = Path(__file__).resolve().parent
    dataset_path = find_dataset_dir()
    output_path = script_dir.parent / "data" / "celebrity_embeddings.pkl"
    backup_path = script_dir.parent / "data" / "celebrity_embeddings.old.pkl"
    
    print(f"Dataset path: {dataset_path}")
    print(f"Output path:  {output_path}")
    
    # Backup existing database if not already backed up
    if output_path.exists() and not backup_path.exists():
        print(f"Backing up existing database to {backup_path}")
        shutil.copy2(output_path, backup_path)
    elif backup_path.exists():
        print(f"Backup already exists at {backup_path}")
        
    identities = collect_identity_folders(dataset_path)
    assert len(identities) == 275, f"Expected 275 identity folders, got {len(identities)}"
    
    print("\nLoading FaceNet512 model...")
    model = Facenet512.loadModel()
    print("FaceNet512 model loaded successfully.")
    
    print("\nExtracting face crops and generating embeddings...")
    database, failed, total_fallbacks = process_identities(identities, model, dataset_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(database, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved {len(database)} celebrity embeddings to {output_path} ({file_size_mb:.2f} MB)")
    print("Done!")


if __name__ == "__main__":
    main()
