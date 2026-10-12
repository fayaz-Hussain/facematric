"""
Celebrity Database Preprocessing Script (Tasks 12.1, 12.2, 12.3)

Merges Pins Face Recognition (105 Western) and Bollywood datasets (100 Bollywood),
generates 512-dimensional FaceNet embeddings using batching and Facenet512,
computes normalized averaged embeddings per celebrity,
and writes to backend/data/celebrity_embeddings.pkl matching CelebrityMatcher schema:
{
    celebrity_name: {
        "embedding": np.ndarray (512,),
        "dataset_source": "Western" | "Bollywood",
        "thumbnail_url": str,
        "thumbnail_path": str,
        "image_count": int
    }
}
"""

import os
import sys
from pathlib import Path
import pickle
import cv2
import numpy as np
from tqdm import tqdm
import gc

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(backend_dir))

from deepface.basemodels import Facenet512


def clean_celebrity_name(folder_name: str) -> str:
    """Standardize celebrity folder name by stripping pins_ prefix and formatting."""
    name = folder_name
    if name.startswith("pins_"):
        name = name[5:]
    name = name.strip().replace(" ", "_")
    return name


def collect_celebrity_dataset(dataset_path: Path):
    """
    Select exactly 205 celebrities (105 Western + 100 Bollywood).
    
    Returns:
        List of tuples: (celebrity_name, dataset_source, list_of_image_paths)
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    western_folders = sorted([f for f in dataset_path.iterdir() if f.is_dir() and f.name.startswith("pins_")])
    bollywood_folders = sorted([f for f in dataset_path.iterdir() if f.is_dir() and not f.name.startswith("pins_")])
    
    print(f"Found {len(western_folders)} Western folders, {len(bollywood_folders)} Bollywood folders.")
    
    # Task 12.3: Exactly 205 celebrities (105 Western + 100 Bollywood)
    selected_western = western_folders[:105]
    selected_bollywood = bollywood_folders[:100]
    
    celebrities = []
    
    for folder in selected_western:
        celeb_name = clean_celebrity_name(folder.name)
        images = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg")))
        if images:
            celebrities.append((celeb_name, "Western", images))
            
    for folder in selected_bollywood:
        celeb_name = clean_celebrity_name(folder.name)
        images = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg")))
        if images:
            celebrities.append((celeb_name, "Bollywood", images))
            
    print(f"Total selected celebrities: {len(celebrities)} (Western: {len(selected_western)}, Bollywood: {len(selected_bollywood)})")
    return celebrities


def extract_face_batch(image_paths, max_images=10):
    """
    Load and preprocess up to max_images images for a celebrity.
    Resizes to (160, 160) and normalizes to [-1, 1].
    """
    selected_paths = image_paths[:max_images]
    batch = []
    valid_paths = []
    
    for p in selected_paths:
        try:
            img = cv2.imread(str(p))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (160, 160))
            norm_img = (img_resized.astype(np.float32) / 127.5) - 1.0
            batch.append(norm_img)
            valid_paths.append(str(p))
        except Exception:
            continue
            
    if not batch:
        return None, []
    return np.array(batch, dtype=np.float32), valid_paths


def process_all_celebrities(celebrities, model):
    """Generate averaged embeddings for all celebrities."""
    database = {}
    failed = []
    
    for celeb_name, source, img_paths in tqdm(celebrities, desc="Generating embeddings"):
        try:
            batch, valid_paths = extract_face_batch(img_paths, max_images=10)
            if batch is None or len(valid_paths) == 0:
                print(f"Warning: No valid images for {celeb_name}")
                failed.append(celeb_name)
                continue
                
            # Batch inference
            raw_embeddings = model.predict(batch, batch_size=len(batch), verbose=0)
            
            # Normalize each embedding
            norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            norm_embeddings = raw_embeddings / norms
            
            # Average embeddings
            avg_embedding = np.mean(norm_embeddings, axis=0)
            avg_norm = np.linalg.norm(avg_embedding)
            if avg_norm > 1e-10:
                avg_embedding = avg_embedding / avg_norm
                
            thumbnail_path = valid_paths[0]
            # Create a relative thumbnail url path
            rel_thumbnail = thumbnail_path.replace("\\", "/")
            
            database[celeb_name] = {
                "embedding": avg_embedding.astype(np.float32),
                "dataset_source": source,
                "thumbnail_url": rel_thumbnail,
                "thumbnail_path": rel_thumbnail,
                "image_count": len(valid_paths)
            }
        except Exception as e:
            print(f"Error processing {celeb_name}: {e}")
            failed.append(celeb_name)
            
        if len(database) % 25 == 0:
            gc.collect()
            
    return database, failed


def main():
    print("=" * 60)
    print("Celebrity Database Preprocessing (Tasks 12.1 - 12.3)")
    print("=" * 60)
    
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    dataset_path = project_root / "celebrity_dataset"
    output_path = script_dir.parent / "data" / "celebrity_embeddings.pkl"
    
    celebrities = collect_celebrity_dataset(dataset_path)
    
    print("\nLoading Facenet512 model...")
    model = Facenet512.loadModel()
    print("Facenet512 loaded successfully.")
    
    print("\nGenerating embeddings...")
    database, failed = process_all_celebrities(celebrities, model)
    
    print(f"\nProcessing summary:")
    print(f"Successfully processed: {len(database)}")
    print(f"Failed: {len(failed)}")
    
    western_count = sum(1 for v in database.values() if v["dataset_source"] == "Western")
    bollywood_count = sum(1 for v in database.values() if v["dataset_source"] == "Bollywood")
    print(f"Western count: {western_count}")
    print(f"Bollywood count: {bollywood_count}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(database, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved celebrity embeddings to {output_path} ({file_size_mb:.2f} MB)")
    print("Done!")


if __name__ == "__main__":
    main()
