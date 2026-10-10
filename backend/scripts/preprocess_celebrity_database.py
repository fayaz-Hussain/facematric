"""
Celebrity Database Preprocessing Script

Generates FaceNet embeddings for all celebrity images and saves them to a pickle file.
This is a one-time preprocessing step before deployment.
"""

import os
import sys
from pathlib import Path
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
import gc

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from deepface import DeepFace


def load_celebrity_images(dataset_path: str) -> dict:
    """
    Load all celebrity images from dataset.
    
    Args:
        dataset_path: Path to celebrity_dataset directory
    
    Returns:
        Dictionary mapping celebrity names to list of image paths
    """
    dataset_dir = Path(dataset_path)
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    
    celebrities = {}
    
    # Iterate through celebrity folders
    for celeb_folder in sorted(dataset_dir.iterdir()):
        if not celeb_folder.is_dir():
            continue
        
        celeb_name = celeb_folder.name
        image_files = []
        
        # Collect all image files
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(celeb_folder.glob(ext))
        
        if image_files:
            celebrities[celeb_name] = [str(img) for img in sorted(image_files)]
    
    return celebrities


def generate_embedding(image_path: str) -> np.ndarray:
    """
    Generate FaceNet embedding for a single image.
    
    Args:
        image_path: Path to image file
    
    Returns:
        128-dimensional embedding vector
    
    Raises:
        Exception if face detection or embedding generation fails
    """
    try:
        # Load image
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Generate embedding using DeepFace
        embedding_obj = DeepFace.represent(
            img_path=img_array,
            model_name="Facenet",
            enforce_detection=True,
            detector_backend="opencv"
        )
        
        # Extract embedding
        if isinstance(embedding_obj, list):
            embedding = np.array(embedding_obj[0]["embedding"])
        else:
            embedding = np.array(embedding_obj["embedding"])
        
        return embedding
    
    except Exception as e:
        raise Exception(f"Failed to process {image_path}: {str(e)}")


def process_celebrity(celeb_name: str, image_paths: list) -> dict:
    """
    Process all images for a single celebrity and average embeddings.
    
    Args:
        celeb_name: Celebrity name
        image_paths: List of image file paths
    
    Returns:
        Dictionary with averaged embedding and metadata
    """
    embeddings = []
    successful_images = []
    
    print(f"\nProcessing {celeb_name} ({len(image_paths)} images)...")
    
    for img_path in image_paths:
        try:
            embedding = generate_embedding(img_path)
            embeddings.append(embedding)
            successful_images.append(img_path)
        except Exception as e:
            print(f"  ⚠️  Skipped {Path(img_path).name}: {str(e)}")
    
    if not embeddings:
        raise Exception(f"No valid embeddings generated for {celeb_name}")
    
    # Average embeddings
    avg_embedding = np.mean(embeddings, axis=0)
    
    # Determine dataset source
    dataset_source = "Bollywood" if any(x in celeb_name for x in [
        "Khan", "Kapoor", "Kumar", "Chopra", "Roshan", "Devgan", "Bachchan"
    ]) else "Western"
    
    print(f"  ✓ Generated embedding from {len(embeddings)}/{len(image_paths)} images")
    
    return {
        "embedding": avg_embedding,
        "dataset_source": dataset_source,
        "thumbnail_path": successful_images[0] if successful_images else None,
        "image_count": len(embeddings)
    }


def main():
    """Main preprocessing function."""
    print("=" * 60)
    print("Celebrity Database Preprocessing")
    print("=" * 60)
    
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    dataset_path = project_root / "celebrity_dataset"
    output_path = script_dir.parent / "data" / "celebrity_embeddings.pkl"
    
    print(f"\nDataset path: {dataset_path}")
    print(f"Output path: {output_path}")
    
    # Load celebrity images
    print("\n📂 Loading celebrity dataset...")
    celebrities = load_celebrity_images(str(dataset_path))
    print(f"Found {len(celebrities)} celebrities")
    
    # Process each celebrity
    print("\n🔄 Generating embeddings...")
    celebrity_embeddings = {}
    failed_celebrities = []
    
    for celeb_name, image_paths in tqdm(celebrities.items(), desc="Processing celebrities"):
        try:
            celeb_data = process_celebrity(celeb_name, image_paths)
            celebrity_embeddings[celeb_name] = celeb_data
            
            # Force garbage collection to free memory
            gc.collect()
            
        except Exception as e:
            print(f"\n❌ Failed to process {celeb_name}: {str(e)}")
            failed_celebrities.append(celeb_name)
    
    # Summary
    print("\n" + "=" * 60)
    print("Processing Complete")
    print("=" * 60)
    print(f"✓ Successfully processed: {len(celebrity_embeddings)} celebrities")
    
    if failed_celebrities:
        print(f"✗ Failed: {len(failed_celebrities)} celebrities")
        for name in failed_celebrities:
            print(f"  - {name}")
    
    # Count by dataset source
    western_count = sum(1 for data in celebrity_embeddings.values() 
                       if data["dataset_source"] == "Western")
    bollywood_count = sum(1 for data in celebrity_embeddings.values() 
                         if data["dataset_source"] == "Bollywood")
    
    print(f"\nDataset breakdown:")
    print(f"  Western: {western_count}")
    print(f"  Bollywood: {bollywood_count}")
    
    # Save to pickle
    print(f"\n💾 Saving embeddings to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(celebrity_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Verify file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Saved successfully ({file_size_mb:.2f} MB)")
    
    print("\n🎉 Done! Celebrity database is ready for use.")
    print(f"\nNext steps:")
    print(f"1. The embeddings file is at: {output_path}")
    print(f"2. Make sure backend/.env has: CELEBRITY_DB_PATH=./data/celebrity_embeddings.pkl")
    print(f"3. Start the backend server: uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
