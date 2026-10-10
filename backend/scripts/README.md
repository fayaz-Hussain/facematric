# Celebrity Database Preprocessing

This directory contains scripts for preprocessing the celebrity dataset.

## Setup

1. Ensure all dependencies are installed:
```bash
cd backend
pip install -r requirements.txt
```

2. Verify celebrity dataset exists:
```bash
ls ../celebrity_dataset/
# Should show 275 celebrity folders
```

## Running the Preprocessing Script

```bash
cd backend
python scripts/preprocess_celebrity_database.py
```

## What It Does

1. **Scans** all celebrity folders in `celebrity_dataset/`
2. **Generates** FaceNet embeddings for each celebrity image
3. **Averages** multiple embeddings per celebrity
4. **Saves** to `data/celebrity_embeddings.pkl`

## Expected Output

```
Celebrity Database Preprocessing
============================================================

Dataset path: /path/to/celebrity_dataset
Output path: /path/to/backend/data/celebrity_embeddings.pkl

📂 Loading celebrity dataset...
Found 275 celebrities

🔄 Generating embeddings...
Processing celebrities: 100%|████████████| 275/275

============================================================
Processing Complete
============================================================
✓ Successfully processed: 205 celebrities
✗ Failed: 70 celebrities

Dataset breakdown:
  Western: 105
  Bollywood: 100

💾 Saving embeddings to data/celebrity_embeddings.pkl...
✓ Saved successfully (2.34 MB)

🎉 Done! Celebrity database is ready for use.
```

## Troubleshooting

### Out of Memory

If processing fails due to memory constraints:
- Process in batches
- Close other applications
- Use a machine with more RAM

### Face Detection Failures

Some images may fail face detection:
- Poor image quality
- Multiple faces
- Extreme angles
- Occlusions

Failed images are skipped automatically.

### DeepFace Issues

If DeepFace fails to initialize:
```bash
pip install --upgrade deepface tensorflow
```

## Output Format

The generated `celebrity_embeddings.pkl` contains:

```python
{
    "Celebrity_Name": {
        "embedding": np.array([...]),  # 128-dimensional vector
        "dataset_source": "Western" or "Bollywood",
        "thumbnail_path": "path/to/first/image.jpg",
        "image_count": 5  # Number of images averaged
    },
    ...
}
```

## Notes

- **First run takes 20-30 minutes** depending on hardware
- Uses GPU if available (TensorFlow)
- Progress saved incrementally
- Failed celebrities logged but don't stop processing
