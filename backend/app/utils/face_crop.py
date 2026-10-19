"""
Shared face crop and preprocessing utility for FaceNet inference.
Used by both the database preprocessing script and the production query path.
"""

from typing import Optional, Tuple, Union, Dict, Any
import os
import cv2
import numpy as np

# Lazy-loaded global detector instance
_crop_detector = None


def get_crop_detector():
    """Get or create singleton MediaPipe FaceLandmarker for face cropping."""
    global _crop_detector
    if _crop_detector is None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        from app.config import settings
        
        model_path = os.path.abspath(settings.mediapipe_model_path)
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=2,  # detect up to 2 faces to identify multi-face photos
            min_face_detection_confidence=0.5
        )
        _crop_detector = vision.FaceLandmarker.create_from_options(options)
    return _crop_detector


def extract_face_crop(
    image: np.ndarray,
    landmarks: Optional[Union[np.ndarray, list]] = None,
    margin: float = 0.10,
    allow_fallback_128: bool = True,
    target_size: Tuple[int, int] = (160, 160),
    return_info: bool = False
) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], Dict[str, Any]]]:
    """
    Crops image to face bounding box plus a margin, resizes to target_size (160x160),
    and normalizes pixel values to [-1.0, 1.0].
    
    Args:
        image: RGB numpy array (uint8) of shape (H, W, 3).
        landmarks: Optional landmarks array/list with pixel coordinates or normalized coords.
                   If not provided, MediaPipe FaceLandmarker is used to detect landmarks.
        margin: Fractional margin to add around bounding box (default: 0.10, i.e. 10%).
        allow_fallback_128: If True, uses the whole image if no face is detected
                            and the image is 128px or smaller.
        target_size: Output image dimensions (width, height) (default: (160, 160)).
        return_info: If True, returns (normalized_image, info_dict).
        
    Returns:
        Normalized float32 numpy array of shape (target_size[0], target_size[1], 3)
        with values in [-1.0, 1.0], or None if no valid crop could be produced.
    """
    info = {
        "face_detected": False,
        "face_count": 0,
        "used_fallback": False,
        "box": None
    }
    
    if image is None or image.size == 0:
        return (None, info) if return_info else None
        
    h, w = image.shape[:2]
    pts = None
    
    # 1. Use provided landmarks if available
    if landmarks is not None:
        if hasattr(landmarks, "points"):
            pts = np.asarray(landmarks.points)[:, :2]
        else:
            arr = np.asarray(landmarks)
            if arr.ndim == 2 and arr.shape[0] >= 4 and arr.shape[1] >= 2:
                pts = arr[:, :2]
                # If normalized coordinates (all <= 1.0), scale to pixel dimensions
                if np.max(pts[:, 0]) <= 1.0 and np.max(pts[:, 1]) <= 1.0 and (w > 1 or h > 1):
                    pts = pts * np.array([w, h])
        if pts is not None and len(pts) > 0:
            info["face_detected"] = True
            info["face_count"] = 1
    
    # 2. Run MediaPipe detector if no landmarks provided
    if pts is None:
        try:
            import mediapipe as mp
            detector = get_crop_detector()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            detection_result = detector.detect(mp_image)
            
            face_count = len(detection_result.face_landmarks) if detection_result.face_landmarks else 0
            info["face_count"] = face_count
            
            if face_count == 1:
                lms = detection_result.face_landmarks[0]
                pts = np.array([[lm.x * w, lm.y * h] for lm in lms])
                info["face_detected"] = True
            elif face_count > 1:
                # Multiple faces: skip
                return (None, info) if return_info else None
            else:
                # No face detected
                info["face_detected"] = False
        except Exception:
            info["face_detected"] = False
            
    # 3. Handle no face detected with 128px fallback
    if pts is None:
        if allow_fallback_128 and max(w, h) <= 128:
            info["used_fallback"] = True
            resized = cv2.resize(image, target_size)
            normalized = (resized.astype(np.float32) / 127.5) - 1.0
            return (normalized, info) if return_info else normalized
        else:
            return (None, info) if return_info else None
            
    # 4. Compute bounding box with margin
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])
    
    bw = max(1.0, max_x - min_x)
    bh = max(1.0, max_y - min_y)
    
    x1 = max(0, int(round(min_x - margin * bw)))
    y1 = max(0, int(round(min_y - margin * bh)))
    x2 = min(w, int(round(max_x + margin * bw)))
    y2 = min(h, int(round(max_y + margin * bh)))
    
    info["box"] = (x1, y1, x2, y2)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        crop = image
        
    resized = cv2.resize(crop, target_size)
    normalized = (resized.astype(np.float32) / 127.5) - 1.0
    
    return (normalized, info) if return_info else normalized
