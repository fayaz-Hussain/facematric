"""
Face Detector Component

Uses MediaPipe Face Mesh to detect faces and extract 468 3D facial landmarks.
"""

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
from typing import Optional, Tuple

from app.models.internal import ImageData, Landmarks
from app.utils.errors import ValidationError, ErrorCode
from app.utils.logger import get_logger
from app.config import settings

logger = get_logger(__name__)

# 3D canonical model points for solvePnP head pose estimation (in mm)
MODEL_POINTS_6 = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (landmark 1)
    (0.0, -330.0, -65.0),        # Chin (landmark 152)
    (-225.0, 170.0, -135.0),     # Left eye outer corner (landmark 33)
    (225.0, 170.0, -135.0),      # Right eye outer corner (landmark 263)
    (-150.0, -150.0, -125.0),    # Left mouth corner (landmark 61)
    (150.0, -150.0, -125.0)      # Right mouth corner (landmark 291)
], dtype=np.float64)


class FaceDetector:
    """
    Detects faces and extracts 468 3D facial landmarks using MediaPipe Face Mesh.
    
    Features:
    - Single face detection validation
    - 468 landmark extraction with (x, y, z) coordinates
    - Confidence score validation
    - Occlusion detection (glasses, masks)
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize MediaPipe Face Landmarker.
        
        Args:
            model_path: Path to face_landmarker.task model file
        """
        self.model_path = model_path or settings.mediapipe_model_path
        self.detector = None
        # MediaPipe only reports up to num_faces detections. The PRD requires
        # MULTIPLE_FACES when a group photo is uploaded, so we must request at
        # least 2 faces even if settings.mediapipe_max_faces is 1.
        self.num_faces_limit = max(2, int(settings.mediapipe_max_faces))
        self._initialize_detector()
    
    def _initialize_detector(self) -> None:
        """Initialize MediaPipe Face Landmarker with configuration."""
        try:
            base_options = python.BaseOptions(
                model_asset_path=self.model_path
            )
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=self.num_faces_limit,
                min_face_detection_confidence=settings.mediapipe_min_detection_confidence,
                min_face_presence_confidence=settings.mediapipe_min_detection_confidence,
                min_tracking_confidence=settings.mediapipe_min_detection_confidence
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
            logger.info(f"MediaPipe Face Landmarker initialized with model: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {str(e)}")
            raise RuntimeError(f"MediaPipe initialization failed: {str(e)}")
    
    def detect_faces(self, image_data: ImageData) -> Landmarks:
        """
        Detect face and extract landmarks from image.
        
        Args:
            image_data: Normalized image data
        
        Returns:
            Landmarks: 468 3D facial landmarks
        
        Raises:
            ValidationError: If face detection fails or doesn't meet requirements
        """
        # Convert to MediaPipe Image format
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_data.array
        )
        
        # Detect faces
        try:
            detection_result = self.detector.detect(mp_image)
        except Exception as e:
            logger.error(f"MediaPipe detection failed: {str(e)}")
            raise ValidationError(
                ErrorCode.PROCESSING_FAILED,
                "Face detection failed. Please try again with a different photo."
            )
        
        # Validate face count
        if not detection_result.face_landmarks:
            raise ValidationError(
                ErrorCode.NO_FACE_DETECTED,
                "We couldn't detect a face in your photo. Please take a clear, "
                "front-facing photo with good lighting."
            )
        
        if len(detection_result.face_landmarks) > 1:
            raise ValidationError(
                ErrorCode.MULTIPLE_FACES,
                "Multiple faces detected. Please upload a photo with only one person."
            )
        
        # Extract landmarks
        face_landmarks = detection_result.face_landmarks[0]
        
        # Convert to numpy array with absolute pixel coordinates
        landmarks_array = np.array([
            [
                lm.x * image_data.width,   # Convert normalized x to pixels
                lm.y * image_data.height,  # Convert normalized y to pixels
                lm.z  # Keep z as is (relative depth)
            ]
            for lm in face_landmarks
        ], dtype=np.float32)
        
        # Standard MediaPipe Face Mesh uses 468 facial mesh landmarks
        # If model includes iris landmarks (478 points), take the standard 468 landmarks
        if len(landmarks_array) > 468:
            landmarks_array = landmarks_array[:468]
        
        # Calculate average landmark confidence
        # MediaPipe Face Landmarker provides visibility and presence scores per landmark
        # Compute confidence as the average of these quality metrics
        avg_confidence = self._calculate_landmark_confidence(
            face_landmarks, 
            landmarks_array, 
            image_data
        )
        
        logger.debug(f"Calculated average landmark confidence: {avg_confidence:.3f}")
        
        # Validate landmark quality
        if avg_confidence < settings.landmark_confidence_threshold:
            raise ValidationError(
                ErrorCode.POOR_QUALITY,
                "Photo quality is too low. Please ensure your face is well-lit "
                "and facing the camera directly."
            )
        
        # Check head pose (yaw and roll gates)
        self._check_pose(landmarks_array, image_data)
        
        # Check for occlusions
        self._check_occlusions(landmarks_array, image_data)
        
        logger.info(
            f"Face detected successfully with {len(landmarks_array)} landmarks, "
            f"confidence: {avg_confidence:.2f}"
        )
        
        return Landmarks(
            points=landmarks_array,
            confidence=avg_confidence,
            image_width=image_data.width,
            image_height=image_data.height
        )
    
    def _calculate_landmark_confidence(
        self,
        face_landmarks: list,
        landmarks_array: np.ndarray,
        image_data: ImageData
    ) -> float:
        """
        Calculate average landmark confidence based on quality metrics.
        
        Since MediaPipe Face Landmarker doesn't directly provide per-landmark
        confidence scores, we compute a proxy confidence based on:
        1. Visibility scores (if available in landmark attributes)
        2. Presence scores (if available in landmark attributes) 
        3. Geometric consistency checks
        
        Args:
            face_landmarks: List of MediaPipe landmarks with attributes
            landmarks_array: Numpy array of landmark coordinates
            image_data: Image data for context
        
        Returns:
            Average confidence score in range [0.0, 1.0]
        """
        confidence_scores = []
        
        # Check if MediaPipe landmarks have visibility/presence attributes
        # These attributes may be available depending on the model version
        for lm in face_landmarks:
            scores_for_landmark = []
            
            # Check for visibility score (how visible the landmark is)
            if hasattr(lm, 'visibility') and lm.visibility is not None:
                scores_for_landmark.append(lm.visibility)
            
            # Check for presence score (confidence that landmark exists)
            if hasattr(lm, 'presence') and lm.presence is not None:
                scores_for_landmark.append(lm.presence)
            
            # If we have any scores for this landmark, average them
            if scores_for_landmark:
                confidence_scores.append(np.mean(scores_for_landmark))
        
        # If MediaPipe provided visibility/presence scores, use their average
        if confidence_scores:
            avg_confidence = float(np.mean(confidence_scores))
            logger.debug(
                f"Calculated confidence from {len(confidence_scores)} landmark "
                f"visibility/presence scores: {avg_confidence:.3f}"
            )
            return avg_confidence
        
        # Fallback: Compute geometric quality metrics
        # Check if landmarks form a reasonable face shape
        geometric_quality = self._assess_geometric_quality(landmarks_array, image_data)
        
        logger.debug(
            f"MediaPipe landmarks lack visibility/presence scores. "
            f"Using geometric quality metric: {geometric_quality:.3f}"
        )
        
        return geometric_quality
    
    def _assess_geometric_quality(
        self,
        landmarks: np.ndarray,
        image_data: ImageData
    ) -> float:
        """
        Assess landmark quality based on geometric consistency.
        
        Checks:
        1. Landmarks are well-distributed (not clustered in one area)
        2. Key facial features have reasonable relative positions
        3. Z-depth values are within reasonable range
        
        Returns:
            Quality score in range [0.0, 1.0]
        """
        quality_score = 1.0  # Start optimistic
        
        # Check 1: Spatial distribution (landmarks shouldn't be too clustered)
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        
        # Calculate face bounding box dimensions relative to image size
        face_width = np.max(x_coords) - np.min(x_coords)
        face_height = np.max(y_coords) - np.min(y_coords)
        
        rel_face_w = face_width / (image_data.width + 1e-6)
        rel_face_h = face_height / (image_data.height + 1e-6)
        face_span = np.sqrt(max(0.0, rel_face_w * rel_face_h))
        
        # Expect face to occupy a reasonable area (> 12% scale)
        distribution_score = min(1.0, face_span / 0.15)
        quality_score *= distribution_score
        
        # Check 2: Z-depth consistency (shouldn't have extreme outliers)
        z_coords = landmarks[:, 2]
        z_std = np.std(z_coords)
        
        # Relax z-depth checks - allow more variation
        if z_std > 0.25:
            quality_score *= 0.95
        elif z_std < 0.0001:
            quality_score *= 0.95
        
        # Check 3: Landmarks within image bounds
        # All landmarks should be within image dimensions
        in_bounds_x = np.logical_and(x_coords >= 0, x_coords <= image_data.width)
        in_bounds_y = np.logical_and(y_coords >= 0, y_coords <= image_data.height)
        in_bounds_ratio = np.mean(np.logical_and(in_bounds_x, in_bounds_y))
        quality_score *= in_bounds_ratio
        
        # Clamp final score to reasonable range [0.0, 1.0]
        quality_score = max(0.0, min(1.0, quality_score))
        
        return float(quality_score)
    
    @staticmethod
    def estimate_head_pose(
        landmarks: np.ndarray,
        width: int,
        height: int
    ) -> Tuple[float, float, float]:
        """
        Estimate head pose (yaw, pitch, roll in degrees) using solvePnP on 6 standard facial landmarks.
        
        Args:
            landmarks: Array of 468 3D landmarks with coordinates in pixels
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Tuple of (yaw, pitch, roll) in degrees
        """
        image_points = np.array([
            landmarks[1, :2],      # Nose tip
            landmarks[152, :2],    # Chin
            landmarks[33, :2],     # Left eye outer corner
            landmarks[263, :2],    # Right eye outer corner
            landmarks[61, :2],     # Left mouth corner
            landmarks[291, :2]     # Right mouth corner
        ], dtype=np.float64)

        focal_length = float(width)
        center = (width / 2.0, height / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        try:
            success, rvec, tvec = cv2.solvePnP(
                MODEL_POINTS_6, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success:
                logger.warning("solvePnP head pose estimation failed")
                return 0.0, 0.0, 0.0
        except Exception as e:
            logger.warning(f"solvePnP head pose estimation failed: {e}")
            return 0.0, 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])

        # Normalize/unwrap angles to [-90, +90]
        if yaw > 90.0:
            yaw -= 180.0
        elif yaw < -90.0:
            yaw += 180.0

        if roll > 90.0:
            roll -= 180.0
        elif roll < -90.0:
            roll += 180.0

        return yaw, pitch, roll

    def _check_pose(
        self,
        landmarks: np.ndarray,
        image_data: ImageData
    ) -> None:
        """
        Validate that the head is facing the camera straight on.
        
        Rejects when |yaw| > 12 degrees or |roll| > 10 degrees with POOR_QUALITY error.
        
        Args:
            landmarks: Array of 468 3D landmarks
            image_data: Normalized image data
            
        Raises:
            ValidationError: If yaw or roll exceeds configured thresholds
        """
        yaw, pitch, roll = self.estimate_head_pose(landmarks, image_data.width, image_data.height)
        logger.debug(f"Estimated head pose: yaw={yaw:.2f}°, pitch={pitch:.2f}°, roll={roll:.2f}°")

        yaw_exceeded = abs(yaw) > settings.max_pose_yaw_deg
        roll_exceeded = abs(roll) > settings.max_pose_roll_deg

        if yaw_exceeded or roll_exceeded:
            logger.warning(
                f"Head pose exceeds threshold: yaw={yaw:.2f}° (max {settings.max_pose_yaw_deg}°), "
                f"roll={roll:.2f}° (max {settings.max_pose_roll_deg}°)"
            )
            if yaw_exceeded and roll_exceeded:
                msg = "Please face the camera directly and keep your head level."
            elif yaw_exceeded:
                msg = "Please face the camera directly."
            else:
                msg = "Please keep your head level."
            raise ValidationError(
                ErrorCode.POOR_QUALITY,
                msg
            )

    def _check_occlusions(
        self,
        landmarks: np.ndarray,
        image_data: ImageData
    ) -> None:
        """
        Detect occlusions by analyzing landmark patterns.
        
        Implements multiple detection strategies:
        - Glasses Detection: Analyzes z-depth variance and geometric patterns in eye regions
        - Mask Detection: Analyzes mouth region visibility and landmark distribution
        
        Args:
            landmarks: Array of 468 3D landmarks
            image_data: Image data for context
        
        Raises:
            ValidationError: If significant occlusion detected
        """
        # Perform glasses detection on eye regions
        self._detect_glasses(landmarks)
        
        # Perform mask detection on mouth region
        self._detect_mask(landmarks)
    
    def _detect_glasses(self, landmarks: np.ndarray) -> None:
        """
        Detect glasses by analyzing eye region landmark patterns.
        
        Detection strategies:
        1. Z-depth variance: Glasses protrude from face plane
        2. Geometric pattern analysis: Check for abnormal eye landmark arrangements
        3. Bilateral symmetry: Glasses affect both eyes similarly
        
        Args:
            landmarks: Array of 468 3D landmarks
        
        Raises:
            ValidationError: If glasses detected
        """
        # Eye region landmark indices
        # Left eye: outer corner, upper lid, lower lid, inner corner, brow points
        left_eye_indices = [362, 385, 387, 263, 373, 380, 374, 381, 382, 466]
        # Right eye: outer corner, upper lid, lower lid, inner corner, brow points
        right_eye_indices = [33, 160, 158, 133, 153, 144, 145, 163, 7, 246]
        
        # Extract eye region points
        left_eye_points = landmarks[left_eye_indices]
        right_eye_points = landmarks[right_eye_indices]
        
        # Strategy 1: Z-depth variance analysis
        # Glasses create abnormal depth variance in eye region
        left_z_std = np.std(left_eye_points[:, 2])
        right_z_std = np.std(right_eye_points[:, 2])
        
        # Threshold for z-depth variance (increased from 0.05 to 0.12 to reduce false positives)
        z_std_threshold = 0.12
        
        # Check if either eye shows high z-variance
        if left_z_std > z_std_threshold or right_z_std > z_std_threshold:
            logger.warning(
                f"Potential glasses detected via z-depth variance: "
                f"left={left_z_std:.4f}, right={right_z_std:.4f}"
            )
            raise ValidationError(
                ErrorCode.OCCLUSION_DETECTED,
                "We detected accessories covering your face. Please upload a photo "
                "without glasses, masks, or other obstructions for accurate analysis."
            )
        
        # Strategy 2: Geometric pattern analysis
        # Calculate eye height (vertical distance)
        # Left eye: distance between upper lid (385) and lower lid (380)
        left_eye_height = np.abs(landmarks[385][1] - landmarks[380][1])
        # Right eye: distance between upper lid (159) and lower lid (145)
        right_eye_height = np.abs(landmarks[159][1] - landmarks[145][1])
        
        # Calculate eye width (horizontal distance)
        # Left eye: outer corner (362) to inner corner (263)
        left_eye_width = np.abs(landmarks[362][0] - landmarks[263][0])
        # Right eye: outer corner (33) to inner corner (133)
        right_eye_width = np.abs(landmarks[33][0] - landmarks[133][0])
        
        # Compute aspect ratios
        left_eye_aspect = left_eye_height / left_eye_width if left_eye_width > 0 else 0
        right_eye_aspect = right_eye_height / right_eye_width if right_eye_width > 0 else 0
        
        # Normal eye aspect ratio is typically 0.2-0.4
        # Glasses can distort this ratio significantly
        # Increased range from [0.1-0.5] to [0.08-0.6] to be more lenient
        aspect_ratio_threshold_low = 0.08
        aspect_ratio_threshold_high = 0.6
        
        if (left_eye_aspect < aspect_ratio_threshold_low or 
            left_eye_aspect > aspect_ratio_threshold_high or
            right_eye_aspect < aspect_ratio_threshold_low or 
            right_eye_aspect > aspect_ratio_threshold_high):
            logger.warning(
                f"Abnormal eye aspect ratios detected: "
                f"left={left_eye_aspect:.3f}, right={right_eye_aspect:.3f}"
            )
            raise ValidationError(
                ErrorCode.OCCLUSION_DETECTED,
                "We detected accessories covering your face. Please upload a photo "
                "without glasses, masks, or other obstructions for accurate analysis."
            )
        
        # Strategy 3: Check for abnormal z-depth positioning
        # Calculate average z-depth for each eye
        left_z_mean = np.mean(left_eye_points[:, 2])
        right_z_mean = np.mean(right_eye_points[:, 2])
        
        # Eyes should have similar z-depth; large difference suggests occlusion
        # Increased threshold from 0.15 to 0.30 to reduce false positives
        z_depth_diff = np.abs(left_z_mean - right_z_mean)
        z_depth_diff_threshold = 0.30
        
        if z_depth_diff > z_depth_diff_threshold:
            logger.warning(
                f"Asymmetric eye depth detected: diff={z_depth_diff:.4f}"
            )
            raise ValidationError(
                ErrorCode.OCCLUSION_DETECTED,
                "We detected accessories covering your face. Please upload a photo "
                "without glasses, masks, or other obstructions for accurate analysis."
            )
    
    def _detect_mask(self, landmarks: np.ndarray) -> None:
        """
        Detect masks by analyzing mouth region visibility patterns.
        
        Detection strategies:
        1. Z-depth variance: Masks create depth discontinuity
        2. Mouth visibility: Check if mouth landmarks are properly detected
        3. Lower face coverage: Analyze chin and jaw landmark patterns
        
        Args:
            landmarks: Array of 468 3D landmarks
        
        Raises:
            ValidationError: If mask detected
        """
        # Mouth region landmark indices
        # Upper lip, lower lip, corners, and surrounding points
        mouth_indices = [61, 291, 0, 17, 84, 181, 91, 146, 78, 308, 324, 318, 13, 14, 87, 317]
        
        # Chin and lower jaw indices for mask coverage detection
        chin_jaw_indices = [152, 377, 400, 378, 379, 365, 397, 288, 361, 323]
        
        # Extract mouth and chin region points
        mouth_points = landmarks[mouth_indices]
        chin_jaw_points = landmarks[chin_jaw_indices]
        
        # Strategy 1: Z-depth variance analysis for mouth region
        mouth_z_std = np.std(mouth_points[:, 2])
        
        # Masks create higher variance in mouth z-depth
        # Increased threshold from 0.075 to 0.15 to reduce false positives
        z_std_threshold_mouth = 0.15  # More lenient threshold for normal facial variation
        
        if mouth_z_std > z_std_threshold_mouth:
            logger.warning(
                f"Potential mask detected via mouth z-depth variance: {mouth_z_std:.4f}"
            )
            raise ValidationError(
                ErrorCode.OCCLUSION_DETECTED,
                "We detected a mask or obstruction covering your mouth. "
                "Please upload a photo without face coverings for accurate analysis."
            )
        
        # Strategy 2: Check mouth visibility and outer lip bounds
        # Upper outer lip center: landmark 0
        # Lower outer lip center: landmark 17
        upper_lip_y = landmarks[0][1]
        lower_lip_y = landmarks[17][1]
        mouth_height = np.abs(upper_lip_y - lower_lip_y)
        
        # Calculate mouth width for context (corners 61 to 291)
        left_corner = landmarks[61][0]
        right_corner = landmarks[291][0]
        mouth_width = np.abs(right_corner - left_corner)
        
        # Outer mouth height-to-width ratio should be reasonable
        # Extremely low ratio (< 0.05) indicates the mouth region is obscured or flattened
        if mouth_width > 0:
            mouth_hw_ratio = mouth_height / mouth_width
            mouth_hw_threshold = 0.04
            
            if mouth_hw_ratio < mouth_hw_threshold:
                logger.warning(
                    f"Abnormally flat mouth detected: ratio={mouth_hw_ratio:.4f}"
                )
                raise ValidationError(
                    ErrorCode.OCCLUSION_DETECTED,
                    "We detected a mask or obstruction covering your mouth. "
                    "Please upload a photo without face coverings for accurate analysis."
                )
        
        # Strategy 3: Check chin/jaw region for mask coverage
        chin_jaw_z_std = np.std(chin_jaw_points[:, 2])
        
        # Masks covering lower face create depth variance in chin/jaw region
        # Increased threshold from 0.08 to 0.16 to reduce false positives
        z_std_threshold_chin = 0.16
        
        if chin_jaw_z_std > z_std_threshold_chin:
            logger.warning(
                f"Lower face occlusion detected: chin_z_std={chin_jaw_z_std:.4f}"
            )
            raise ValidationError(
                ErrorCode.OCCLUSION_DETECTED,
                "We detected a mask or obstruction covering your lower face. "
                "Please upload a photo without face coverings for accurate analysis."
            )


# Global instance (will be initialized when needed)
face_detector: Optional[FaceDetector] = None


def get_face_detector() -> FaceDetector:
    """
    Get or create global face detector instance.
    
    Returns:
        FaceDetector instance
    """
    global face_detector
    if face_detector is None:
        face_detector = FaceDetector()
    return face_detector
