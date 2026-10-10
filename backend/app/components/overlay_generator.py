"""
Overlay Generator Component

Generates visual overlay data for rendering golden ratio guides on analyzed faces.
"""

import numpy as np
from typing import List, Tuple, Dict, Any

from app.models.internal import Landmarks, OverlayDataInternal
from app.utils.logger import get_logger
from app.config import settings

logger = get_logger(__name__)


class OverlayGenerator:
    """
    Generates data for visual overlays including landmarks and golden ratio guides.
    """
    
    PHI = settings.phi
    
    def generate(self, landmarks: Landmarks) -> OverlayDataInternal:
        """
        Generate complete overlay data for visualization.
        
        Args:
            landmarks: 468 3D facial landmarks
        
        Returns:
            OverlayDataInternal with landmarks and golden ratio guides
        """
        # Extract 2D landmarks (drop z-coordinate)
        landmarks_2d = landmarks.get_2d_landmarks()
        
        # Calculate golden ratio guides
        midline_x = self._calculate_midline(landmarks_2d)
        horizontal_thirds_y = self._calculate_horizontal_thirds(landmarks_2d)
        phi_spiral_center = self._calculate_phi_spiral_center(landmarks_2d)
        
        logger.info(
            f"Overlay data generated: midline_x={midline_x:.1f}, "
            f"thirds={[f'{y:.1f}' for y in horizontal_thirds_y]}, "
            f"spiral_center=({phi_spiral_center[0]:.1f}, {phi_spiral_center[1]:.1f})"
        )
        
        return OverlayDataInternal(
            landmarks_2d=landmarks_2d,
            midline_x=midline_x,
            horizontal_thirds_y=horizontal_thirds_y,
            phi_spiral_center=phi_spiral_center
        )
    
    def calculate_golden_ratio_guides(
        self,
        landmarks_2d: np.ndarray,
        image_width: int,
        image_height: int
    ) -> Dict[str, Any]:
        """
        Calculate golden ratio guide coordinates for overlay visualization.
        
        Computes horizontal thirds, vertical midline, and phi spiral center
        based on facial landmark positions. Requirements 9.2-9.5.
        
        Args:
            landmarks_2d: 468 2D landmarks array of shape (468, 2) with [x, y] coordinates
            image_width: Width of the source image in pixels
            image_height: Height of the source image in pixels
        
        Returns:
            Dictionary with structure:
            {
                'horizontal_thirds': [x1, x2],           # x-coordinates for vertical divisions
                'vertical_midline': y,                   # y-coordinate of horizontal midline
                'phi_spiral': {'cx': x, 'cy': y}        # Center point for phi spiral
            }
        
        Notes:
            - horizontal_thirds divides image width into thirds at 1/3 and 2/3 points
            - vertical_midline is the y-coordinate at image center height
            - phi_spiral center is positioned near face center for spiral overlay
            - All coordinates are in pixel space matching image dimensions
        """
        # Validate input
        if landmarks_2d.shape != (468, 2):
            raise ValueError(
                f"landmarks_2d must have shape (468, 2), got {landmarks_2d.shape}"
            )
        if image_width <= 0 or image_height <= 0:
            raise ValueError(
                f"image_width and image_height must be positive, "
                f"got {image_width}x{image_height}"
            )
        
        # Calculate horizontal thirds (vertical divisions)
        # Divide image width into thirds
        x1 = image_width / 3.0
        x2 = 2.0 * image_width / 3.0
        horizontal_thirds = [float(x1), float(x2)]
        
        # Calculate vertical midline (y-coordinate at center)
        vertical_midline = image_height / 2.0
        
        # Calculate phi spiral center based on face geometry
        # Use face center from landmarks (average of key points)
        left_face = landmarks_2d[234]    # Left face contour
        right_face = landmarks_2d[454]   # Right face contour
        top_face = landmarks_2d[10]      # Forehead top
        bottom_face = landmarks_2d[152]  # Chin bottom
        
        # Center x: average of left and right face
        center_x = (left_face[0] + right_face[0]) / 2.0
        
        # Center y: average of top and bottom face
        center_y = (top_face[1] + bottom_face[1]) / 2.0
        
        phi_spiral = {
            'cx': float(center_x),
            'cy': float(center_y)
        }
        
        return {
            'horizontal_thirds': horizontal_thirds,
            'vertical_midline': float(vertical_midline),
            'phi_spiral': phi_spiral
        }
    
    def _calculate_midline(self, landmarks_2d: np.ndarray) -> float:
        """
        Calculate vertical midline x-coordinate.
        
        Uses average of left and right face contour points.
        
        Args:
            landmarks_2d: 468 2D landmarks
        
        Returns:
            X-coordinate of vertical midline
        """
        # Left and right face points
        left_face = landmarks_2d[234]  # Left face contour
        right_face = landmarks_2d[454]  # Right face contour
        
        midline = (left_face[0] + right_face[0]) / 2.0
        return float(midline)
    
    def _calculate_horizontal_thirds(self, landmarks_2d: np.ndarray) -> List[float]:
        """
        Calculate horizontal division lines based on golden ratio.
        
        Divides face into thirds approximately following φ proportions.
        
        Args:
            landmarks_2d: 468 2D landmarks
        
        Returns:
            List of 3 y-coordinates for horizontal divisions
        """
        # Top and bottom of face
        face_top_y = landmarks_2d[10][1]      # Forehead top
        face_bottom_y = landmarks_2d[152][1]  # Chin bottom
        
        face_height = face_bottom_y - face_top_y
        
        # Divide into thirds (approximate golden ratio divisions)
        # Upper third at ~38.2% (1/φ ≈ 0.618, inverse ≈ 0.382)
        # Middle third at ~61.8%
        third_1_y = face_top_y + (face_height * 0.382)
        third_2_y = face_top_y + (face_height * 0.618)
        third_3_y = face_bottom_y
        
        return [float(third_1_y), float(third_2_y), float(third_3_y)]
    
    def _calculate_phi_spiral_center(
        self,
        landmarks_2d: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate center point for phi spiral overlay.
        
        Places spiral center near nose bridge, which is typically
        the focal point for golden ratio spirals on faces.
        
        Args:
            landmarks_2d: 468 2D landmarks
        
        Returns:
            (x, y) tuple for spiral center
        """
        # Use nose bridge as spiral center
        nose_bridge = landmarks_2d[6]  # Nose bridge landmark
        
        return (float(nose_bridge[0]), float(nose_bridge[1]))


# Global instance
overlay_generator = OverlayGenerator()
