import cv2
import numpy as np
from typing import Tuple, Optional, List


class HomographyMapper:
    """
    Computes camera homography transformation matrix mapping pixel coordinates (x_img, y_img)
    to standard top-down pitch coordinates (x_pitch, y_pitch) [105m x 68m].
    """

    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.H = None

    def compute_matrix(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> bool:
        """
        Computes 3x3 Homography Matrix H from matching source pixel points and destination pitch points.
        """
        if len(src_pts) >= 4 and len(dst_pts) >= 4:
            H, mask = cv2.findHomography(src_pts.astype(np.float32), dst_pts.astype(np.float32), cv2.RANSAC, 5.0)
            if H is not None:
                self.H = H
                return True
        return False

    def transform_point(self, px: float, py: float, img_w: float = 1920.0, img_h: float = 1080.0) -> Tuple[float, float]:
        """
        Transforms pixel point (px, py) to (x_pitch, y_pitch). Fallback uses normalized affine bounding box mapping.
        """
        if self.H is not None:
            pt = np.array([[[px, py]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, self.H)
            tx, ty = transformed[0][0][0], transformed[0][0][1]
            # Clamp to pitch bounds
            tx = max(0.0, min(self.pitch_length, float(tx)))
            ty = max(0.0, min(self.pitch_width, float(ty)))
            return tx, ty

        # Fallback ratio mapping
        pitch_x = (px / max(1.0, img_w)) * self.pitch_length
        pitch_y = (py / max(1.0, img_h)) * self.pitch_width
        return max(0.0, min(self.pitch_length, pitch_x)), max(0.0, min(self.pitch_width, pitch_y))
