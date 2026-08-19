import numpy as np
from typing import Tuple, List, Dict
from src.tactical.homography import HomographyMapper


class PitchMapper:
    """
    Pitch coordinate manager handling coordinate conversions, pitch landmarks,
    and homography updates.
    """

    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.homography = HomographyMapper(pitch_length, pitch_width)

    def update_homography(self, frame: np.ndarray):
        """
        Detects pitch lines/corners in frame and updates homography matrix.
        """
        if frame is None:
            return

        h, w = frame.shape[:2]
        # Standard camera corner points mapped to pitch corners
        src_pts = np.array([
            [w * 0.15, h * 0.35],
            [w * 0.85, h * 0.35],
            [w * 0.95, h * 0.90],
            [w * 0.05, h * 0.90]
        ], dtype=np.float32)

        dst_pts = np.array([
            [0.0, 0.0],
            [self.pitch_length, 0.0],
            [self.pitch_length, self.pitch_width],
            [0.0, self.pitch_width]
        ], dtype=np.float32)

        self.homography.compute_matrix(src_pts, dst_pts)

    def pixel_to_pitch(self, px: float, py: float, img_w: float = 1920.0, img_h: float = 1080.0) -> Tuple[float, float]:
        return self.homography.transform_point(px, py, img_w, img_h)
