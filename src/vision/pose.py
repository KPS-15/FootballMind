import math
import numpy as np
from typing import Dict, Tuple, List


class PlayerPoseEstimator:
    """
    Estimates keypoint pose and body orientation angle (0..360 deg) for detected players.
    """

    def __init__(self):
        pass

    def estimate_orientation(self, bbox: List[float], velocity: Tuple[float, float]) -> float:
        """
        Estimates body orientation from movement velocity and bounding box aspect ratio.
        """
        vx, vy = velocity
        speed = math.hypot(vx, vy)

        if speed > 0.5:
            # Heading direction derived from velocity vector
            heading = math.degrees(math.atan2(vy, vx)) % 360.0
            return round(heading, 1)

        # Default forward facing towards attacking goal (0 deg = east)
        return 0.0
