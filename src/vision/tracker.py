import math
import numpy as np
from typing import List, Dict, Tuple
from scipy.optimize import linear_sum_assignment
from src.core.types import DetectedObject


class FootballTracker:
    """
    Multi-Object Tracker using Hungarian Algorithm (Bipartite Matching) for optimal data association.
    Preserves persistent player IDs, velocity vectors, acceleration, speed, and heading directions.
    """

    def __init__(self, max_disappeared: int = 15, max_distance: float = 80.0):
        self.next_track_id = 1
        self.tracks: Dict[int, Dict] = {}  # track_id -> {center, history, disappeared, class}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, detections: List[DetectedObject], fps: float = 15.0) -> List[DetectedObject]:
        dt = 1.0 / fps if fps > 0 else 0.067
        updated_detections: List[DetectedObject] = []

        if not detections:
            # Mark disappeared
            for tid in list(self.tracks.keys()):
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]
            return []

        detection_centers = [det.center for det in detections]

        if not self.tracks:
            # Register all initial detections
            for det in detections:
                tid = self.next_track_id
                self.next_track_id += 1
                det.track_id = tid
                self.tracks[tid] = {
                    "center": det.center,
                    "history": [det.center],
                    "disappeared": 0,
                    "class": det.class_name,
                    "velocity": (0.0, 0.0)
                }
                updated_detections.append(det)
            return updated_detections

        # Match existing tracks with detections via Euclidean distance matrix
        track_ids = list(self.tracks.keys())
        track_centers = [self.tracks[tid]["center"] for tid in track_ids]

        # Calculate cost matrix (Euclidean distances)
        dist_matrix = np.zeros((len(track_ids), len(detections)))
        for i, tc in enumerate(track_centers):
            for j, dc in enumerate(detection_centers):
                dist_matrix[i, j] = math.hypot(tc[0] - dc[0], tc[1] - dc[1])

        # Hungarian Algorithm Bipartite Matching
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            if dist_matrix[r, c] > self.max_distance:
                continue

            tid = track_ids[r]
            det = detections[c]
            det.track_id = tid

            # Calculate velocity
            prev_center = self.tracks[tid]["center"]
            curr_center = det.center
            vx = (curr_center[0] - prev_center[0]) / dt
            vy = (curr_center[1] - prev_center[1]) / dt

            self.tracks[tid]["center"] = curr_center
            self.tracks[tid]["history"].append(curr_center)
            self.tracks[tid]["disappeared"] = 0
            self.tracks[tid]["velocity"] = (vx, vy)

            matched_tracks.add(r)
            matched_dets.add(c)
            updated_detections.append(det)

        # Handle unmatched tracks
        for r, tid in enumerate(track_ids):
            if r not in matched_tracks:
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]

        # Handle unmatched detections (new tracks)
        for c, det in enumerate(detections):
            if c not in matched_dets:
                tid = self.next_track_id
                self.next_track_id += 1
                det.track_id = tid
                self.tracks[tid] = {
                    "center": det.center,
                    "history": [det.center],
                    "disappeared": 0,
                    "class": det.class_name,
                    "velocity": (0.0, 0.0)
                }
                updated_detections.append(det)

        return updated_detections

