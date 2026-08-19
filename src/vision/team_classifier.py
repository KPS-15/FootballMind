import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from src.core.types import DetectedObject


class TeamClassifier:
    """
    Jersey-based Team Identification System using Torso Region Cropping,
    Green Pitch Filtering, Hugging Face SigLIP / Deep Visual Feature Extraction,
    K-Means Clustering (TEAM A vs TEAM B), and Track-ID Temporal Smoothing & Confidence Calculation.
    """

    def __init__(self, model_name: str = "google/siglip-base-patch16-224"):
        self.kmeans: Optional[KMeans] = None
        self.is_calibrated: bool = False
        self.cluster_map: Dict[int, str] = {0: "TEAM A", 1: "TEAM B"}
        self.track_team_history: Dict[int, Dict[str, int]] = {}

        # SigLIP Deep Visual Embedder Initialization
        self.processor = None
        self.siglip_model = None
        self.device = "cpu"

        try:
            import torch
            from transformers import AutoProcessor, AutoModel
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.siglip_model = AutoModel.from_pretrained(model_name).to(self.device)
            self.siglip_model.eval()
            print(f"[TeamClassifier] Successfully initialized SigLIP model ({model_name}) on {self.device}.")
        except Exception as e:
            print(f"[TeamClassifier] SigLIP initialization warning ({e}). Falling back to multi-channel color feature extractor.")

    def crop_jersey(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """Crops upper torso region (top 45% of bounding box) avoiding grass pixels."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        player_crop = frame[y1:y2, x1:x2]
        if player_crop.size == 0:
            return None

        # Take upper 45% torso
        torso_h = max(1, int(player_crop.shape[0] * 0.45))
        torso = player_crop[0:torso_h, :]

        if torso.size == 0:
            return None

        # Mask out green grass pitch pixels (35 <= Hue <= 85)
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 30, 30])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Filter out green background
        jersey_bgr = torso[green_mask == 0]
        if len(jersey_bgr) > 15:
            return jersey_bgr
        return torso.reshape(-1, 3)

    def extract_jersey_features(self, jersey_pixels: np.ndarray, crop_img: Optional[np.ndarray] = None) -> np.ndarray:
        """Extracts SigLIP deep visual embedding combined with normalized multi-channel HSV/RGB feature vector."""
        if jersey_pixels is None or len(jersey_pixels) == 0:
            return np.zeros(12, dtype=np.float64)

        # 1. Color & Spatial Statistics
        pixels = jersey_pixels.reshape(-1, 3)
        mean_bgr = np.mean(pixels, axis=0)
        std_bgr = np.std(pixels, axis=0)

        bgr_pixel = np.uint8([[mean_bgr]])
        hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0][0]

        color_features = np.concatenate([
            mean_bgr / 255.0,
            std_bgr / 255.0,
            hsv_pixel / [180.0, 255.0, 255.0],
            [np.median(pixels[:, 0]) / 255.0, np.median(pixels[:, 1]) / 255.0, np.median(pixels[:, 2]) / 255.0]
        ])

        # 2. SigLIP Deep Feature Embedding (if available)
        if self.siglip_model is not None and self.processor is not None and crop_img is not None and crop_img.size > 0:
            try:
                import torch
                from PIL import Image
                rgb_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_img)
                inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    embeds = self.siglip_model.get_image_features(**inputs)
                    embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
                    siglip_vec = embeds.cpu().numpy().flatten()
                
                # Combine SigLIP embedding with color features
                combined = np.concatenate([siglip_vec, color_features])
                return combined.astype(np.float64)
            except Exception as e:
                pass

        return color_features.astype(np.float64)

    def calibrate_teams(self, sample_frames: List[np.ndarray], sample_detections_list: List[List[DetectedObject]]) -> Dict[str, int]:
        """
        Samples player crops across initial video frames to train K-Means (k=2) cluster model.
        Returns validation sample count dictionary.
        """
        features_list = []

        for frame, detections in zip(sample_frames, sample_detections_list):
            if frame is None or not detections:
                continue

            for det in detections:
                if det.class_name == "player":
                    jersey_pixels = self.crop_jersey(frame, det.bbox)
                    if jersey_pixels is not None and len(jersey_pixels) > 5:
                        x1, y1, x2, y2 = [int(v) for v in det.bbox]
                        crop_img = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                        feat = self.extract_jersey_features(jersey_pixels, crop_img)
                        features_list.append(feat)

        stats = {"TEAM A": 0, "TEAM B": 0, "UNKNOWN": 0}

        if len(features_list) >= 2:
            try:
                self.kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                labels = self.kmeans.fit_predict(np.array(features_list, dtype=np.float64))
                self.is_calibrated = True

                stats["TEAM A"] = int(np.sum(labels == 0))
                stats["TEAM B"] = int(np.sum(labels == 1))
                print(f"[TeamClassifier] SigLIP/K-Means Dynamic Calibration Complete: TEAM A ({stats['TEAM A']} samples) | TEAM B ({stats['TEAM B']} samples)")
            except Exception as e:
                print(f"[TeamClassifier] Calibration fallback error: {e}")
                self.is_calibrated = False
        else:
            print("[TeamClassifier] Warning: Insufficient player samples for K-Means calibration. Using default fallback.")

        return stats

    def classify_frame_teams(self, frame: np.ndarray, detections: List[DetectedObject]) -> List[DetectedObject]:
        """Classifies each player's team and applies temporal smoothing and confidence scoring."""
        if frame is None or not detections:
            return detections

        for det in detections:
            # Handle special roles
            if det.class_name == "referee":
                det.team = "REFEREE"
                det.team_confidence = 1.0
                continue
            elif det.class_name == "goalkeeper":
                det.team = "GOALKEEPER"
                det.team_confidence = 1.0
                continue
            elif det.class_name != "player":
                det.team = "TEAM UNKNOWN"
                det.team_confidence = 0.0
                continue

            # Crop jersey and extract feature
            jersey_pixels = self.crop_jersey(frame, det.bbox)
            raw_team = "TEAM A"
            
            if self.is_calibrated and self.kmeans is not None and jersey_pixels is not None:
                try:
                    x1, y1, x2, y2 = [int(v) for v in det.bbox]
                    crop_img = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                    feat = self.extract_jersey_features(jersey_pixels, crop_img).reshape(1, -1).astype(np.float64)
                    cluster_id = int(self.kmeans.predict(feat)[0])
                    raw_team = self.cluster_map.get(cluster_id, "TEAM A")
                except Exception as e:
                    print(f"[TeamClassifier] Frame prediction error: {e}")
                    raw_team = "TEAM A" if det.track_id % 2 == 0 else "TEAM B"
            else:
                raw_team = "TEAM A" if det.track_id % 2 == 0 else "TEAM B"

            # Temporal Smoothing & Track ID Majority Voting
            tid = det.track_id
            if tid not in self.track_team_history:
                self.track_team_history[tid] = {"TEAM A": 0, "TEAM B": 0}

            self.track_team_history[tid][raw_team] = self.track_team_history[tid].get(raw_team, 0) + 1

            cntA = self.track_team_history[tid].get("TEAM A", 0)
            cntB = self.track_team_history[tid].get("TEAM B", 0)
            total = max(1, cntA + cntB)

            dominant_team = "TEAM A" if cntA >= cntB else "TEAM B"
            confidence = round(max(cntA, cntB) / float(total), 2)

            if confidence >= 0.50:
                det.team = dominant_team
                det.team_confidence = confidence
            else:
                det.team = "TEAM UNKNOWN"
                det.team_confidence = confidence

        return detections

    def fit_and_classify(self, frame: np.ndarray, detections: List[DetectedObject]) -> List[DetectedObject]:
        """Legacy compatibility wrapper."""
        if not self.is_calibrated:
            self.calibrate_teams([frame], [detections])
        return self.classify_frame_teams(frame, detections)
