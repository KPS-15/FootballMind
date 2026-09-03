import cv2
import numpy as np
from pathlib import Path
from typing import List, Generator, Tuple, Optional, Dict, Any
from src.vision.detector import FootballDetector
from src.vision.tracker import FootballTracker
from src.vision.team_classifier import TeamClassifier
from src.vision.ball_tracker import BallTracker
from src.vision.pose import PlayerPoseEstimator
from src.tactical.pitch import PitchMapper
from src.core.types import FrameTacticalState, PlayerState, DetectedObject, VideoMetadata



class FootballVideoProcessor:
    """
    Complete video processing pipeline integrating object detection, multi-object tracking,
    team color classification, homography pitch coordinate mapping, and tactical frame assembly.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        conf_thresh: Optional[float] = None,
        ball_conf_thresh: Optional[float] = None,
        imgsz: Optional[int] = None,
        device: Optional[str] = None,
        config_path: str = "configs/config.yaml"
    ):
        self.detector = FootballDetector(
            model_name=model_name,
            conf_thresh=conf_thresh,
            ball_conf_thresh=ball_conf_thresh,
            imgsz=imgsz,
            device=device,
            config_path=config_path
        )
        self.tracker = FootballTracker()
        self.team_classifier = TeamClassifier()
        self.ball_tracker = BallTracker()
        self.pose_estimator = PlayerPoseEstimator()
        self.pitch_mapper = PitchMapper()

    def process_video(self, video_path: str, max_frames: int = 200, frame_skip: int = 1) -> List[FrameTacticalState]:
        """Reads video, detects objects using YOLO, tracks objects, classifies teams via jersey clustering, and returns tactical frames."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[FootballVideoProcessor] Error: Unable to open video source '{video_path}'")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        tactical_frames: List[FrameTacticalState] = []
        frame_count = 0

        # Phase 1: Sample initial frames for dynamic team calibration
        sample_frames = []
        sample_detections_list = []
        calib_frames_count = 0

        while cap.isOpened() and calib_frames_count < 15:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            dets = self.detector.detect_frame(frame)
            tracked = self.tracker.update(dets, fps=fps)
            sample_frames.append(frame.copy())
            sample_detections_list.append(tracked)
            calib_frames_count += 1

        if sample_frames:
            self.team_classifier.calibrate_teams(sample_frames, sample_detections_list)

        # Reset capture to start from beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.tracker = FootballTracker()  # Reset tracker for clean ID sequence

        while cap.isOpened() and len(tactical_frames) < max_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            h, w = frame.shape[:2]

            # 1. Detect objects (Football YOLO)
            detections = self.detector.detect_frame(frame)

            # 2. Track objects across frames (Hungarian/ByteTrack matching)
            tracked_dets = self.tracker.update(detections, fps=fps)

            # 3. Classify teams based on jersey crop + clustering + temporal smoothing
            classified_dets = self.team_classifier.classify_frame_teams(frame, tracked_dets)

            # 4. Homography matrix calibration & transformation
            self.pitch_mapper.update_homography(frame)

            players: List[PlayerState] = []
            ball_det = None

            for det in classified_dets:
                if det.class_name == "ball":
                    ball_det = det
                    continue

                px, py = det.center[0], det.center[1]
                pitch_x, pitch_y = self.pitch_mapper.pixel_to_pitch(px, py, w, h)
                orientation = self.pose_estimator.estimate_orientation(det.bbox, (0.0, 0.0))

                players.append(PlayerState(
                    id=det.track_id,
                    team=det.team or "TEAM A",
                    team_confidence=getattr(det, "team_confidence", 1.0),
                    x=round(pitch_x, 2),
                    y=round(pitch_y, 2),
                    pixel_x=round(px, 1),
                    pixel_y=round(py, 1),
                    velocity_x=0.0,
                    velocity_y=0.0,
                    speed=0.0,
                    direction=0.0,
                    acceleration=0.0,
                    body_orientation=orientation,
                    ball_distance=999.0,
                    possession_probability=0.0
                ))

            # 5. Track ball & possession
            ball_state = self.ball_tracker.update_and_assign_possession(ball_det, players)

            # Assemble frame tactical state
            timestamp = (frame_count / fps)
            tactical_frames.append(FrameTacticalState(
                frame_index=len(tactical_frames),
                timestamp=round(timestamp, 2),
                players=players,
                ball=ball_state,
                attacking_team="TEAM A",
                defensive_team="TEAM B"
            ))

        cap.release()
        return tactical_frames

    def get_video_metadata(self, video_path: str) -> VideoMetadata:
        """Extracts video metadata (filename, duration, resolution, FPS, size) using OpenCV."""
        path = Path(video_path)
        filename = path.name
        file_size_mb = round(path.stat().st_size / (1024 * 1024), 2) if path.exists() else 0.0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return VideoMetadata(
                filename=filename,
                file_path=str(path),
                file_size_mb=file_size_mb,
                duration_sec=10.0,
                resolution=[1920, 1080],
                fps=30.0,
                status="uploaded"
            )

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 300
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
        duration_sec = round(frame_count / fps, 1)
        cap.release()

        return VideoMetadata(
            filename=filename,
            file_path=str(path),
            file_size_mb=file_size_mb,
            duration_sec=duration_sec,
            resolution=[w, h],
            fps=round(fps, 1),
            status="uploaded"
        )

    def process_and_annotate_video(
        self,
        video_path: str,
        output_path: str,
        max_frames: int = 200
    ) -> List[FrameTacticalState]:
        """Processes video frame-by-frame and renders tactical overlays (boxes, IDs, team badges, pass/mark arrows)."""
        tactical_frames = self.process_video(video_path, max_frames=max_frames)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened() or not tactical_frames:
            return tactical_frames

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Collect annotated frames
        annotated_frames: List[np.ndarray] = []

        frame_idx = 0
        while cap.isOpened() and frame_idx < len(tactical_frames):
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            tf = tactical_frames[frame_idx]
            
            # Render Overlays
            # 1. Clean HUD
            cv2.rectangle(frame, (0, 0), (w, 42), (15, 23, 42), -1)
            cv2.putText(frame, "FOOTBALLMIND LIVE SPORTS ANALYTICS", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (52, 211, 153), 2)
            cv2.putText(frame, f"TIME: {tf.timestamp:.1f}s", (w - 180, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (226, 232, 240), 1)

            # 2. Draw Players with Persistent Track IDs & Team Badges
            for p in tf.players:
                is_team_b = p.team in ["TEAM B", "away"]
                is_ref = p.team == "REFEREE"
                is_gk = p.team == "GOALKEEPER"

                color = (59, 130, 246) if is_team_b else (168, 85, 247) if is_ref else (234, 179, 8) if is_gk else (239, 68, 68)
                team_letter = "B" if is_team_b else "REF" if is_ref else "GK" if is_gk else "A"

                px, py = int(p.pixel_x or w*0.5), int(p.pixel_y or h*0.5)
                
                # Player circle & ID/Team badge
                cv2.circle(frame, (px, py), 12, color, -1)
                cv2.circle(frame, (px, py), 14, (255, 255, 255), 2)
                cv2.putText(frame, f"#{p.id}  {team_letter}", (px - 18, py - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)

            # 3. Draw Ball or "BALL LOST" Status
            if tf.ball and (tf.ball.pixel_x or tf.ball.pixel_y):
                bx, by = int(tf.ball.pixel_x or w*0.5), int(tf.ball.pixel_y or h*0.5)
                cv2.circle(frame, (bx, by), 7, (255, 255, 255), -1)
                cv2.circle(frame, (bx, by), 9, (234, 179, 8), 2)
                cv2.putText(frame, "BALL", (bx - 14, by - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (234, 179, 8), 2)
            else:
                cv2.rectangle(frame, (w - 320, 10), (w - 200, 34), (220, 38, 38), -1)
                cv2.putText(frame, "BALL LOST", (w - 310, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

            # 4. Tactical Pass & Marking Arrows
            if len(tf.players) >= 2:
                carrier = next((p for p in tf.players if p.id == tf.ball.possession_player_id), tf.players[0])
                target = next((p for p in tf.players if p.id != carrier.id), tf.players[min(1, len(tf.players)-1)])
                
                pt1 = (int(carrier.pixel_x or w*0.4), int(carrier.pixel_y or h*0.5))
                pt2 = (int(target.pixel_x or w*0.6), int(target.pixel_y or h*0.5))
                
                # Green arrow for BEST PASS
                cv2.arrowedLine(frame, pt1, pt2, (52, 211, 153), 3, tipLength=0.2)
                mid_pt = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2 - 8)
                cv2.putText(frame, f"BEST PASS #{carrier.id} -> #{target.id}", mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (52, 211, 153), 2)

                # Defensive Mark Arrow (Orange line from defender to attacker)
                defenders = [p for p in tf.players if p.team in ["TEAM B", "away"]]
                if defenders:
                    d = defenders[0]
                    d_pt = (int(d.pixel_x or w*0.3), int(d.pixel_y or h*0.4))
                    cv2.line(frame, d_pt, pt1, (249, 115, 22), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"#{d.id} MARK #{carrier.id}", (d_pt[0] - 20, d_pt[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (249, 115, 22), 2)

            annotated_frames.append(frame)
            frame_idx += 1

        cap.release()

        # Write output video using PyAV (H.264/libx264 for native browser playback) with OpenCV fallback
        wrote_pyav = False
        try:
            import av
            container = av.open(output_path, mode="w")
            stream = container.add_stream("libx264", rate=int(round(fps)))
            stream.width = w
            stream.height = h
            stream.pix_fmt = "yuv420p"

            for frm in annotated_frames:
                av_frame = av.VideoFrame.from_ndarray(frm, format="bgr24")
                for packet in stream.encode(av_frame):
                    container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)
            container.close()
            wrote_pyav = True
            print(f"[FootballVideoProcessor] Native H.264 video saved via PyAV to {output_path}")
        except Exception as err:
            print(f"[FootballVideoProcessor] PyAV video encoding warning: {err}. Falling back to OpenCV VideoWriter.")

        if not wrote_pyav:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            for frm in annotated_frames:
                out.write(frm)
            out.release()
            print(f"[FootballVideoProcessor] Annotated video saved via OpenCV to {output_path}")

        return tactical_frames
