import pytest
import cv2
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient
from backend.main import app
from src.core.types import VideoMetadata, FrameTacticalState
from src.tactical.video_analyzer import VideoTacticalAnalyzer
from src.vision.video_processor import FootballVideoProcessor
from src.data.synthetic_generator import SyntheticMatchGenerator

client = TestClient(app)


def create_dummy_video(video_path: Path, width: int = 640, height: int = 480, fps: int = 30, num_frames: int = 30):
    video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw dummy players (red and blue circles)
        cv2.circle(frame, (100 + i*2, 200), 10, (0, 0, 255), -1)  # Red home
        cv2.circle(frame, (300, 200 + i), 10, (255, 0, 0), -1)    # Blue away
        cv2.circle(frame, (200 + i, 200), 5, (0, 255, 255), -1)   # Ball
        out.write(frame)

    out.release()


def test_team_classifier_clustering_and_smoothing():
    from src.vision.team_classifier import TeamClassifier
    from src.core.types import DetectedObject

    classifier = TeamClassifier()

    # Synthetic image frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw Red Team A torso
    frame[50:100, 100:150] = (0, 0, 255)
    # Draw Blue Team B torso
    frame[50:100, 300:350] = (255, 0, 0)

    det_a = DetectedObject(track_id=39, class_name="player", confidence=0.9, bbox=[100, 50, 150, 150], center=[125, 100])
    det_b = DetectedObject(track_id=41, class_name="player", confidence=0.9, bbox=[300, 50, 350, 150], center=[325, 100])

    # Calibrate
    stats = classifier.calibrate_teams([frame], [[det_a, det_b]])
    assert "TEAM A" in stats and "TEAM B" in stats

    # Classify across 3 consecutive frames
    for _ in range(3):
        res = classifier.classify_frame_teams(frame, [det_a, det_b])

    assert det_a.team in ["TEAM A", "TEAM B"]
    assert det_b.team in ["TEAM A", "TEAM B"]
    assert det_a.team != det_b.team
    assert det_a.team_confidence >= 0.50



def test_video_tactical_analyzer_calculation():
    gen = SyntheticMatchGenerator()
    frames = gen.generate_sequence(num_frames=20)

    metadata = VideoMetadata(
        filename="test_attack.mp4",
        file_path="uploads/test_attack.mp4",
        file_size_mb=12.4,
        duration_sec=18.4,
        resolution=[1920, 1080],
        fps=30.0,
        status="uploaded"
    )

    analyzer = VideoTacticalAnalyzer()
    res = analyzer.analyze_video_sequence(frames, metadata)

    assert res.sequence_type in ["ATTACKING", "DEFENSIVE", "MIXED"]
    assert res.ball_carrier_id is not None
    assert res.best_pass is not None
    assert "from" in res.best_pass and "to" in res.best_pass
    assert res.open_space_channel != ""
    assert res.defensive_danger_score >= 0 and res.defensive_danger_score <= 100
    assert len(res.defensive_recommendations) > 0
    assert len(res.events) > 0
    assert "FOOTBALLMIND VIDEO ANALYSIS" in res.summary_text


def test_api_video_upload_and_analyze_routes(tmp_path):
    video_file = tmp_path / "sample_match.mp4"
    create_dummy_video(video_file, width=640, height=480, fps=15, num_frames=15)

    # 1. Upload Video
    with open(video_file, "rb") as f:
        response = client.post("/api/video/upload", files={"file": ("sample_match.mp4", f, "video/mp4")})

    assert response.status_code == 200
    upload_data = response.json()
    assert upload_data["status"] == "success"
    assert "video_metadata" in upload_data
    file_path = upload_data["video_metadata"]["file_path"]

    # 2. Analyze Video
    response_analyze = client.post("/api/video/analyze", json={
        "file_path": file_path,
        "max_frames": 20
    })

    assert response_analyze.status_code == 200
    analysis_data = response_analyze.json()
    assert "sequence_type" in analysis_data
    assert "best_pass" in analysis_data
    assert "defensive_danger_score" in analysis_data
    assert "download_url" in analysis_data
    assert "carrier_label" in analysis_data
    assert "best_pass_label" in analysis_data

    # 3. Test Download Route
    download_url = analysis_data["download_url"]
    assert download_url is not None
    response_download = client.get(download_url)
    assert response_download.status_code == 200
    assert response_download.headers["content-type"] == "video/mp4"
    assert len(response_download.content) > 0

