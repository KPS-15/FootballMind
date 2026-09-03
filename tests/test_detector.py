import os
import cv2
import yaml
import pytest
import numpy as np
from pathlib import Path
from src.core.types import DetectedObject, PlayerState, BallState
from src.vision.detector import FootballDetector, load_vision_config
from src.vision.tracker import FootballTracker
from src.vision.ball_tracker import BallTracker
from src.vision.video_processor import FootballVideoProcessor
from src.vision.dataset_validator import DatasetValidator, DatasetValidationError


def test_load_vision_config():
    cfg = load_vision_config("configs/config.yaml")
    assert isinstance(cfg, dict)
    assert cfg.get("yolo_model") == "yolo11m.pt"
    assert cfg.get("confidence_threshold") == 0.35
    assert cfg.get("ball_confidence_threshold") == 0.20
    assert cfg.get("imgsz") == 1280

    # Non-existent config
    cfg_empty = load_vision_config("non_existent_config.yaml")
    assert cfg_empty == {}


def test_detector_initialization_defaults(monkeypatch):
    # Ensure env var doesn't interfere with default test
    monkeypatch.delenv("FOOTBALLMIND_YOLO_MODEL", raising=False)
    monkeypatch.delenv("YOLO_MODEL", raising=False)

    detector = FootballDetector()
    assert detector.model_name == "yolo11m.pt"
    assert detector.conf_thresh == 0.35
    assert detector.ball_conf_thresh == 0.20
    assert detector.imgsz == 1280


def test_detector_env_var_override(monkeypatch):
    monkeypatch.setenv("FOOTBALLMIND_YOLO_MODEL", "custom_yolo11.pt")
    monkeypatch.setenv("FOOTBALLMIND_CONF_THRESH", "0.40")
    monkeypatch.setenv("FOOTBALLMIND_BALL_CONF_THRESH", "0.15")
    monkeypatch.setenv("FOOTBALLMIND_IMGSZ", "640")

    detector = FootballDetector()
    assert detector.model_name == "custom_yolo11.pt"
    assert detector.conf_thresh == 0.40
    assert detector.ball_conf_thresh == 0.15
    assert detector.imgsz == 640


def test_class_name_mapping():
    detector = FootballDetector()

    # 1. Roboflow dictionary mapping
    roboflow_dict = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
    assert detector.map_class_name(0, roboflow_dict) == "ball"
    assert detector.map_class_name(1, roboflow_dict) == "goalkeeper"
    assert detector.map_class_name(2, roboflow_dict) == "player"
    assert detector.map_class_name(3, roboflow_dict) == "referee"

    # 2. Alternative terminology / synonyms list
    names_list = ["Football", "GK", "Official", "Athlete"]
    assert detector.map_class_name(0, names_list) == "ball"
    assert detector.map_class_name(1, names_list) == "goalkeeper"
    assert detector.map_class_name(2, names_list) == "referee"
    assert detector.map_class_name(3, names_list) == "player"

    # 3. COCO dataset indices
    coco_names = {0: "person", 32: "sports ball"}
    assert detector.map_class_name(0, coco_names) == "player"
    assert detector.map_class_name(32, coco_names) == "ball"

    # 4. Unknown class fallback
    unknown_dict = {99: "unknown_object"}
    assert detector.map_class_name(99, unknown_dict) == "unknown_object"
    assert detector.map_class_name(100, unknown_dict) == "player"


def test_dataset_validator_valid_yaml(tmp_path):
    # Create mock dataset directories
    train_dir = tmp_path / "train" / "images"
    val_dir = tmp_path / "val" / "images"
    test_dir = tmp_path / "test" / "images"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    data_yaml = tmp_path / "data.yaml"
    content = {
        "path": str(tmp_path),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"},
        "nc": 4
    }
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.dump(content, f)

    validator = DatasetValidator(str(data_yaml))
    info = validator.validate()

    assert info["valid"] is True
    assert info["num_classes"] == 4
    assert info["has_ball"] is True
    assert info["has_player"] is True
    assert info["has_referee"] is True
    assert info["has_goalkeeper"] is True
    assert len(info["warnings"]) == 0
    assert info["class_mapping"][0] == "ball"


def test_dataset_validator_missing_classes(tmp_path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()

    data_yaml = tmp_path / "data.yaml"
    # Dataset with only player (no ball, referee, goalkeeper)
    content = {
        "path": str(tmp_path),
        "train": "train",
        "val": "val",
        "names": ["player"]
    }
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.dump(content, f)

    validator = DatasetValidator(str(data_yaml))
    info = validator.validate()

    assert info["valid"] is True
    assert info["has_player"] is True
    assert info["has_ball"] is False
    assert any("ball" in w.lower() for w in info["warnings"])


def test_dataset_validator_invalid_and_missing_yaml(tmp_path):
    # 1. Non-existent file
    validator = DatasetValidator(str(tmp_path / "non_existent.yaml"))
    with pytest.raises(DatasetValidationError, match="not found"):
        validator.validate()

    # 2. Missing required 'train' key
    bad_yaml = tmp_path / "bad.yaml"
    with open(bad_yaml, "w", encoding="utf-8") as f:
        yaml.dump({"val": "val", "names": ["player"]}, f)

    validator_bad = DatasetValidator(str(bad_yaml))
    with pytest.raises(DatasetValidationError, match="Missing required 'train'"):
        validator_bad.validate()


def test_small_object_ball_diagnostics():
    frame_shape = (1080, 1920, 3)

    # 1. Test when no ball is detected
    dets_no_ball = [
        DetectedObject(track_id=1, class_name="player", confidence=0.90, bbox=[100, 100, 150, 250], center=[125, 175])
    ]
    diag_none = FootballDetector.compute_ball_diagnostics(dets_no_ball, frame_shape)
    assert diag_none["ball_detected"] is False
    assert "No ball detected" in diag_none["warning"]

    # 2. Test when tiny ball is detected (<12px)
    dets_tiny_ball = [
        DetectedObject(track_id=2, class_name="ball", confidence=0.75, bbox=[500, 400, 508, 408], center=[504, 404])
    ]
    diag_tiny = FootballDetector.compute_ball_diagnostics(dets_tiny_ball, frame_shape)
    assert diag_tiny["ball_detected"] is True
    assert diag_tiny["is_small_object"] is True
    assert diag_tiny["width_px"] == 8.0
    assert diag_tiny["height_px"] == 8.0
    assert "High resolution" in diag_tiny["warning"]

    # 3. Test when normal-sized ball is detected
    dets_normal_ball = [
        DetectedObject(track_id=3, class_name="ball", confidence=0.88, bbox=[500, 400, 530, 430], center=[515, 415])
    ]
    diag_normal = FootballDetector.compute_ball_diagnostics(dets_normal_ball, frame_shape)
    assert diag_normal["ball_detected"] is True
    assert diag_normal["is_small_object"] is False
    assert diag_normal["warning"] is None


def test_image_and_batch_inference(tmp_path):
    detector = FootballDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. detect_frame
    dets = detector.detect_frame(frame)
    assert isinstance(dets, list)

    # 2. detect_image with file path
    img_path = tmp_path / "test_frame.jpg"
    cv2.imwrite(str(img_path), frame)
    dets_img = detector.detect_image(img_path)
    assert isinstance(dets_img, list)

    # 3. detect_batch
    batch_dets = detector.detect_batch([frame, frame])
    assert len(batch_dets) == 2
    assert isinstance(batch_dets[0], list)


def test_ball_tracker_and_tracker_integration():
    tracker = FootballTracker()
    ball_tracker = BallTracker(possession_distance_threshold=10.0)

    # Initial frame detections
    det_p1 = DetectedObject(track_id=1, class_name="player", confidence=0.9, bbox=[100, 100, 140, 200], center=[120.0, 150.0])
    det_ball = DetectedObject(track_id=2, class_name="ball", confidence=0.85, bbox=[122, 148, 130, 156], center=[126.0, 152.0])

    # 1. Multi-object tracker
    tracked = tracker.update([det_p1, det_ball])
    assert len(tracked) == 2
    assert tracked[0].track_id is not None

    # 2. Ball possession tracker
    player_state = PlayerState(
        id=tracked[0].track_id,
        team="home",
        team_confidence=0.95,
        x=120.0,
        y=150.0,
        pixel_x=120.0,
        pixel_y=150.0,
        velocity_x=0.0,
        velocity_y=0.0,
        speed=0.0,
        direction=0.0,
        acceleration=0.0,
        body_orientation=0.0,
        ball_distance=2.83,
        possession_probability=0.9
    )

    ball_state = ball_tracker.update_and_assign_possession(tracked[1], [player_state])
    assert isinstance(ball_state, BallState)
    assert ball_state.possession_player_id == player_state.id
    assert ball_state.possession_team == "home"
    assert ball_state.pixel_x == 126.0
    assert ball_state.pixel_y == 152.0


def test_video_processor_with_yolo11(tmp_path):
    video_path = tmp_path / "dummy_match.mp4"
    output_video_path = tmp_path / "annotated_dummy.mp4"

    # Create 10-frame dummy video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 15.0, (640, 480))
    for i in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (100 + i*5, 200), 12, (0, 0, 255), -1)
        cv2.circle(frame, (300, 200 + i*2), 12, (255, 0, 0), -1)
        cv2.circle(frame, (200 + i*3, 200), 6, (0, 255, 255), -1)
        out.write(frame)
    out.release()

    processor = FootballVideoProcessor(model_name="yolo11m.pt")
    assert processor.detector.model_name == "yolo11m.pt"

    # Process and annotate
    frames = processor.process_and_annotate_video(str(video_path), str(output_video_path), max_frames=5)
    assert len(frames) == 5
    assert output_video_path.exists()
    assert output_video_path.stat().st_size > 0


def test_roboflow_workflow_client_parsing():
    from src.vision.roboflow_client import RoboflowWorkflowClient

    client = RoboflowWorkflowClient(api_key="test_key_dummy")
    assert client.workspace_name == "k-p-shohil"
    assert client.workflow_id == "general-segmentation-api"

    # 1. Mock workflow dictionary output format
    mock_workflow_output = {
        "output": [
            {"class": "ball", "confidence": 0.92, "x": 320.0, "y": 240.0, "width": 16.0, "height": 16.0},
            {"class": "player", "confidence": 0.88, "x": 150.0, "y": 200.0, "width": 40.0, "height": 80.0},
            {"class": "referee", "confidence": 0.81, "x": 400.0, "y": 210.0, "width": 35.0, "height": 75.0},
            {"class": "goalkeeper", "confidence": 0.95, "x": 50.0, "y": 240.0, "width": 45.0, "height": 85.0}
        ]
    }

    dets = client.parse_predictions(mock_workflow_output)
    assert len(dets) == 4
    assert dets[0].class_name == "ball"
    assert dets[0].bbox == [312.0, 232.0, 328.0, 248.0]
    assert dets[0].center == [320.0, 240.0]
    assert dets[0].confidence == 0.92

    assert dets[1].class_name == "player"
    assert dets[2].class_name == "referee"
    assert dets[3].class_name == "goalkeeper"

    # 2. Mock list format
    mock_list_output = [
        {"class_name": "soccer ball", "score": 0.79, "bbox": [100.0, 100.0, 120.0, 120.0]}
    ]
    dets_list = client.parse_predictions(mock_list_output)
    assert len(dets_list) == 1
    assert dets_list[0].class_name == "ball"
    assert dets_list[0].bbox == [100.0, 100.0, 120.0, 120.0]
    assert dets_list[0].center == [110.0, 110.0]

    # 3. Empty input
    assert client.parse_predictions({}) == []
    assert client.parse_predictions(None) == []


def test_roboflow_detector_integration_mock(monkeypatch):
    from src.vision.roboflow_client import RoboflowWorkflowClient

    detector = FootballDetector()
    detector.use_roboflow = True
    detector.roboflow_client = RoboflowWorkflowClient(api_key="test_key")

    mock_dets = [
        DetectedObject(track_id=1, class_name="ball", confidence=0.95, bbox=[10, 10, 25, 25], center=[17.5, 17.5])
    ]

    monkeypatch.setattr(detector.roboflow_client, "detect_frame", lambda frame: mock_dets)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.detect_frame(frame)
    assert len(res) == 1
    assert res[0].class_name == "ball"

