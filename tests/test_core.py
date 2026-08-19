import pytest
from src.core.types import BoundingBox, DetectedObject, PlayerState, BallState, FrameTacticalState
from src.core.state_encoder import FootballStateEncoder
from src.data.synthetic_generator import SyntheticMatchGenerator


def test_bounding_box_center():
    box = BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=60.0)
    assert box.center == (20.0, 40.0)


def test_detected_object_schema():
    det = DetectedObject(
        track_id=1,
        class_name="player",
        confidence=0.92,
        bbox=[10.0, 20.0, 30.0, 60.0],
        center=[20.0, 40.0]
    )
    assert det.track_id == 1
    assert det.class_name == "player"


def test_state_encoder():
    gen = SyntheticMatchGenerator()
    frames = gen.generate_sequence(num_frames=10)
    encoder = FootballStateEncoder()

    encoded = encoder.encode_frame(frames[0])
    assert "player_features" in encoded
    assert len(encoded["player_features"]) == 22

    vec = encoder.extract_feature_vector(frames[0], player_id=1)
    assert len(vec) == 16


def test_tracking_exporter(tmp_path):
    from src.tracking.exporter import TrackingExporter
    gen = SyntheticMatchGenerator()
    frames = gen.generate_sequence(num_frames=5)

    jsonl_path = tmp_path / "tracking.jsonl"
    parquet_path = tmp_path / "tracking.parquet"

    TrackingExporter.export_jsonl(frames, str(jsonl_path))
    TrackingExporter.export_parquet(frames, str(parquet_path))

    assert jsonl_path.exists()
    assert parquet_path.exists()
    assert jsonl_path.stat().st_size > 0
    assert parquet_path.stat().st_size > 0

