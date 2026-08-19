import json
import pandas as pd
from pathlib import Path
from typing import List
from src.core.types import FrameTacticalState


class TrackingExporter:
    """
    Exports structured tracking data to Parquet and JSONL files for analytics and ML training.
    """

    @staticmethod
    def export_jsonl(frames: List[FrameTacticalState], output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for frame in frames:
                f.write(frame.model_dump_json() + "\n")

    @staticmethod
    def export_parquet(frames: List[FrameTacticalState], output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for frame in frames:
            for p in frame.players:
                rows.append({
                    "frame_index": frame.frame_index,
                    "timestamp": frame.timestamp,
                    "player_id": p.id,
                    "team": p.team,
                    "x": p.x,
                    "y": p.y,
                    "velocity_x": p.velocity_x,
                    "velocity_y": p.velocity_y,
                    "speed": p.speed,
                    "direction": p.direction,
                    "acceleration": p.acceleration,
                    "ball_x": frame.ball.x,
                    "ball_y": frame.ball.y,
                    "possession_player_id": frame.ball.possession_player_id
                })
        df = pd.DataFrame(rows)
        df.to_parquet(output_path, index=False)
