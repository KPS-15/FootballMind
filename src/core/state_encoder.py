import math
import numpy as np
from typing import List, Dict, Any, Tuple
from src.core.types import FrameTacticalState, PlayerState, BallState


class FootballStateEncoder:
    """
    Unified state encoder calculating tactical metrics, spatial relations,
    passing lane vectors, pitch control, and numerical feature vectors for ML models.
    """

    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

    def encode_frame(self, frame: FrameTacticalState) -> Dict[str, Any]:
        """
        Calculates spatial relationships, nearest distances, defensive pressure,
        and available pitch space for all players in a frame.
        """
        home_players = [p for p in frame.players if p.team == "home"]
        away_players = [p for p in frame.players if p.team == "away"]

        possessor = next((p for p in frame.players if p.id == frame.ball.possession_player_id), None)
        carrier_team = possessor.team if possessor else "home"

        player_features: Dict[int, Dict[str, Any]] = {}

        for p in frame.players:
            teammates = [t for t in frame.players if t.team == p.team and t.id != p.id]
            opponents = [o for o in frame.players if o.team != p.team]

            # Nearest teammate
            nearest_teammate_dist = min([math.hypot(p.x - t.x, p.y - t.y) for t in teammates], default=99.0)
            
            # Nearest opponent & pressure index
            nearest_opp_dist = min([math.hypot(p.x - o.x, p.y - o.y) for o in opponents], default=99.0)
            defensive_pressure = max(0.0, 1.0 - (nearest_opp_dist / 10.0))  # 1.0 = heavy pressure (<1m)

            # Density around player (within 8m radius)
            tm_density = sum(1 for t in teammates if math.hypot(p.x - t.x, p.y - t.y) <= 8.0)
            opp_density = sum(1 for o in opponents if math.hypot(p.x - o.x, p.y - o.y) <= 8.0)

            # Available pitch space radius
            available_space = min(nearest_opp_dist, 15.0)

            # Distance to ball & distance to attacking goal (105, 34 for Home, 0, 34 for Away)
            goal_x = 105.0 if p.team == "home" else 0.0
            dist_to_goal = math.hypot(p.x - goal_x, p.y - 34.0)
            ball_dist = math.hypot(p.x - frame.ball.x, p.y - frame.ball.y)

            player_features[p.id] = {
                "player_id": p.id,
                "team": p.team,
                "x": p.x,
                "y": p.y,
                "speed": p.speed,
                "nearest_teammate_dist": round(nearest_teammate_dist, 2),
                "nearest_opponent_dist": round(nearest_opp_dist, 2),
                "defensive_pressure": round(defensive_pressure, 2),
                "teammate_density": tm_density,
                "opponent_density": opp_density,
                "available_space": round(available_space, 2),
                "distance_to_goal": round(dist_to_goal, 2),
                "ball_distance": round(ball_dist, 2)
            }

        return {
            "frame_index": frame.frame_index,
            "timestamp": frame.timestamp,
            "ball": frame.ball.model_dump(),
            "attacking_team": carrier_team,
            "player_features": player_features
        }

    def extract_feature_vector(self, frame: FrameTacticalState, player_id: int) -> np.ndarray:
        """
        Extracts a flat numerical feature vector [1 x 16] suitable for ML model inference.
        """
        encoded = self.encode_frame(frame)
        pf = encoded["player_features"].get(player_id)

        if not pf:
            return np.zeros(16, dtype=np.float32)

        p = next((p for p in frame.players if p.id == player_id), None)
        bx, by = frame.ball.x, frame.ball.y

        vec = [
            p.x / 105.0 if p else 0.5,
            p.y / 68.0 if p else 0.5,
            p.velocity_x if p else 0.0,
            p.velocity_y if p else 0.0,
            p.speed / 10.0 if p else 0.0,
            p.direction / 360.0 if p else 0.0,
            bx / 105.0,
            by / 68.0,
            pf["nearest_teammate_dist"] / 50.0,
            pf["nearest_opponent_dist"] / 50.0,
            pf["defensive_pressure"],
            pf["teammate_density"] / 10.0,
            pf["opponent_density"] / 10.0,
            pf["available_space"] / 20.0,
            pf["distance_to_goal"] / 105.0,
            pf["ball_distance"] / 50.0
        ]
        return np.array(vec, dtype=np.float32)
