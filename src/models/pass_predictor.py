import math
from typing import List, Optional, Tuple
from src.core.types import ReceiverPrediction, FrameTacticalState, PlayerState
from src.core.state_encoder import FootballStateEncoder


class PassPredictor:
    """
    Predicts intended pass receiver, target landing coordinate [x, y], and pass success probability
    using a dynamic Time-to-Intercept (TTI) kinematic physics model.
    """

    def __init__(self, ball_initial_speed: float = 18.0, drag_coeff: float = 0.04):
        self.encoder = FootballStateEncoder()
        self.v0 = ball_initial_speed  # m/s
        self.mu = drag_coeff           # aerodynamic pitch drag coefficient

    def calculate_ball_travel_time(self, distance: float) -> float:
        """Calculates ball travel time considering exponential drag velocity decay: v(t) = v0 * exp(-mu * t)."""
        term = 1.0 - (self.mu * distance / self.v0)
        if term > 0.05:
            return float(-math.log(term) / self.mu)
        return float(distance / 12.0)  # fallback linear velocity

    def predict_pass(self, frame: FrameTacticalState, passer_id: int) -> ReceiverPrediction:
        passer = next((p for p in frame.players if p.id == passer_id), None)
        if not passer:
            return ReceiverPrediction(
                receiver_id=None,
                probability=0.0,
                target_location=[52.5, 34.0],
                is_baseline_model=True,
                model_type="Time-to-Intercept (TTI) Kinematic Physics Model"
            )

        teammates = [t for t in frame.players if t.team == passer.team and t.id != passer.id]
        opponents = [o for o in frame.players if o.team != passer.team]

        if not teammates:
            return ReceiverPrediction(
                receiver_id=None,
                probability=0.0,
                target_location=[passer.x + 5.0, passer.y],
                is_baseline_model=True,
                model_type="Time-to-Intercept (TTI) Kinematic Physics Model"
            )

        best_receiver = None
        best_prob = 0.0
        best_target = [passer.x + 5.0, passer.y]

        for tm in teammates:
            dist = math.hypot(tm.x - passer.x, tm.y - passer.y)
            if dist > 55.0 or dist < 2.0:
                continue

            t_ball_total = self.calculate_ball_travel_time(dist)

            min_time_margin = 99.0
            l2 = dist * dist

            for opp in opponents:
                if l2 == 0:
                    continue

                # Project opponent onto passing vector line segment
                t_proj_ratio = max(0.0, min(1.0, ((opp.x - passer.x) * (tm.x - passer.x) + (opp.y - passer.y) * (tm.y - passer.y)) / l2))
                proj_x = passer.x + t_proj_ratio * (tm.x - passer.x)
                proj_y = passer.y + t_proj_ratio * (tm.y - passer.y)

                # Distance for opponent to reach projection point
                d_opp = math.hypot(opp.x - proj_x, opp.y - proj_y)

                # Opponent arrival time: 0.35s reaction latency + sprint at 7.0 m/s
                t_opp = 0.35 + (d_opp / 7.0)

                # Ball arrival time at projection point
                t_ball_proj = t_ball_total * t_proj_ratio
                
                margin = t_opp - t_ball_proj
                if margin < min_time_margin:
                    min_time_margin = margin

            # Logistic Sigmoid mapping of Time Margin (positive margin = ball arrives first)
            # sigmoid(3.0 * (min_time_margin - 0.15))
            k = 3.0
            x_shift = min_time_margin - 0.15
            success_prob = 1.0 / (1.0 + math.exp(-k * x_shift))
            success_prob = max(0.02, min(0.98, success_prob))

            if success_prob > best_prob:
                best_prob = success_prob
                best_receiver = tm.id
                best_target = [round(tm.x, 2), round(tm.y, 2)]

        return ReceiverPrediction(
            receiver_id=best_receiver,
            probability=round(best_prob, 3),
            target_location=best_target,
            is_baseline_model=False,
            model_type="Time-to-Intercept (TTI) Kinematic Physics Model"
        )

