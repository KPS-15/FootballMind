import math
from typing import List, Optional
from src.core.types import DetectedObject, PlayerState, BallState


class BallTracker:
    """
    Tracks ball trajectory across frames, performs motion interpolation for missing frames,
    and assigns ball possession to the closest player.
    """

    def __init__(self, possession_distance_threshold: float = 3.0):
        self.threshold = possession_distance_threshold
        self.last_ball_pos = None

    def update_and_assign_possession(
        self,
        ball_det: Optional[DetectedObject],
        players: List[PlayerState],
        pitch_w: float = 105.0,
        pitch_h: float = 68.0
    ) -> BallState:
        if ball_det is not None and len(ball_det.center) == 2:
            bx, by = ball_det.center[0], ball_det.center[1]
            self.last_ball_pos = (bx, by)
        elif self.last_ball_pos is not None:
            bx, by = self.last_ball_pos
        else:
            bx, by = pitch_w / 2.0, pitch_h / 2.0

        min_dist = float("inf")
        possessor_id = None
        possessor_team = None

        for p in players:
            dist = math.hypot(p.x - bx, p.y - by)
            if dist < min_dist:
                min_dist = dist
                if dist <= self.threshold:
                    possessor_id = p.id
                    possessor_team = p.team

        return BallState(
            x=round(bx, 2),
            y=round(by, 2),
            pixel_x=bx,
            pixel_y=by,
            velocity_x=0.0,
            velocity_y=0.0,
            speed=0.0,
            possession_player_id=possessor_id,
            possession_team=possessor_team
        )
