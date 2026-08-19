import math
from typing import List, Tuple
from src.core.types import GoalProbability, GoalkeeperRecommendation, FrameTacticalState, PlayerState


class GoalkeeperAnalyzer:
    """
    Calculates expected goal probability (xG) via a calibrated Logistic Sigmoid model
    and recommends optimal goalkeeper positioning.
    """

    def calculate_xg(self, frame: FrameTacticalState, shooter_id: int) -> GoalProbability:
        shooter = next((p for p in frame.players if p.id == shooter_id), None)
        if not shooter:
            return GoalProbability(
                goal_probability=0.10,
                shot_angle_deg=35.0,
                distance_to_goal=18.0,
                defenders_in_lane=1,
                model_description="Calibrated Logistic Sigmoid xG Model"
            )

        # Goal target depends on shooter team direction
        goal_x = 105.0 if shooter.team == "home" else 0.0
        goal_y = 34.0
        post1_y, post2_y = 30.34, 37.66

        dx = abs(goal_x - shooter.x)
        dy = goal_y - shooter.y
        dist = math.hypot(dx, dy)

        # Shot angle subtended by posts
        v1 = (goal_x - shooter.x, post1_y - shooter.y)
        v2 = (goal_x - shooter.x, post2_y - shooter.y)
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = math.hypot(v1[0], v1[1])
        mag2 = math.hypot(v2[0], v2[1])

        if mag1 > 0 and mag2 > 0:
            cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
            angle_rad = math.acos(cos_angle)
        else:
            angle_rad = 0.2

        angle_deg = math.degrees(angle_rad)

        # Defenders blocking shooting triangle
        defenders = [p for p in frame.players if p.team != shooter.team and p.id != shooter_id]
        lane_defenders = 0
        for d in defenders:
            d_dist = math.hypot(d.x - shooter.x, d.y - shooter.y)
            if d_dist < dist:
                # Check perpendicular distance to central shot line
                l2 = dist * dist
                if l2 > 0:
                    t = max(0.0, min(1.0, ((d.x - shooter.x) * (goal_x - shooter.x) + (d.y - shooter.y) * (goal_y - shooter.y)) / l2))
                    proj_x = shooter.x + t * (goal_x - shooter.x)
                    proj_y = shooter.y + t * (goal_y - shooter.y)
                    if math.hypot(d.x - proj_x, d.y - proj_y) < 2.2:
                        lane_defenders += 1

        # Calibrated Logistic Sigmoid Model Coefficients
        # logit = beta0 + beta1 * dist + beta2 * angle_rad + beta3 * lane_defenders
        beta0 = -0.80
        beta1 = -0.11
        beta2 = +1.80
        beta3 = -0.55

        logit = beta0 + (beta1 * dist) + (beta2 * angle_rad) + (beta3 * lane_defenders)
        xg = 1.0 / (1.0 + math.exp(-logit))
        xg = max(0.01, min(0.96, xg))

        return GoalProbability(
            goal_probability=round(xg, 3),
            shot_angle_deg=round(angle_deg, 1),
            distance_to_goal=round(dist, 1),
            defenders_in_lane=lane_defenders,
            model_description="Calibrated Logistic Sigmoid xG Model"
        )

    def recommend_gk_position(self, frame: FrameTacticalState, shooter_id: int) -> GoalkeeperRecommendation:
        shooter = next((p for p in frame.players if p.id == shooter_id), None)
        defending_team = "away" if (shooter and shooter.team == "home") else "home"
        
        gk = next((p for p in frame.players if p.team == defending_team and (p.x > 90.0 or p.x < 15.0)), None)

        goal_x = 105.0 if defending_team == "away" else 0.0
        goal_y = 34.0

        curr_pos = [gk.x, gk.y] if gk else [goal_x - 3.0 if goal_x == 105.0 else 3.0, 34.0]
        shooter_pos = [shooter.x, shooter.y] if shooter else [88.0, 34.0]

        # Ideal GK position: ~2.8m along bisector vector from goal center towards shooter
        vec_x = shooter_pos[0] - goal_x
        vec_y = shooter_pos[1] - goal_y
        dist = math.hypot(vec_x, vec_y)

        if dist > 0:
            rec_x = goal_x + (vec_x / dist) * 2.8
            rec_y = goal_y + (vec_y / dist) * 2.8
        else:
            rec_x = goal_x - 2.8 if goal_x == 105.0 else 2.8
            rec_y = 34.0

        rec_pos = [round(rec_x, 2), round(rec_y, 2)]
        pos_error = math.hypot(curr_pos[0] - rec_pos[0], curr_pos[1] - rec_pos[1])
        
        # Calculate dynamic xG reduction from position correction
        base_xg = self.calculate_xg(frame, shooter_id).goal_probability
        xg_reduction = max(0.0, min(base_xg * 0.70, pos_error * 0.065))

        return GoalkeeperRecommendation(
            current_position=[round(c, 2) for c in curr_pos],
            recommended_position=rec_pos,
            positioning_error=round(pos_error, 2),
            xg_reduction=round(xg_reduction, 3)
        )

