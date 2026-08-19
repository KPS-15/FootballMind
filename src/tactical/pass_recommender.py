import math
from typing import List
from src.core.types import PassRecommendation, FrameTacticalState
from src.models.pass_predictor import PassPredictor


class PassRecommender:
    """
    Evaluates and ranks candidate pass options for the ball carrier.
    PASS SCORE = Success Probability * Attacking Advantage * Space Created
    """

    def __init__(self):
        self.pass_predictor = PassPredictor()

    def recommend_passes(self, frame: FrameTacticalState, passer_id: int) -> List[PassRecommendation]:
        passer = next((p for p in frame.players if p.id == passer_id), None)
        if not passer:
            return []

        teammates = [t for t in frame.players if t.team == passer.team and t.id != passer.id]
        opponents = [o for o in frame.players if o.team != passer.team]

        recommendations: List[PassRecommendation] = []
        target_goal_x = 105.0 if passer.team == "home" else 0.0

        for tm in teammates:
            # 1. Success Probability
            pass_pred = self.pass_predictor.predict_pass(frame, passer_id)
            success_prob = pass_pred.probability if pass_pred.receiver_id == tm.id else 0.70

            # 2. Attacking Advantage (forward progression toward goal)
            passer_dist_to_goal = math.hypot(passer.x - target_goal_x, passer.y - 34.0)
            tm_dist_to_goal = math.hypot(tm.x - target_goal_x, tm.y - 34.0)
            progression = (passer_dist_to_goal - tm_dist_to_goal) / 50.0
            attacking_adv = max(0.1, min(1.0, 0.5 + progression))

            # 3. Space Created (nearest opponent distance to receiver)
            nearest_opp_dist = min([math.hypot(tm.x - o.x, tm.y - o.y) for o in opponents], default=15.0)
            space_created = max(0.1, min(1.0, nearest_opp_dist / 12.0))

            # Final Score calculation
            score = round(success_prob * attacking_adv * space_created, 2)

            recommendations.append(PassRecommendation(
                receiver_id=tm.id,
                score=score,
                success_probability=round(success_prob, 2),
                attacking_advantage=round(attacking_adv, 2),
                space_created=round(space_created, 2),
                start_pos=[round(passer.x, 2), round(passer.y, 2)],
                end_pos=[round(tm.x, 2), round(tm.y, 2)]
            ))

        # Sort recommendations descending by score
        recommendations.sort(key=lambda r: r.score, reverse=True)
        return recommendations[:5]
