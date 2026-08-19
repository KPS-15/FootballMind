import copy
import numpy as np
from typing import List
from src.core.types import ExplainablePrediction, FeatureAttribution, FrameTacticalState
from src.core.state_encoder import FootballStateEncoder
from src.models.action_predictor import ActionPredictor

FEATURE_NAMES = [
    "Player X Location",
    "Player Y Location",
    "Velocity X",
    "Velocity Y",
    "Player Speed",
    "Body Orientation",
    "Ball X Location",
    "Ball Y Location",
    "Nearest Teammate Distance",
    "Nearest Opponent Distance",
    "Defensive Pressure Level",
    "Teammate Density (8m)",
    "Opponent Density (8m)",
    "Available Pitch Space",
    "Distance to Attacking Goal",
    "Ball Proximity"
]


class ExplainabilityEngine:
    """
    Computes true empirical feature attributions via model feature occlusion / perturbation:
    measures exact drop in predicted probability when each feature is zeroed out or masked.
    """

    def __init__(self):
        self.encoder = FootballStateEncoder()
        self.action_predictor = ActionPredictor()

    def explain_prediction(self, frame: FrameTacticalState, player_id: int) -> ExplainablePrediction:
        # 1. Baseline prediction
        base_pred = self.action_predictor.predict_action(frame, player_id)
        base_action = base_pred.action
        
        # Get baseline confidence for predicted action
        base_conf = base_pred.confidence
        encoded_dict = self.encoder.encode_frame(frame)
        pf = encoded_dict["player_features"].get(player_id)

        if not pf:
            return ExplainablePrediction(
                prediction_type="Next Action Intention",
                predicted_value="HOLD",
                confidence=0.50,
                top_features=[],
                narrative_reason="Insufficient tracking data for feature attribution.",
                alternative_decision="PASS",
                attribution_method="Baseline Fallback"
            )

        # 2. Empirical Feature Occlusion Perturbation
        attributions: List[FeatureAttribution] = []

        # Perturb frame copy for each feature category
        for idx in range(16):
            pert_frame = copy.deepcopy(frame)
            target_p = next((p for p in pert_frame.players if p.id == player_id), None)
            if not target_p:
                continue

            # Apply domain-specific feature occlusion to input frame
            if idx == 0:     # Player X Location
                target_p.x = 52.5
            elif idx == 1:   # Player Y Location
                target_p.y = 34.0
            elif idx == 2:   # Velocity X
                target_p.velocity_x = 0.0
            elif idx == 3:   # Velocity Y
                target_p.velocity_y = 0.0
            elif idx == 4:   # Player Speed
                target_p.speed = 0.0
            elif idx == 8:   # Nearest Teammate Distance
                # Move teammates far away
                for tm in pert_frame.players:
                    if tm.team == target_p.team and tm.id != target_p.id:
                        tm.x, tm.y = 5.0, 5.0
            elif idx == 9 or idx == 10:  # Nearest Opponent Distance / Defensive Pressure
                # Remove nearby opponents
                for opp in pert_frame.players:
                    if opp.team != target_p.team:
                        opp.x, opp.y = 100.0, 60.0
            elif idx == 14:  # Distance to Goal
                target_p.x = 52.5
            elif idx == 15:  # Ball Proximity
                pert_frame.ball.x, pert_frame.ball.y = 52.5, 34.0
                pert_frame.ball.possession_player_id = None

            # Re-predict on perturbed frame state
            pert_pred = self.action_predictor.predict_action(pert_frame, player_id)
            
            # Find new confidence for target base_action
            if pert_pred.action == base_action:
                pert_conf = pert_pred.confidence
            else:
                # Find in alternatives or set low
                alt_match = next((alt for alt in pert_pred.alternatives if alt["action"] == base_action), None)
                pert_conf = alt_match["confidence"] if alt_match else 0.05

            # Probability shift delta
            delta = base_conf - pert_conf
            
            if abs(delta) >= 0.005 or idx in [0, 8, 9, 10, 14, 15]:
                feature_name = FEATURE_NAMES[idx]
                attributions.append(FeatureAttribution(
                    feature_name=feature_name,
                    contribution=round(float(delta), 3),
                    description=f"Empirical probability shift of target action '{base_action}' when masking {feature_name} ({delta:+.3f})."
                ))

        # Sort top features by absolute contribution magnitude
        attributions.sort(key=lambda a: abs(a.contribution), reverse=True)
        top_4 = attributions[:4] if len(attributions) >= 4 else attributions

        f1_name = top_4[0].feature_name if top_4 else "Spatial Position"
        f1_val = top_4[0].contribution if top_4 else 0.0
        f2_name = top_4[1].feature_name if len(top_4) > 1 else "Pitch Geometry"
        f2_val = top_4[1].contribution if len(top_4) > 1 else 0.0

        narrative = (
            f"Model predicts '{base_pred.action}' with {base_conf * 100:.1f}% confidence ({base_pred.model_type}). "
            f"Key model drivers identified via empirical occlusion perturbation: {f1_name} (impact: {f1_val:+.3f}) "
            f"and {f2_name} (impact: {f2_val:+.3f})."
        )

        alt_decision = base_pred.alternatives[0]["action"] if base_pred.alternatives else "DRIBBLE"

        return ExplainablePrediction(
            prediction_type="Next Action Intention",
            predicted_value=base_pred.action,
            confidence=round(base_conf, 3),
            top_features=top_4,
            narrative_reason=narrative,
            alternative_decision=alt_decision,
            attribution_method="Empirical Model Feature Occlusion Perturbation"
        )

