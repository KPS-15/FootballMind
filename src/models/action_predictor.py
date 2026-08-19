import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from src.core.types import ActionPrediction, FrameTacticalState
from src.core.state_encoder import FootballStateEncoder
from src.models.temporal_model import FootballTemporalLSTM

ACTIONS = ["PASS", "DRIBBLE", "SHOT", "CROSS", "TACKLE", "CARRY", "HOLD", "CLEARANCE"]


class ActionPredictor:
    """
    Predicts next player action with probabilities, alternative decisions, and confidence scores.
    Uses deep PyTorch temporal LSTM model when trained weights exist, with a calibrated
    multinomial softmax spatial-physics baseline for zero-shot inference.
    """

    def __init__(self, model_path: str = "models/action_predictor.pt"):
        self.encoder = FootballStateEncoder()
        self.nn_model = FootballTemporalLSTM(input_size=16, hidden_size=64, num_actions=8)
        self.has_weights = False
        self.model_path = model_path

        if os.path.exists(model_path):
            try:
                self.nn_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                self.nn_model.eval()
                self.has_weights = True
                print(f"[ActionPredictor] Successfully loaded PyTorch weights from {model_path}")
            except Exception as e:
                print(f"[ActionPredictor] Could not load model weights ({e}). Operating in calibrated baseline mode.")
                self.nn_model.eval()
        else:
            self.nn_model.eval()

    def predict_action(
        self,
        frame: FrameTacticalState,
        player_id: int,
        sequence_frames: Optional[List[FrameTacticalState]] = None
    ) -> ActionPrediction:
        encoded_dict = self.encoder.encode_frame(frame)
        pf = encoded_dict["player_features"].get(player_id)

        if not pf:
            return ActionPrediction(
                action="HOLD",
                confidence=0.50,
                time_horizon=5.0,
                alternatives=[{"action": "PASS", "confidence": 0.30}],
                is_baseline_model=True,
                model_type="Fallback Baseline",
                calibration_status="Uncalibrated Fallback"
            )

        # 1. PyTorch Neural Model Inference (if trained weights available)
        if self.has_weights:
            try:
                if sequence_frames and len(sequence_frames) >= 10:
                    seq_vecs = [self.encoder.extract_feature_vector(f, player_id) for f in sequence_frames[-10:]]
                else:
                    curr_vec = self.encoder.extract_feature_vector(frame, player_id)
                    seq_vecs = [curr_vec for _ in range(10)]

                x_tensor = torch.tensor(np.array([seq_vecs]), dtype=torch.float32)
                with torch.no_grad():
                    logits, _, _ = self.nn_model(x_tensor)
                    probs = torch.softmax(logits[0], dim=0).numpy()

                sorted_indices = np.argsort(probs)[::-1]
                top_action = ACTIONS[sorted_indices[0]]
                top_conf = float(probs[sorted_indices[0]])

                alternatives = [
                    {"action": ACTIONS[idx], "confidence": float(round(probs[idx], 3))}
                    for idx in sorted_indices[1:4]
                ]

                return ActionPrediction(
                    action=top_action,
                    confidence=round(top_conf, 3),
                    time_horizon=5.0,
                    alternatives=alternatives,
                    is_baseline_model=False,
                    model_type="FootballTemporalLSTM Neural Model",
                    calibration_status="Calibrated PyTorch Neural Model"
                )
            except Exception as e:
                print(f"[ActionPredictor] Neural inference exception ({e}), falling back to calibrated baseline.")

        # 2. Calibrated Multinomial Softmax Spatial-Physics Baseline Model
        p = next((p for p in frame.players if p.id == player_id), None)
        dist_to_goal = pf["distance_to_goal"]
        pressure = pf["defensive_pressure"]
        space = pf["available_space"]
        ball_dist = pf["ball_distance"]
        speed = p.speed if p else 0.0

        is_possessor = (frame.ball.possession_player_id == player_id)

        # Logits derived from spatial domain physics
        logit_pass = 1.8 + (space / 8.0) - (pressure * 1.5) if is_possessor else -1.0
        logit_dribble = 1.2 + (space / 10.0) + (speed / 4.0) - (pressure * 0.8) if is_possessor else -1.5
        logit_shot = (3.0 - (dist_to_goal / 8.0)) if (is_possessor and dist_to_goal < 25.0) else -3.5
        logit_cross = 2.2 if (is_possessor and (pf["y"] < 16.0 or pf["y"] > 52.0) and pf["x"] > 68.0) else -3.0
        logit_tackle = 3.2 if (not is_possessor and ball_dist < 2.5) else -3.0
        logit_carry = 1.0 + (speed / 3.0) if is_possessor else -1.5
        logit_hold = 0.5 - (pressure * 0.5)
        logit_clearance = 2.8 if (is_possessor and pf["x"] < 22.0 and pressure > 0.5) else -3.5

        logits = np.array([logit_pass, logit_dribble, logit_shot, logit_cross, logit_tackle, logit_carry, logit_hold, logit_clearance])
        
        # Temperature scaling (\tau = 1.2) for calibrated probabilities
        tau = 1.2
        exp_logits = np.exp(logits / tau)
        probs = exp_logits / np.sum(exp_logits)

        sorted_indices = np.argsort(probs)[::-1]
        top_action = ACTIONS[sorted_indices[0]]
        top_conf = float(probs[sorted_indices[0]])

        alternatives = [
            {"action": ACTIONS[idx], "confidence": float(round(probs[idx], 3))}
            for idx in sorted_indices[1:4]
        ]

        return ActionPrediction(
            action=top_action,
            confidence=round(top_conf, 3),
            time_horizon=5.0,
            alternatives=alternatives,
            is_baseline_model=True,
            model_type="Temperature-Calibrated Spatial-Physics Baseline Model",
            calibration_status="Calibrated Baseline (Multinomial Softmax)"
        )

