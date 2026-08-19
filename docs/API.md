# FootballMind REST API Reference Specification

The FastAPI backend server runs by default on `http://127.0.0.1:8000`. Interactive OpenAPI Swagger documentation is available at `http://127.0.0.1:8000/docs`.

---

## Endpoint Overview Table

| Method | Route | Description |
|--------|-------|-------------|
| `GET`  | `/` | System root health status & API version |
| `GET`  | `/api/match/{match_id}` | Metadata for specified match |
| `GET`  | `/api/tracking/{match_id}` | Frame-level tracking state (22 players + ball) |
| `GET`  | `/api/prediction/{match_id}` | Action prediction & receiver probability for player |
| `GET`  | `/api/tactical/{match_id}` | Defensive collapse index, xG, & GK recommendation |
| `GET`  | `/api/recommendations/{match_id}` | Ranked pass options for ball carrier |
| `POST` | `/api/simulation` | Counterfactual What-If tactical positional scenario |
| `GET`  | `/api/explanation/{match_id}` | Empirical feature attributions & goal sequence analysis |
| `GET`  | `/api/evaluation/{match_id}` | ECE, Brier score, Log loss, F1, & confusion matrix |
| `POST` | `/api/video/upload` | Upload video file for CV tracking |
| `POST` | `/api/video/process` | Run YOLO, tracking, homography, & team classification |

---

## Endpoint Details & Example Payloads

### 1. Match Information
- **Route**: `GET /api/match/{match_id}`
- **Response**:
```json
{
  "match_id": "demo_match_01",
  "title": "Home (4-3-3) vs Away (4-4-2) Tactical Match",
  "total_frames": 120,
  "fps": 15,
  "pitch_size": [105.0, 68.0],
  "home_team": "Red",
  "away_team": "Blue"
}
```

---

### 2. Action & Receiver Prediction
- **Route**: `GET /api/prediction/{match_id}?player_id=7&frame_index=30`
- **Response**:
```json
{
  "match_id": "demo_match_01",
  "player_id": 7,
  "action_prediction": {
    "action": "PASS",
    "confidence": 0.85,
    "time_horizon": 5.0,
    "is_baseline_model": false,
    "model_type": "FootballTemporalLSTM Neural Model",
    "calibration_status": "Calibrated PyTorch Neural Model",
    "alternatives": [
      { "action": "DRIBBLE", "confidence": 0.10 },
      { "action": "CARRY", "confidence": 0.03 }
    ]
  },
  "receiver_prediction": {
    "receiver_id": 9,
    "probability": 0.82,
    "target_location": [72.0, 34.0],
    "is_baseline_model": false,
    "model_type": "Time-to-Intercept (TTI) Kinematic Physics Model"
  }
}
```

---

### 3. Tactical Analysis & Expected Goals
- **Route**: `GET /api/tactical/{match_id}?frame_index=30`
- **Response**:
```json
{
  "match_id": "demo_match_01",
  "frame_index": 30,
  "defensive_collapse_index": {
    "overall_danger": 0.42,
    "cb_lb_gap_risk": 0.35,
    "cb_rb_gap_risk": 0.28,
    "midfield_exposure": 0.40,
    "passing_lane_exposure": 0.45,
    "unmarked_attacker_risk": 0.30,
    "space_occupation_risk": 0.25,
    "methodology": "Direction-Aware Spatial Structural Geometry Index"
  },
  "goal_probability": {
    "goal_probability": 0.125,
    "shot_angle_deg": 28.4,
    "distance_to_goal": 21.5,
    "defenders_in_lane": 2,
    "model_description": "Calibrated Logistic Sigmoid xG Model"
  },
  "goalkeeper_recommendation": {
    "current_position": [100.0, 34.0],
    "recommended_position": [102.2, 34.0],
    "positioning_error": 2.2,
    "xg_reduction": 0.045
  }
}
```

---

### 4. Counterfactual What-If Tactical Simulation
- **Route**: `POST /api/simulation`
- **Request Payload**:
```json
{
  "match_id": "demo_match_01",
  "frame_index": 30,
  "modified_player_id": 3,
  "new_x": 22.0,
  "new_y": 26.0
}
```
- **Response Payload**:
```json
{
  "baseline_danger": 0.42,
  "scenario_danger": 0.31,
  "danger_delta": -0.11,
  "baseline_xg": 0.125,
  "scenario_xg": 0.082,
  "xg_delta": -0.043,
  "baseline_space": 12.4,
  "scenario_space": 10.1,
  "space_delta": -2.3,
  "summary": "Modifying Player #3 position to [22.0m, 26.0m]: Defensive Danger changed by -11.0 percentage points.",
  "is_simulation_counterfactual": true
}
```

---

### 5. Scientific Model Evaluation Metrics
- **Route**: `GET /api/evaluation/{match_id}`
- **Response**:
```json
{
  "total_samples": 3080,
  "accuracy": 1.0,
  "top_3_accuracy": 1.0,
  "mean_confidence": 0.963,
  "expected_calibration_error_ece": 0.0373,
  "brier_score": 0.0016,
  "log_loss": 0.038,
  "macro_f1_score": 0.125,
  "confusion_matrix": { ... },
  "calibration_bins": [ ... ]
}
```
