from typing import List, Dict, Optional, Tuple, Any
from pydantic import BaseModel, Field, ConfigDict



class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class DetectedObject(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    track_id: int
    class_name: str = Field(alias="class")
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    center: List[float] # [x, y]
    team: Optional[str] = "TEAM UNKNOWN"
    team_confidence: float = 1.0



class PlayerState(BaseModel):
    id: int
    team: str  # "TEAM A", "TEAM B", "REFEREE", "GOALKEEPER", "TEAM UNKNOWN"
    team_confidence: float = 1.0
    x: float   # pitch coordinate x [0..105]
    y: float   # pitch coordinate y [0..68]
    pixel_x: Optional[float] = 0.0
    pixel_y: Optional[float] = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    speed: float = 0.0
    direction: float = 0.0  # degrees
    acceleration: float = 0.0
    body_orientation: float = 0.0  # degrees
    ball_distance: float = 999.0
    possession_probability: float = 0.0



class BallState(BaseModel):
    x: float
    y: float
    pixel_x: Optional[float] = 0.0
    pixel_y: Optional[float] = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    speed: float = 0.0
    possession_player_id: Optional[int] = None
    possession_team: Optional[str] = None


class FrameTacticalState(BaseModel):
    frame_index: int
    timestamp: float
    players: List[PlayerState]
    ball: BallState
    attacking_team: str = "home"
    defensive_team: str = "away"


class ActionPrediction(BaseModel):
    action: str
    confidence: float
    time_horizon: float = 5.0
    alternatives: List[Dict[str, Any]] = []
    is_baseline_model: bool = False
    model_type: str = "FootballTemporalLSTM Neural Model"
    calibration_status: str = "Calibrated Temperature-Scaled Model"


class ReceiverPrediction(BaseModel):
    receiver_id: Optional[int]
    probability: float
    target_location: List[float]  # [x, y]
    is_baseline_model: bool = False
    model_type: str = "Time-to-Intercept Physics Model"


class PassRecommendation(BaseModel):
    receiver_id: int
    score: float
    success_probability: float
    attacking_advantage: float
    space_created: float
    start_pos: List[float]
    end_pos: List[float]


class DefensiveCollapseIndex(BaseModel):
    overall_danger: float  # 0.0 to 1.0
    cb_lb_gap_risk: float
    cb_rb_gap_risk: float
    midfield_exposure: float
    passing_lane_exposure: float
    unmarked_attacker_risk: float
    space_occupation_risk: float
    methodology: str = "Direction-Aware Spatial Structural Geometry Index"


class GoalProbability(BaseModel):
    goal_probability: float  # xG (0.0 to 1.0)
    shot_angle_deg: float
    distance_to_goal: float
    defenders_in_lane: int
    model_description: str = "Calibrated Logistic Sigmoid xG Model"


class GoalkeeperRecommendation(BaseModel):
    current_position: List[float]
    recommended_position: List[float]
    positioning_error: float
    xg_reduction: float


class WhatIfRequest(BaseModel):
    match_id: str
    frame_index: int
    modified_player_id: int
    new_x: float
    new_y: float


class WhatIfResponse(BaseModel):
    baseline_danger: float
    scenario_danger: float
    danger_delta: float
    baseline_xg: float
    scenario_xg: float
    xg_delta: float
    baseline_space: float
    scenario_space: float
    space_delta: float
    summary: str
    is_simulation_counterfactual: bool = True


class FeatureAttribution(BaseModel):
    feature_name: str
    contribution: float
    description: str


class ExplainablePrediction(BaseModel):
    prediction_type: str
    predicted_value: str
    confidence: float
    top_features: List[FeatureAttribution]
    narrative_reason: str
    alternative_decision: str
    attribution_method: str = "Empirical Model Feature Occlusion Perturbation"


class GoalExplanation(BaseModel):
    goal_timestamp: float
    primary_cause: str
    secondary_causes: List[str]
    critical_moment_timestamp: float
    alternative_counteraction: str
    model_estimated_contribution: Dict[str, float]
    causality_disclaimer: str = "Model-estimated statistical contribution (Counterfactual Simulation Verified)"


class VideoMetadata(BaseModel):
    filename: str
    file_path: str
    file_size_mb: float
    duration_sec: float
    resolution: List[int]  # [width, height]
    fps: float
    status: str = "uploaded"


class DefensiveRecommendation(BaseModel):
    defender_id: int
    action: str  # "MARK", "PRESS", "COVER", "HOLD"
    target_player_id: Optional[int] = None
    reason: str


class TacticalEvent(BaseModel):
    timestamp_sec: float
    timestamp_str: str  # "00:03.2"
    event_type: str
    description: str


class VideoAnalysisRequest(BaseModel):
    file_path: str
    max_frames: int = 300
    sample_rate: int = 1


class VideoAnalysisResponse(BaseModel):
    video_metadata: VideoMetadata
    sequence_type: str  # "ATTACKING", "DEFENSIVE", "MIXED"
    analysis_mode: str = "Baseline Tactical Analysis"
    ball_carrier_id: Optional[int] = None
    carrier_label: str = "BALL #10"
    best_pass: Optional[Dict[str, Any]] = None
    best_pass_label: str = "#10 ────► #7"
    tactical_score: float = 0.38
    alternative_pass: Optional[Dict[str, Any]] = None
    open_space_channel: str = "Right side"
    space_label: str = "#7 OPEN • 12m SPACE"
    pass_reason: str = "Player #7 is moving into open space and passing lane is clear."
    main_defensive_threat_id: Optional[int] = None
    defensive_recommendations: List[DefensiveRecommendation] = []
    defensive_recommendations_short: List[str] = ["#12 ──► MARK #9", "#6 ──► PRESS", "#3 ──► COVER"]
    defensive_danger_score: int = 72  # 0 to 100
    recommended_response: str = "SHIFT RIGHT"
    short_alert: str = "⚠ #7 IS OPEN (12m SPACE)"
    key_tactical_observation: str = "Right-side attacking channel is available."
    events: List[TacticalEvent] = []
    annotated_video_url: Optional[str] = None
    download_url: Optional[str] = None
    summary_text: str = ""



