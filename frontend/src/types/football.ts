export interface PlayerState {
  id: number;
  team: 'TEAM A' | 'TEAM B' | 'REFEREE' | 'GOALKEEPER' | 'TEAM UNKNOWN' | 'home' | 'away' | string;
  team_confidence?: number;
  x: number;
  y: number;
  pixel_x?: number;
  pixel_y?: number;
  velocity_x: number;
  velocity_y: number;
  speed: number;
  direction: number;
  acceleration: number;
  body_orientation: number;
  ball_distance: number;
  possession_probability: number;
}


export interface BallState {
  x: number;
  y: number;
  pixel_x?: number;
  pixel_y?: number;
  velocity_x: number;
  velocity_y: number;
  speed: number;
  possession_player_id?: number | null;
  possession_team?: string | null;
}

export interface FrameTacticalState {
  frame_index: number;
  timestamp: number;
  players: PlayerState[];
  ball: BallState;
  attacking_team: string;
  defensive_team: string;
}

export interface ActionPrediction {
  action: string;
  confidence: number;
  time_horizon: number;
  alternatives: { action: string; confidence: number }[];
  is_baseline_model?: boolean;
  model_type?: string;
  calibration_status?: string;
}

export interface PassRecommendation {
  receiver_id: number;
  score: number;
  success_probability: number;
  attacking_advantage: number;
  space_created: number;
  start_pos: [number, number];
  end_pos: [number, number];
}

export interface DefensiveCollapseIndex {
  overall_danger: number;
  cb_lb_gap_risk: number;
  cb_rb_gap_risk: number;
  midfield_exposure: number;
  passing_lane_exposure: number;
  unmarked_attacker_risk: number;
  space_occupation_risk: number;
  methodology?: string;
}

export interface GoalProbability {
  goal_probability: number;
  shot_angle_deg: number;
  distance_to_goal: number;
  defenders_in_lane: number;
  model_description?: string;
}

export interface GoalkeeperRecommendation {
  current_position: [number, number];
  recommended_position: [number, number];
  positioning_error: number;
  xg_reduction: number;
}

export interface WhatIfResponse {
  baseline_danger: number;
  scenario_danger: number;
  danger_delta: number;
  baseline_xg: number;
  scenario_xg: number;
  xg_delta: number;
  baseline_space: number;
  scenario_space: number;
  space_delta: number;
  summary: string;
}

export interface FeatureAttribution {
  feature_name: string;
  contribution: number;
  description: string;
}

export interface ExplainablePrediction {
  prediction_type: string;
  predicted_value: string;
  confidence: number;
  top_features: FeatureAttribution[];
  narrative_reason: string;
  alternative_decision: string;
  attribution_method?: string;
}

export interface GoalExplanation {
  goal_timestamp: number;
  primary_cause: string;
  secondary_causes: string[];
  critical_moment_timestamp: number;
  alternative_counteraction: string;
  model_estimated_contribution: Record<string, number>;
  causality_disclaimer?: string;
}

export interface VideoMetadata {
  filename: string;
  file_path: string;
  file_size_mb: number;
  duration_sec: number;
  resolution: [number, number];
  fps: number;
  status: string;
}

export interface DefensiveRecommendation {
  defender_id: number;
  action: 'MARK' | 'PRESS' | 'COVER' | 'HOLD' | string;
  target_player_id?: number | null;
  reason: string;
}

export interface TacticalEvent {
  timestamp_sec: number;
  timestamp_str: string;
  event_type: string;
  description: string;
}

export interface VideoAnalysisResponse {
  video_metadata: VideoMetadata;
  sequence_type: 'ATTACKING' | 'DEFENSIVE' | 'MIXED' | string;
  analysis_mode: string;
  ball_carrier_id?: number | null;
  carrier_label?: string;
  best_pass?: {
    from: number;
    to: number;
    score: number;
    success_probability?: number;
    attacking_advantage?: number;
    space_created?: number;
  } | null;
  best_pass_label?: string;
  tactical_score?: number;
  alternative_pass?: {
    from: number;
    to: number;
    score: number;
    success_probability?: number;
  } | null;
  open_space_channel: string;
  space_label?: string;
  pass_reason: string;
  main_defensive_threat_id?: number | null;
  defensive_recommendations: DefensiveRecommendation[];
  defensive_recommendations_short?: string[];
  defensive_danger_score: number;
  recommended_response: string;
  short_alert?: string;
  key_tactical_observation: string;
  events: TacticalEvent[];
  annotated_video_url?: string | null;
  download_url?: string | null;
  summary_text: string;
}




