from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from src.data.synthetic_generator import SyntheticMatchGenerator
from src.models.action_predictor import ActionPredictor
from src.models.pass_predictor import PassPredictor
from src.tactical.pass_recommender import PassRecommender
from src.tactical.defensive_analysis import DefensiveAnalyzer
from src.tactical.goalkeeper import GoalkeeperAnalyzer
from src.explainability.explainer import ExplainabilityEngine
from src.explainability.goal_analyzer import GoalSequenceAnalyzer
from src.evaluation.evaluator import ModelEvaluator

router = APIRouter(tags=["Match Intelligence"])

generator = SyntheticMatchGenerator()
match_frames = generator.generate_sequence(num_frames=120)

action_predictor = ActionPredictor()
pass_predictor = PassPredictor()
pass_recommender = PassRecommender()
defensive_analyzer = DefensiveAnalyzer()
gk_analyzer = GoalkeeperAnalyzer()
explainer = ExplainabilityEngine()
goal_analyzer = GoalSequenceAnalyzer()
evaluator = ModelEvaluator()


@router.get("/match/{match_id}")
def get_match_info(match_id: str):
    return {
        "match_id": match_id,
        "title": "Home (4-3-3) vs Away (4-4-2) Tactical Match",
        "total_frames": len(match_frames),
        "fps": 15,
        "pitch_size": [105.0, 68.0],
        "home_team": "Red",
        "away_team": "Blue"
    }


@router.get("/tracking/{match_id}")
def get_match_tracking(match_id: str, frame_index: int = 30):
    idx = max(0, min(len(match_frames) - 1, frame_index))
    frame = match_frames[idx]
    return frame.model_dump()


@router.get("/prediction/{match_id}")
def get_match_predictions(match_id: str, player_id: int = 7, frame_index: int = 30):
    idx = max(0, min(len(match_frames) - 1, frame_index))
    frame = match_frames[idx]
    
    action_pred = action_predictor.predict_action(frame, player_id)
    receiver_pred = pass_predictor.predict_pass(frame, player_id)

    return {
        "match_id": match_id,
        "player_id": player_id,
        "action_prediction": action_pred.model_dump(),
        "receiver_prediction": receiver_pred.model_dump()
    }


@router.get("/tactical/{match_id}")
def get_tactical_metrics(match_id: str, frame_index: int = 30):
    idx = max(0, min(len(match_frames) - 1, frame_index))
    frame = match_frames[idx]

    def_index = defensive_analyzer.analyze_defensive_structure(frame)
    xg_prob = gk_analyzer.calculate_xg(frame, frame.ball.possession_player_id or 7)
    gk_rec = gk_analyzer.recommend_gk_position(frame, frame.ball.possession_player_id or 7)

    return {
        "match_id": match_id,
        "frame_index": frame_index,
        "defensive_collapse_index": def_index.model_dump(),
        "goal_probability": xg_prob.model_dump(),
        "goalkeeper_recommendation": gk_rec.model_dump()
    }


@router.get("/recommendations/{match_id}")
def get_pass_recommendations(match_id: str, player_id: int = 7, frame_index: int = 30):
    idx = max(0, min(len(match_frames) - 1, frame_index))
    frame = match_frames[idx]

    recs = pass_recommender.recommend_passes(frame, player_id)
    return {
        "match_id": match_id,
        "passer_id": player_id,
        "recommendations": [r.model_dump() for r in recs]
    }


@router.get("/explanation/{match_id}")
def get_explainability(match_id: str, player_id: int = 7, frame_index: int = 30):
    idx = max(0, min(len(match_frames) - 1, frame_index))
    frame = match_frames[idx]

    pred_exp = explainer.explain_prediction(frame, player_id)
    goal_exp = goal_analyzer.analyze_goal_sequence(match_frames[max(0, idx-30):idx+1])

    return {
        "match_id": match_id,
        "prediction_explanation": pred_exp.model_dump(),
        "goal_explanation": goal_exp.model_dump()
    }


@router.get("/evaluation/{match_id}")
def get_evaluation_metrics(match_id: str):
    return evaluator.compute_metrics()
