import pytest
from src.data.synthetic_generator import SyntheticMatchGenerator
from src.explainability.explainer import ExplainabilityEngine
from src.explainability.goal_analyzer import GoalSequenceAnalyzer


def test_explainability_engine():
    gen = SyntheticMatchGenerator()
    frames = gen.generate_sequence(num_frames=10)
    engine = ExplainabilityEngine()

    exp = engine.explain_prediction(frames[0], player_id=8)
    assert exp.prediction_type == "Next Action Intention"
    assert exp.attribution_method == "Empirical Model Feature Occlusion Perturbation"
    assert len(exp.top_features) > 0
    assert exp.top_features[0].feature_name != ""


def test_goal_sequence_analyzer():
    gen = SyntheticMatchGenerator()
    frames = gen.generate_sequence(num_frames=30)
    analyzer = GoalSequenceAnalyzer()

    goal_exp = analyzer.analyze_goal_sequence(frames)
    assert goal_exp.critical_moment_timestamp >= 0.0
    assert "Defensive" in goal_exp.primary_cause or "Unmarked" in goal_exp.primary_cause
    assert "simulated_xg_reduction" in goal_exp.model_estimated_contribution
