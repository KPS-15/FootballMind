import pytest
from src.evaluation.evaluator import ModelEvaluator


def test_model_evaluator_metrics():
    evaluator = ModelEvaluator()
    evaluator.log_prediction(
        predicted_action="PASS",
        actual_action="PASS",
        confidence=0.85,
        top_k_actions=["PASS", "DRIBBLE", "SHOT"],
        all_class_probs={"PASS": 0.85, "DRIBBLE": 0.10, "SHOT": 0.05}
    )
    evaluator.log_prediction(
        predicted_action="DRIBBLE",
        actual_action="PASS",
        confidence=0.60,
        top_k_actions=["DRIBBLE", "PASS", "HOLD"],
        all_class_probs={"DRIBBLE": 0.60, "PASS": 0.30, "HOLD": 0.10}
    )

    metrics = evaluator.compute_metrics()
    assert metrics["total_samples"] == 2
    assert "accuracy" in metrics
    assert "expected_calibration_error_ece" in metrics
    assert "brier_score" in metrics
    assert "log_loss" in metrics
    assert "macro_f1_score" in metrics
    assert "confusion_matrix" in metrics
    assert 0.0 <= metrics["expected_calibration_error_ece"] <= 1.0


def test_benchmark_eval():
    evaluator = ModelEvaluator()
    evaluator.run_benchmark_eval(num_samples=30)
    metrics = evaluator.compute_metrics()
    assert metrics["total_samples"] > 0
    assert 0.0 <= metrics["accuracy"] <= 1.0
