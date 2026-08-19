from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.experiment_tracker import ExperimentTracker


def evaluate_all_models():
    evaluator = ModelEvaluator()
    metrics = evaluator.compute_metrics()
    print("==================================================")
    print("=== FootballMind Model Scientific Evaluation   ===")
    print("==================================================")
    print(f"Total Test Samples:            {metrics['total_samples']}")
    print(f"Action Prediction Accuracy:    {metrics['accuracy'] * 100:.2f}%")
    print(f"Top-3 Accuracy:                {metrics['top_3_accuracy'] * 100:.2f}%")
    print(f"Mean Prediction Confidence:    {metrics['mean_confidence'] * 100:.2f}%")
    print(f"Expected Calibration Error:    {metrics['expected_calibration_error_ece']:.4f}")
    print(f"Brier Score:                   {metrics['brier_score']:.4f}")
    print(f"Log Loss (Cross-Entropy):      {metrics['log_loss']:.4f}")
    print(f"Macro F1-Score:                {metrics['macro_f1_score']:.3f}")
    print("==================================================")

    try:
        tracker = ExperimentTracker()
        tracker.log_run(
            experiment_name="FootballMind_Full_Model_Evaluation",
            hyperparameters={"eval_samples": metrics['total_samples']},
            metrics=metrics,
            notes="Full system evaluation with ECE, Brier score, and Macro F1"
        )
    except Exception as e:
        print(f"[Evaluate] Warning: Could not log experiment ({e})")

    return metrics


if __name__ == "__main__":
    evaluate_all_models()


