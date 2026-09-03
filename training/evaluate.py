from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.experiment_tracker import ExperimentTracker
from training.evaluate_detector import benchmark_detector_latency


def evaluate_all_models(include_detector: bool = True):
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

    if include_detector:
        try:
            print("\n--- YOLO Vision Detector Latency Benchmark ---")
            bench = benchmark_detector_latency(num_frames=5)
            metrics["detector_model"] = bench["model_name"]
            metrics["detector_latency_ms"] = bench["mean_latency_ms"]
            metrics["detector_fps"] = bench["inference_fps"]
            print(f"Model:                         {bench['model_name']}")
            print(f"Mean Frame Latency:            {bench['mean_latency_ms']:.2f} ms")
            print(f"Inference FPS:                 {bench['inference_fps']:.1f} FPS")
            print("==================================================")
        except Exception as e:
            print(f"[Evaluate] Detector benchmark notice: {e}")

    try:
        tracker = ExperimentTracker()
        tracker.log_run(
            experiment_name="FootballMind_Full_Model_Evaluation",
            hyperparameters={"eval_samples": metrics['total_samples']},
            metrics=metrics,
            notes="Full system evaluation with ECE, Brier score, Macro F1, and YOLO11m detector latency"
        )
    except Exception as e:
        print(f"[Evaluate] Warning: Could not log experiment ({e})")

    return metrics


if __name__ == "__main__":
    evaluate_all_models()
