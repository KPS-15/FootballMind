import os
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
from src.vision.detector import FootballDetector
from src.vision.dataset_validator import DatasetValidator


def evaluate_detector_on_dataset(
    model_path: str = "yolo11m.pt",
    data_yaml: str = "datasets/data.yaml",
    imgsz: int = 1280,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    device: Optional[str] = None,
    split: str = "val"
) -> Dict[str, Any]:
    """
    Runs official Ultralytics validation on a Roboflow/Ultralytics dataset.
    Extracts overall mAP50, mAP50-95, precision, recall, and per-class ball/player metrics.
    """
    from ultralytics import YOLO

    validator = DatasetValidator(data_yaml)
    val_info = validator.validate()

    model = YOLO(model_path)
    kwargs = {
        "data": data_yaml,
        "imgsz": imgsz,
        "conf": conf_thresh,
        "iou": iou_thresh,
        "split": split,
        "verbose": False
    }
    if device:
        kwargs["device"] = device

    metrics = model.val(**kwargs)

    # Extract metrics dictionary
    box_metrics = getattr(metrics, "box", None)
    mp = float(box_metrics.mp) if box_metrics else 0.0
    mr = float(box_metrics.mr) if box_metrics else 0.0
    map50 = float(box_metrics.map50) if box_metrics else 0.0
    map50_95 = float(box_metrics.map) if box_metrics else 0.0

    # Per-class metrics
    class_map = val_info.get("class_mapping", {})
    ball_recall = 0.0
    ball_precision = 0.0
    player_recall = 0.0
    player_precision = 0.0

    if box_metrics and hasattr(box_metrics, "p") and hasattr(box_metrics, "r"):
        p_list = box_metrics.p
        r_list = box_metrics.r
        for idx, mapped_name in class_map.items():
            if idx < len(r_list) and idx < len(p_list):
                if mapped_name == "ball":
                    ball_recall = float(r_list[idx])
                    ball_precision = float(p_list[idx])
                elif mapped_name == "player":
                    player_recall = float(r_list[idx])
                    player_precision = float(p_list[idx])

    # Speed metrics
    speed = getattr(metrics, "speed", {})
    preprocess_ms = speed.get("preprocess", 1.0)
    inference_ms = speed.get("inference", 15.0)
    postprocess_ms = speed.get("postprocess", 1.0)
    total_ms = preprocess_ms + inference_ms + postprocess_ms
    fps = round(1000.0 / total_ms, 1) if total_ms > 0 else 0.0

    report = {
        "model": model_path,
        "dataset": data_yaml,
        "split": split,
        "imgsz": imgsz,
        "precision": round(mp, 4),
        "recall": round(mr, 4),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "ball_recall": round(ball_recall, 4),
        "ball_precision": round(ball_precision, 4),
        "player_recall": round(player_recall, 4),
        "player_precision": round(player_precision, 4),
        "latency_ms": round(total_ms, 2),
        "inference_fps": fps
    }
    return report


def benchmark_detector_latency(
    model_name: str = "yolo11m.pt",
    num_frames: int = 30,
    imgsz: int = 1280
) -> Dict[str, Any]:
    """
    Benchmarks model inference throughput and latency on video frames.
    """
    detector = FootballDetector(model_name=model_name, imgsz=imgsz)
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Warmup
    for _ in range(3):
        detector.detect_frame(dummy_frame)

    latencies = []
    for _ in range(num_frames):
        t0 = time.perf_counter()
        detector.detect_frame(dummy_frame)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = float(np.mean(latencies))
    std_ms = float(np.std(latencies))
    fps = round(1000.0 / avg_ms, 1) if avg_ms > 0 else 0.0

    return {
        "model_name": model_name,
        "resolution": f"{1280}x{720}",
        "imgsz": imgsz,
        "num_frames": num_frames,
        "mean_latency_ms": round(avg_ms, 2),
        "std_latency_ms": round(std_ms, 2),
        "inference_fps": fps,
    }


def compare_models(models: List[str] = ["yolo11m.pt", "yolo11s.pt", "yolo11n.pt"]) -> Dict[str, Any]:
    """
    Compares candidate YOLO models on latency and compatibility.
    """
    results = {}
    for m in models:
        try:
            print(f"[Benchmark] Evaluating {m}...")
            bench = benchmark_detector_latency(model_name=m, num_frames=10)
            results[m] = {
                "status": "success",
                "fps": bench["inference_fps"],
                "latency_ms": bench["mean_latency_ms"]
            }
        except Exception as e:
            results[m] = {
                "status": "unsupported_or_error",
                "error": str(e)
            }
    return results


def print_evaluation_summary(metrics: Dict[str, Any]):
    print("=" * 60)
    print(" FOOTBALLMIND VISION DETECTOR EVALUATION REPORT")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f" {k:<25}: {v:.4f}")
        else:
            print(f" {k:<25}: {v}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="FootballMind Detector Evaluation Suite")
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="YOLO model path or name")
    parser.add_argument("--data", type=str, default="datasets/data.yaml", help="Dataset data.yaml path")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference resolution")
    parser.add_argument("--benchmark-only", action="store_true", help="Run latency benchmark only")
    parser.add_argument("--compare", action="store_true", help="Compare multiple YOLO model sizes")

    args = parser.parse_args()

    if args.compare:
        print("[EvaluateDetector] Running model comparison suite...")
        res = compare_models(["yolo11m.pt", "yolo11s.pt", "yolo11n.pt"])
        for m, stats in res.items():
            print(f"Model: {m} -> {stats}")
        return

    if args.benchmark_only or not Path(args.data).exists():
        if not Path(args.data).exists() and not args.benchmark_only:
            print(f"[EvaluateDetector] Notice: '{args.data}' not found. Running latency & throughput benchmark instead.")
        bench = benchmark_detector_latency(model_name=args.model, imgsz=args.imgsz)
        print_evaluation_summary(bench)
    else:
        report = evaluate_detector_on_dataset(model_path=args.model, data_yaml=args.data, imgsz=args.imgsz)
        print_evaluation_summary(report)


if __name__ == "__main__":
    main()
