import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from src.vision.dataset_validator import DatasetValidator, DatasetValidationError


def get_optimal_training_params(
    device: Optional[str] = None,
    requested_batch: int = 16,
    requested_imgsz: int = 1280,
    requested_workers: int = 8
) -> Dict[str, Any]:
    """
    Computes optimal training parameters based on hardware capabilities.
    Automatically reduces batch size or image resolution if GPU memory is limited or on CPU.
    """
    resolved_device = device
    if not resolved_device or resolved_device == "auto":
        resolved_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    batch = requested_batch
    imgsz = requested_imgsz
    workers = requested_workers

    if resolved_device == "cpu" or not torch.cuda.is_available():
        # CPU constrained mode: reduce batch and image size for feasibility
        batch = min(4, requested_batch)
        imgsz = min(640, requested_imgsz)
        workers = min(2, requested_workers)
        print(f"[TrainDetector] Running on CPU. Auto-adjusting batch to {batch}, imgsz to {imgsz}, workers to {workers}.")
    else:
        # GPU available: inspect VRAM
        try:
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"[TrainDetector] CUDA GPU detected with {gpu_mem_gb:.1f} GB VRAM.")
            if gpu_mem_gb < 8.0:
                batch = min(4, requested_batch)
                imgsz = min(640, requested_imgsz)
                print(f"[TrainDetector] Low VRAM (<8GB). Adjusted batch to {batch}, imgsz to {imgsz}.")
            elif gpu_mem_gb < 16.0:
                batch = min(8, requested_batch)
                print(f"[TrainDetector] Medium VRAM (<16GB). Adjusted batch to {batch}.")
        except Exception:
            pass

    return {
        "device": resolved_device,
        "batch": batch,
        "imgsz": imgsz,
        "workers": workers
    }


def train_yolo_detector(
    model: str = "yolo11m.pt",
    data: str = "datasets/data.yaml",
    epochs: int = 100,
    imgsz: int = 1280,
    batch: int = 16,
    device: Optional[str] = None,
    workers: int = 8,
    project: str = "runs/detect",
    name: str = "football_yolo11m",
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    save_period: int = 10,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Trains Ultralytics YOLO model (YOLO11m default) on the validated football dataset.
    """
    print("=" * 60)
    print(" FOOTBALLMIND YOLO DETECTOR TRAINING PIPELINE")
    print("=" * 60)
    print(f"Base Model:       {model}")
    print(f"Dataset YAML:     {data}")
    print(f"Target Epochs:    {epochs}")
    print(f"Target Img Size:  {imgsz}")
    print(f"Target Batch:     {batch}")

    # 1. Validate Dataset YAML
    validator = DatasetValidator(data)
    try:
        val_info = validator.validate()
        print(f"[TrainDetector] Dataset validation PASSED: {val_info['num_classes']} classes detected: {val_info['raw_classes']}")
        if val_info["warnings"]:
            for w in val_info["warnings"]:
                print(f"[TrainDetector] Dataset Notice: {w}")
    except DatasetValidationError as e:
        print(f"[TrainDetector] Dataset validation error: {e}")
        if not dry_run:
            raise

    # 2. Compute hardware-aware parameters
    params = get_optimal_training_params(
        device=device,
        requested_batch=batch,
        requested_imgsz=imgsz,
        requested_workers=workers
    )

    if dry_run:
        print("[TrainDetector] Dry run completed successfully. Configuration is valid.")
        return {
            "status": "dry_run_success",
            "model": model,
            "params": params,
            "data_info": val_info if 'val_info' in locals() else None
        }

    # 3. Load YOLO model and begin training
    from ultralytics import YOLO
    yolo_model = YOLO(model)

    print(f"[TrainDetector] Starting YOLO training on device '{params['device']}'...")
    try:
        results = yolo_model.train(
            data=data,
            epochs=epochs,
            imgsz=params["imgsz"],
            batch=params["batch"],
            device=params["device"],
            workers=params["workers"],
            project=project,
            name=name,
            save_period=save_period,
            exist_ok=True
        )
        print("=" * 60)
        print(" TRAINING COMPLETE")
        print("=" * 60)
        return {
            "status": "success",
            "results": results,
            "best_weights": os.path.join(project, name, "weights", "best.pt")
        }
    except Exception as e:
        print(f"[TrainDetector] Training error: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="FootballMind YOLO Detector Training Pipeline")
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="Base YOLO model weights (default: yolo11m.pt)")
    parser.add_argument("--data", type=str, default="datasets/data.yaml", help="Path to Roboflow/Ultralytics data.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image resolution for training (default: 1280)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16, auto-scaled if needed)")
    parser.add_argument("--device", type=str, default=None, help="Device (e.g. '0', 'cpu', 'auto')")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers count")
    parser.add_argument("--project", type=str, default="runs/detect", help="Output directory for runs")
    parser.add_argument("--name", type=str, default="football_yolo11m", help="Run name")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset and parameters without training")

    args = parser.parse_args()
    train_yolo_detector(
        model=args.model,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
