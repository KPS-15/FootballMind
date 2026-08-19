import os
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


class ExperimentTracker:
    """
    Automated Experiment Tracker for FootballMind ML models.
    Logs hyperparameters, training loss curves, validation metrics (Accuracy, ECE, Brier Score, F1),
    and model artifact paths to JSON and CSV formats for scientific reproducibility.
    """

    def __init__(self, log_dir: str = "experiments"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.json_file = self.log_dir / "experiment_log.json"
        self.csv_file = self.log_dir / "experiment_runs.csv"

    def log_run(
        self,
        experiment_name: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, Any],
        artifact_paths: Optional[Dict[str, str]] = None,
        notes: str = ""
    ) -> Dict[str, Any]:
        timestamp = datetime.now().isoformat()
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        run_record = {
            "run_id": run_id,
            "timestamp": timestamp,
            "experiment_name": experiment_name,
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "artifact_paths": artifact_paths or {},
            "notes": notes
        }

        # 1. Update JSON log
        existing_runs = []
        if self.json_file.exists():
            try:
                with open(self.json_file, "r", encoding="utf-8") as f:
                    existing_runs = json.load(f)
            except Exception:
                existing_runs = []

        existing_runs.append(run_record)
        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(existing_runs, f, indent=2)

        # 2. Update CSV log
        csv_row = {
            "run_id": run_id,
            "timestamp": timestamp,
            "experiment_name": experiment_name,
            "epochs": hyperparameters.get("epochs", ""),
            "lr": hyperparameters.get("lr", ""),
            "hidden_size": hyperparameters.get("hidden_size", ""),
            "val_accuracy": metrics.get("val_accuracy", metrics.get("accuracy", "")),
            "val_loss": metrics.get("val_loss", ""),
            "ece": metrics.get("expected_calibration_error_ece", ""),
            "brier_score": metrics.get("brier_score", ""),
            "log_loss": metrics.get("log_loss", ""),
            "macro_f1": metrics.get("macro_f1_score", "")
        }

        file_exists = self.csv_file.exists()
        with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(csv_row)

        print(f"[ExperimentTracker] Successfully logged experiment '{experiment_name}' -> {run_id}")
        return run_record

    def list_runs(self) -> List[Dict[str, Any]]:
        if not self.json_file.exists():
            return []
        try:
            with open(self.json_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
