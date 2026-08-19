# FootballMind Experiment Tracking Documentation

## Purpose & Overview
**FootballMind** includes an integrated experiment tracking engine ([ExperimentTracker](file:///c:/Users/Admin/Downloads/ball/src/evaluation/experiment_tracker.py)) designed to log all ML training runs, hyperparameter sweeps, loss trajectories, validation metrics, and model artifact versions for complete scientific reproducibility.

---

## Log Output Formats

Experiment runs are automatically logged to two local persistent formats:

1. **`experiments/experiment_log.json`**: Structured JSON array storing full run metadata, hyperparameter dictionaries, evaluation metrics, artifact file paths, and notes.
2. **`experiments/experiment_runs.csv`**: Tabular CSV file containing run summary columns for rapid tabular comparison across training iterations.

---

## Logged Attributes & Metric Definitions

Each logged experiment run records the following attributes:

| Attribute Category | Field Name | Description |
|-------------------|------------|-------------|
| **Identifier** | `run_id` | Unique timestamped run identifier (e.g. `run_20260817_225010`) |
| **Identifier** | `timestamp` | ISO format execution timestamp |
| **Identifier** | `experiment_name` | Descriptive name of training or evaluation run |
| **Hyperparameters** | `epochs` | Total training epochs executed |
| **Hyperparameters** | `lr` | Learning rate parameter (e.g. `0.002`) |
| **Hyperparameters** | `hidden_size` | LSTM hidden layer dimension size (e.g. `64`) |
| **Hyperparameters** | `sequence_length` | Temporal frame window size (e.g. `10`) |
| **Metrics** | `val_accuracy` | Multi-class validation prediction accuracy |
| **Metrics** | `val_loss` | Cross-entropy validation loss |
| **Metrics** | `expected_calibration_error_ece` | 10-bin Expected Calibration Error |
| **Metrics** | `brier_score` | Multiclass Brier score ($\frac{1}{N} \sum (p - y)^2$) |
| **Metrics** | `log_loss` | Multi-class cross-entropy loss |
| **Metrics** | `macro_f1_score` | Unweighted average F1-score across all 8 classes |
| **Artifacts** | `artifact_paths` | Dictionary mapping artifact name to saved file path |

---

## How to Run Experiments & Log Results

### 1. Training Action Model
Run the PyTorch training script to execute model training and automatically log hyperparameter metrics:
```bash
python -m training.train_action
```

### 2. Running Scientific Evaluation
Run the model evaluator to benchmark performance across tracking samples and log ECE/Brier score metrics:
```bash
python -m training.evaluate
```

---

## Python API Usage Example

```python
from src.evaluation.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker()

# Log a custom training run
tracker.log_run(
    experiment_name="LSTM_Hyperparameter_Sweep_v1",
    hyperparameters={
        "epochs": 20,
        "lr": 0.001,
        "hidden_size": 128,
        "batch_size": 32
    },
    metrics={
        "val_accuracy": 0.952,
        "val_loss": 0.041,
        "expected_calibration_error_ece": 0.035,
        "brier_score": 0.002
    },
    artifact_paths={
        "model_weights": "models/action_predictor.pt"
    },
    notes="20 epochs training with batch size 32"
)
```
