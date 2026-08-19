import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from src.models.temporal_model import FootballTemporalLSTM
from src.data.synthetic_generator import SyntheticMatchGenerator
from src.core.state_encoder import FootballStateEncoder


def train_action_model(epochs: int = 15, save_dir: str = "models"):
    print("[TrainAction] Starting PyTorch action predictor training run...")
    gen = SyntheticMatchGenerator(seed=42)
    encoder = FootballStateEncoder()

    X_data = []
    y_data = []

    # Generate multi-match sequences for comprehensive training
    for match_seed in [42, 123, 456, 789]:
        gen = SyntheticMatchGenerator(seed=match_seed)
        frames = gen.generate_sequence(num_frames=250)

        for idx in range(10, len(frames)):
            seq = frames[idx-10:idx]
            target_frame = frames[idx]
            
            # Sample for all active players in target frame
            for p in target_frame.players:
                seq_vecs = [encoder.extract_feature_vector(f, p.id) for f in seq]
                label = gen.get_ground_truth_action(target_frame, p.id)
                X_data.append(seq_vecs)
                y_data.append(label)

    X_tensor = torch.tensor(np.array(X_data), dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.long)

    # 80/20 Train / Validation Split
    num_samples = len(X_tensor)
    indices = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(indices)

    split_idx = int(num_samples * 0.8)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
    X_val, y_val = X_tensor[val_idx], y_tensor[val_idx]

    print(f"[TrainAction] Dataset built: {len(X_train)} train samples, {len(X_val)} validation samples.")

    model = FootballTemporalLSTM(input_size=16, hidden_size=64, num_actions=8)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits, _, _ = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_logits, _, _ = model(X_val)
            val_loss = criterion(val_logits, y_val).item()
            preds = torch.argmax(val_logits, dim=1)
            val_acc = (preds == y_val).float().mean().item()
        model.train()

        print(f"[TrainAction] Epoch {epoch+1:02d}/{epochs:02d} - Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(save_dir, "action_predictor.pt")
    torch.save(model.state_dict(), save_path)
    print(f"[TrainAction] PyTorch Action Model saved successfully to {save_path}")

    # Log experiment run
    try:
        from src.evaluation.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker()
        tracker.log_run(
            experiment_name="FootballTemporalLSTM_Action_Training",
            hyperparameters={
                "epochs": epochs,
                "lr": 0.002,
                "hidden_size": 64,
                "num_actions": 8,
                "input_size": 16,
                "sequence_length": 10
            },
            metrics={
                "train_loss": round(float(loss.item()), 4),
                "val_loss": round(float(val_loss), 4),
                "val_accuracy": round(float(val_acc), 4)
            },
            artifact_paths={"model_weights": save_path},
            notes="Trained on synthetic multi-match sequence dataset"
        )
    except Exception as e:
        print(f"[TrainAction] Warning: Could not log experiment ({e})")


if __name__ == "__main__":
    train_action_model()


