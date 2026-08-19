import math
import numpy as np
from typing import Dict, List, Any, Optional
from src.models.action_predictor import ActionPredictor, ACTIONS
from src.data.synthetic_generator import SyntheticMatchGenerator


class ModelEvaluator:
    """
    Evaluates FootballMind predictions against ground truth events using statistically valid metrics:
    Expected Calibration Error (ECE), Brier Score, Log Loss, Macro/Weighted F1, and $8 \times 8$ Confusion Matrix.
    """

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def log_prediction(
        self,
        predicted_action: str,
        actual_action: str,
        confidence: float,
        top_k_actions: List[str],
        all_class_probs: Optional[Dict[str, float]] = None
    ):
        self.records.append({
            "predicted": predicted_action,
            "actual": actual_action,
            "confidence": float(confidence),
            "top_k": top_k_actions,
            "all_probs": all_class_probs or {predicted_action: confidence},
            "is_correct": predicted_action == actual_action,
            "is_top_k_correct": actual_action in top_k_actions
        })

    def run_benchmark_eval(self, num_samples: int = 200):
        """Runs automated evaluation benchmark on synthetic tracking dataset."""
        gen = SyntheticMatchGenerator(seed=101)
        frames = gen.generate_sequence(num_frames=num_samples)
        predictor = ActionPredictor()

        for idx in range(10, len(frames)):
            frame = frames[idx]
            seq = frames[idx-10:idx]
            for p in frame.players:
                actual_label_idx = gen.get_ground_truth_action(frame, p.id)
                actual_action = ACTIONS[actual_label_idx]

                pred = predictor.predict_action(frame, p.id, sequence_frames=seq)
                top_k = [pred.action] + [alt["action"] for alt in pred.alternatives]
                
                # Build complete probability dictionary
                probs_dict = {pred.action: pred.confidence}
                for alt in pred.alternatives:
                    probs_dict[alt["action"]] = alt["confidence"]
                
                # Assign small baseline prob to unlisted
                rem_prob = max(0.0, 1.0 - sum(probs_dict.values()))
                unlisted = [a for a in ACTIONS if a not in probs_dict]
                for u in unlisted:
                    probs_dict[u] = rem_prob / max(1, len(unlisted))

                self.log_prediction(
                    predicted_action=pred.action,
                    actual_action=actual_action,
                    confidence=pred.confidence,
                    top_k_actions=top_k[:3],
                    all_class_probs=probs_dict
                )

    def compute_metrics(self) -> Dict[str, Any]:
        if not self.records:
            self.run_benchmark_eval(num_samples=150)

        total = len(self.records)
        if total == 0:
            return {"error": "No records available for evaluation."}

        correct = sum(1 for r in self.records if r["is_correct"])
        top_k_correct = sum(1 for r in self.records if r["is_top_k_correct"])
        acc = correct / total
        top_3_acc = top_k_correct / total
        mean_conf = float(np.mean([r["confidence"] for r in self.records]))

        # 1. Expected Calibration Error (ECE) via 10 equal-width bins [0..1]
        num_bins = 10
        bins = [[] for _ in range(num_bins)]
        for r in self.records:
            bin_idx = min(num_bins - 1, int(r["confidence"] * num_bins))
            bins[bin_idx].append(r)

        ece = 0.0
        bin_details = []
        for b_idx, bin_records in enumerate(bins):
            if bin_records:
                bin_acc = sum(1 for r in bin_records if r["is_correct"]) / len(bin_records)
                bin_conf = sum(r["confidence"] for r in bin_records) / len(bin_records)
                weight = len(bin_records) / total
                ece += weight * abs(bin_acc - bin_conf)
                bin_details.append({
                    "bin_range": f"{b_idx/num_bins:.1f}-{(b_idx+1)/num_bins:.1f}",
                    "count": len(bin_records),
                    "bin_accuracy": round(bin_acc, 3),
                    "bin_confidence": round(bin_conf, 3)
                })

        # 2. Multi-class Brier Score and Log Loss
        brier_sum = 0.0
        log_loss_sum = 0.0

        for r in self.records:
            probs = r["all_probs"]
            actual = r["actual"]

            for a in ACTIONS:
                p_val = probs.get(a, 0.0)
                y_val = 1.0 if a == actual else 0.0
                brier_sum += (p_val - y_val) ** 2

            p_actual = max(1e-15, probs.get(actual, 0.01))
            log_loss_sum += -math.log(p_actual)

        brier_score = brier_sum / total
        log_loss = log_loss_sum / total

        # 3. 8x8 Confusion Matrix & Per-Class Precision/Recall/F1
        conf_matrix = {act_a: {act_p: 0 for act_p in ACTIONS} for act_a in ACTIONS}
        class_tp = {a: 0 for a in ACTIONS}
        class_fp = {a: 0 for a in ACTIONS}
        class_fn = {a: 0 for a in ACTIONS}

        for r in self.records:
            act_a = r["actual"] if r["actual"] in ACTIONS else "HOLD"
            act_p = r["predicted"] if r["predicted"] in ACTIONS else "HOLD"
            conf_matrix[act_a][act_p] += 1

            if act_a == act_p:
                class_tp[act_a] += 1
            else:
                class_fp[act_p] += 1
                class_fn[act_a] += 1

        f1_scores = []
        precisions = []
        recalls = []

        for a in ACTIONS:
            tp = class_tp[a]
            fp = class_fp[a]
            fn = class_fn[a]

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            precisions.append(prec)
            recalls.append(rec)
            f1_scores.append(f1)

        macro_f1 = float(np.mean(f1_scores))

        return {
            "total_samples": total,
            "accuracy": round(acc, 3),
            "top_3_accuracy": round(top_3_acc, 3),
            "mean_confidence": round(mean_conf, 3),
            "expected_calibration_error_ece": round(ece, 4),
            "brier_score": round(brier_score, 4),
            "log_loss": round(log_loss, 4),
            "macro_f1_score": round(macro_f1, 3),
            "confusion_matrix": conf_matrix,
            "calibration_bins": bin_details
        }

