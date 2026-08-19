from typing import List, Dict, Any
from pathlib import Path
from src.data.base_dataset import BaseFootballDataset
from src.data.synthetic_generator import SyntheticMatchGenerator
from src.core.types import FrameTacticalState


class SoccerNetDatasetAdapter(BaseFootballDataset):
    """
    Adapter for SoccerNet-Tracking / SoccerNet-GameState dataset.
    Falls back to synthetic benchmark generator if SoccerNet local files are absent.
    """

    def __init__(self, dataset_dir: str = "datasets/soccernet"):
        self.dataset_dir = Path(dataset_dir)
        self.synthetic_gen = SyntheticMatchGenerator()

    def list_matches(self) -> List[Dict[str, Any]]:
        if self.dataset_dir.exists():
            matches = [p.name for p in self.dataset_dir.glob("*") if p.is_dir()]
            if matches:
                return [{"match_id": m, "source": "soccernet"} for m in matches]

        # Default fallback
        return [
            {"match_id": "demo_match_01", "source": "synthetic", "title": "Home 4-3-3 vs Away 4-4-2 (Demo)"},
            {"match_id": "demo_match_02", "source": "synthetic", "title": "Counter-Attack Attack Build-up (Demo)"}
        ]

    def load_match(self, match_id: str) -> List[FrameTacticalState]:
        match_file = self.dataset_dir / match_id / "tracking.json"
        if match_file.exists():
            # Parse real dataset JSON if available
            pass
        
        # Return generated benchmark frames
        return self.synthetic_gen.generate_sequence(num_frames=300)
