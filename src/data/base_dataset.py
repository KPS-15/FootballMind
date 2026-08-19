from abc import ABC, abstractmethod
from typing import List, Dict, Any
from src.core.types import FrameTacticalState


class BaseFootballDataset(ABC):
    """
    Abstract Base Class for Football Intelligence Datasets (SoccerNet, StatsBomb, Metrica, Synthetic).
    """

    @abstractmethod
    def load_match(self, match_id: str) -> List[FrameTacticalState]:
        """Loads tracking & event frames for a given match ID."""
        pass

    @abstractmethod
    def list_matches(self) -> List[Dict[str, Any]]:
        """Lists available match metadata in the dataset."""
        pass
