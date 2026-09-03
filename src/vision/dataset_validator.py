import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


class DatasetValidationError(Exception):
    """Raised when dataset configuration or structure is invalid."""
    pass


class DatasetValidator:
    """
    Validates Roboflow and Ultralytics YOLO formatted datasets for FootballMind.
    Ensures data.yaml exists, required train/val/test paths exist, and maps dataset classes
    to standard football domain labels (ball, player, referee, goalkeeper).
    """

    # Canonical football domain classes
    STANDARD_CLASSES = ["ball", "player", "referee", "goalkeeper"]

    def __init__(self, data_yaml_path: str = "datasets/data.yaml"):
        self.data_yaml_path = Path(data_yaml_path)
        self.raw_config: Dict[str, Any] = {}
        self.classes: List[str] = []
        self.class_mapping: Dict[int, str] = {}
        self.train_path: Optional[Path] = None
        self.val_path: Optional[Path] = None
        self.test_path: Optional[Path] = None

    def validate(self) -> Dict[str, Any]:
        """
        Validates the dataset configuration file and directory structure.
        Returns a dictionary with validation status, summary, and detected class mapping.
        """
        if not self.data_yaml_path.exists():
            raise DatasetValidationError(f"Dataset configuration file not found at: {self.data_yaml_path}")

        try:
            with open(self.data_yaml_path, "r", encoding="utf-8") as f:
                self.raw_config = yaml.safe_load(f) or {}
        except Exception as e:
            raise DatasetValidationError(f"Failed to parse YAML file '{self.data_yaml_path}': {e}")

        if not isinstance(self.raw_config, dict):
            raise DatasetValidationError(f"Invalid YAML structure in '{self.data_yaml_path}': root must be a mapping/dict.")

        base_dir = self.data_yaml_path.parent
        root_dir = self.raw_config.get("path")
        if root_dir:
            base_dir = Path(root_dir) if Path(root_dir).is_absolute() else base_dir / root_dir

        # Validate train split
        train_rel = self.raw_config.get("train")
        if not train_rel:
            raise DatasetValidationError(f"Missing required 'train' path in '{self.data_yaml_path}'")
        self.train_path = Path(train_rel) if Path(train_rel).is_absolute() else base_dir / train_rel

        # Validate val/validation split
        val_rel = self.raw_config.get("val") or self.raw_config.get("validation")
        if not val_rel:
            raise DatasetValidationError(f"Missing required 'val' or 'validation' path in '{self.data_yaml_path}'")
        self.val_path = Path(val_rel) if Path(val_rel).is_absolute() else base_dir / val_rel

        # Optional test split
        test_rel = self.raw_config.get("test")
        if test_rel:
            self.test_path = Path(test_rel) if Path(test_rel).is_absolute() else base_dir / test_rel

        # Parse and validate classes / names
        names = self.raw_config.get("names")
        if names is None:
            raise DatasetValidationError(f"Missing 'names' (class list or dict) in '{self.data_yaml_path}'")

        if isinstance(names, list):
            self.classes = [str(name).strip() for name in names]
            self.class_mapping = {i: self._normalize_class_name(name) for i, name in enumerate(self.classes)}
        elif isinstance(names, dict):
            # Dict mapping {0: 'ball', 1: 'player', ...}
            sorted_keys = sorted(names.keys(), key=lambda k: int(k) if str(k).isdigit() else k)
            self.classes = [str(names[k]).strip() for k in sorted_keys]
            self.class_mapping = {int(k) if str(k).isdigit() else i: self._normalize_class_name(str(names[k])) for i, k in enumerate(sorted_keys)}
        else:
            raise DatasetValidationError(f"'names' in '{self.data_yaml_path}' must be a list or dict of class names.")

        nc = self.raw_config.get("nc")
        if nc is not None and int(nc) != len(self.classes):
            print(f"[DatasetValidator] Warning: 'nc' ({nc}) does not match number of classes ({len(self.classes)}). Using {len(self.classes)} classes.")

        # Check domain coverage
        normalized_set = set(self.class_mapping.values())
        has_ball = "ball" in normalized_set
        has_player = "player" in normalized_set
        has_referee = "referee" in normalized_set
        has_goalkeeper = "goalkeeper" in normalized_set

        warnings = []
        if not has_ball:
            warnings.append("Dataset does not contain a recognized 'ball' class. Ball detection will be unavailable or heuristic.")
        if not has_player:
            warnings.append("Dataset does not contain a recognized 'player' class.")
        if not has_referee:
            warnings.append("Dataset does not contain a 'referee' class. Referees may be detected as players.")
        if not has_goalkeeper:
            warnings.append("Dataset does not contain a 'goalkeeper' class. Goalkeepers may be detected as players.")

        return {
            "valid": True,
            "data_yaml": str(self.data_yaml_path),
            "num_classes": len(self.classes),
            "raw_classes": self.classes,
            "class_mapping": self.class_mapping,
            "train_path": str(self.train_path),
            "val_path": str(self.val_path),
            "test_path": str(self.test_path) if self.test_path else None,
            "has_ball": has_ball,
            "has_player": has_player,
            "has_referee": has_referee,
            "has_goalkeeper": has_goalkeeper,
            "warnings": warnings,
        }

    @staticmethod
    def _normalize_class_name(raw_name: str) -> str:
        """Normalizes various dataset annotations to standard FootballMind names."""
        name_lower = raw_name.lower().strip()
        if any(k in name_lower for k in ["ball", "football", "soccer", "sports ball"]):
            return "ball"
        if any(k in name_lower for k in ["goalkeeper", "gk", "keeper", "goalie"]):
            return "goalkeeper"
        if any(k in name_lower for k in ["referee", "ref", "judge", "official"]):
            return "referee"
        if any(k in name_lower for k in ["player", "person", "athlete", "outfield"]):
            return "player"
        return name_lower
