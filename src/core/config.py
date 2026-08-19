import os
import yaml
import torch
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


class Config:
    def __init__(self, config_path: str = None):
        path = config_path or os.getenv("FOOTBALLMIND_CONFIG", str(DEFAULT_CONFIG_PATH))
        self.data = {}
        if Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                self.data = yaml.safe_load(f) or {}

    @property
    def device(self) -> str:
        setting = self.data.get("system", {}).get("device", "auto")
        if setting == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return setting

    @property
    def pitch_length(self) -> float:
        return float(self.data.get("pitch", {}).get("length", 105.0))

    @property
    def pitch_width(self) -> float:
        return float(self.data.get("pitch", {}).get("width", 68.0))

    @property
    def demo_mode(self) -> bool:
        return bool(self.data.get("data", {}).get("demo_mode", True))

    def get(self, key: str, default=None):
        keys = key.split(".")
        val = self.data
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val


config = Config()
