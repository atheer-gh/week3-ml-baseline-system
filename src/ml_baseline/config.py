from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainCfg:
    features_path: Path
    target: str
    session_id: int = 42
    train_size: float = 0.8
