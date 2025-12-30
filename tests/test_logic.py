# tests/test_logic.py
from ml_baseline import config
from pathlib import Path

def test_config_initialization():
 
    cfg = config.TrainCfg(
        features_path=Path("data/processed/feature_table.csv"),
        target="churn"
    )
    assert cfg.target == "churn"
    assert cfg.session_id == 42  