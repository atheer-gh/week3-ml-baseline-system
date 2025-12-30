import json
import hashlib
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timezone
from ml_baseline import config, splits, pipeline, metrics


def get_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_training(cfg: config.TrainCfg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = Path.cwd() / "models" / "runs" / f"{ts}__session{cfg.session_id}"
    for d in ["schema", "metrics", "model", "env"]:
        (run_dir / d).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg.features_path)
    schema = {
        "target": cfg.target,
        "data_hash": get_sha256(cfg.features_path),
        "features": [c for c in df.columns if c != cfg.target],
    }
    (run_dir / "schema" / "input_schema.json").write_text(json.dumps(schema, indent=2))

    train_df, test_df = splits.split_data(df, cfg)
    model = pipeline.build_baseline_pipeline()
    model.fit(train_df.drop(columns=[cfg.target]), train_df[cfg.target])

    y_pred = model.predict(test_df.drop(columns=[cfg.target]))
    run_metrics = metrics.get_metrics(test_df[cfg.target], y_pred)
    (run_dir / "metrics" / "scores.json").write_text(json.dumps(run_metrics, indent=2))

    joblib.dump(model, run_dir / "model" / "baseline_model.joblib")

    registry_dir = Path.cwd() / "models" / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    (registry_dir / "latest.txt").write_text(run_dir.name)

    return run_dir
