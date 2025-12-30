import typer
import pandas as pd
import numpy as np
from pathlib import Path
from ml_baseline import config, train

app = typer.Typer(no_args_is_help=True)


@app.command(name="make-sample-data")
def make_sample_data(output_path: str = "data/processed/feature_table.csv"):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(
        {
            "id": range(100),
            "feature_1": np.random.randn(100),
            "is_high_value": np.random.randint(0, 2, 100),
            "churn": np.random.randint(0, 2, 100),
        }
    )
    data.to_csv(path, index=False)
    typer.echo(f"Sample data created at: {path}")


@app.command(name="train")
def train_cmd(
    target: str = typer.Option(...),
    input_path: str = "data/processed/feature_table.csv",
):
    cfg = config.TrainCfg(features_path=Path(input_path).resolve(), target=target)
    run_dir = train.run_training(cfg)
    typer.echo(f"Run created at {run_dir}")
