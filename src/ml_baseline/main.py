import typer
import pandas as pd
from ml_baseline import library

app = typer.Typer()

@app.command()
def train(
    target: str = typer.Option(..., help="name of the target column"),
    input_path: str = typer.Option("data/processed/feature_table.csv", help="Path to input feature table CSV")
):
    
    print(f" Loading data from {input_path}...")
    df = library.load_data(input_path)
    
    print(f" Splitting data (stratified by {target})...")
    train_df, test_df = library.split_data(df, target)
    
    X_train, y_train = library.get_features_and_target(train_df, target)
    X_test, y_test = library.get_features_and_target(test_df, target)

    print(f" Training Dummy Baseline (most_frequent)...")
    model = library.train_baseline(X_train, y_train)
    
    metrics = library.evaluate_model(model, X_test, y_test)
    
    model_path = "models/baseline_model.joblib"
    library.save_model(model, model_path)

    print("SUCCESS: Training complete.")
    print(f"METRIC: Baseline Accuracy = {metrics['accuracy']:.2f}")
    print(f"ARTIFACT: Model saved to {model_path}")
    

@app.command()
def make_sample_data():
    """Create a sample feature table CSV for testing."""
    import numpy as np
    from pathlib import Path
    
    path = Path("data/processed/feature_table.csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    data = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'churn': np.random.choice([0, 1], size=100)
    })
    data.to_csv(path, index=False)
    print(f" Sample data created at {path}")

if __name__ == "__main__":
    app()