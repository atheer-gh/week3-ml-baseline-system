import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report

def load_data(file_path: str):
    return pd.read_csv(file_path)

def get_features_and_target(df: pd.DataFrame, target_column: str):
    y = df[target_column]
    X = df.drop(columns=[target_column])
    return X, y

def split_data(df, target_column, test_size=0.2):
   
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df[target_column], 
        random_state=42
    )
    return train_df, test_df

def train_baseline(X_train, y_train):

    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, output_dict=True)

from pathlib import Path

def save_model(model, model_path: str):
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)  
    joblib.dump(model, path)

    