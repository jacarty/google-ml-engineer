"""
Lab 3: Custom training script with hyperparameter tuning support.

Based on Lab 2's train.py with one key addition:
- Reports metrics back to Vertex AI using cloudml-hypertune

Usage (local):
    python lab3_train.py --data-path census_income.csv --n-estimators 100 --max-depth 5

Usage (Vertex AI tuning):
    Vertex AI passes --n-estimators, --max-depth, etc. automatically per trial
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import os


def load_data(data_path):
    """Load data from GCS or local path."""
    print(f"Loading data from: {data_path}")
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def preprocess_data(df, target_column):
    """Prepare features and target."""
    X = df.drop(columns=[target_column])
    y = df[target_column].str.strip()
    y = (y == '>50K').astype(int)
    X = pd.get_dummies(X, drop_first=True)
    return X, y


def save_model(model, model_dir):
    """Save trained model."""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")


def main():
    parser = argparse.ArgumentParser()

    # Data arguments
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--target-column', type=str, default='income_bracket')
    parser.add_argument('--model-dir', type=str, default='/tmp/model')

    # Hyperparameters — Vertex AI tuning will override these
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--min-samples-split', type=int, default=2)

    args = parser.parse_args()

    # Load and preprocess
    df = load_data(args.data_path)
    X, y = preprocess_data(df, args.target_column)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    hyperparams = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'min_samples_split': args.min_samples_split,
    }

    print(f"\nHyperparameters: {hyperparams}")

    model = GradientBoostingClassifier(**hyperparams, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  ROC AUC:  {roc_auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))

    # ========================================================
    # KEY ADDITION FOR LAB 3: Report metric to Vertex AI
    # ========================================================
    # cloudml-hypertune is only available inside Vertex AI containers.
    # When running locally, this block is safely skipped.
    try:
        import hypertune
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='accuracy',
            metric_value=accuracy
        )
        print(f"Reported accuracy={accuracy:.4f} to Vertex AI tuning service")
    except ImportError:
        print("(cloudml-hypertune not available — running locally, skipping metric report)")

    # Save model
    save_model(model, args.model_dir)


if __name__ == '__main__':
    main()
