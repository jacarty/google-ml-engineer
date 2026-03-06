"""
Lab 9 — GBT Training Script for Vertex AI
Trains a GradientBoostingRegressor on JFK temperature data with temporal features.
"""
import argparse
import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from google.cloud import storage, aiplatform

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-uri", type=str, required=True, help="GCS path to data folder")
    parser.add_argument("--model-uri", type=str, required=True, help="GCS path for model output")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--min-samples-split", type=int, default=10)
    parser.add_argument("--subsample", type=float, default=0.8)
    return parser.parse_args()

def load_data(data_uri):
    """Load train/val/test CSVs from GCS."""
    dfs = {}
    for split in ["train", "val", "test"]:
        path = f"{data_uri}/jfk_weather_{split}.csv"
        dfs[split] = pd.read_csv(path)
        print(f"  Loaded {split}: {len(dfs[split])} rows")
    return dfs["train"], dfs["val"], dfs["test"]

def get_feature_cols(df):
    """Select feature columns (exclude target and raw date/calendar)."""
    exclude = ["date", "temp", "temp_max", "temp_min", "year", "month", "day",
               "day_of_week", "day_of_year", "week_of_year"]
    return [c for c in df.columns if c not in exclude]

def main():
    args = parse_args()
    print(f"\n{'='*60}")
    print(f"Lab 9 — GBT Training")
    print(f"{'='*60}")
    print(f"  n_estimators: {args.n_estimators}")
    print(f"  max_depth: {args.max_depth}")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  subsample: {args.subsample}")

    # Load data
    print("\n📂 Loading data...")
    df_train, df_val, df_test = load_data(args.data_uri)

    feature_cols = get_feature_cols(df_train)
    print(f"  Features: {len(feature_cols)}")

    X_train = df_train[feature_cols].values
    y_train = df_train["temp"].values
    X_val = df_val[feature_cols].values
    y_val = df_val["temp"].values
    X_test = df_test[feature_cols].values
    y_test = df_test["temp"].values

    # Train
    print("\n🏋️ Training...")
    model = GradientBoostingRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_samples_split=args.min_samples_split,
        subsample=args.subsample,
        min_samples_leaf=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    results = {}
    for name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        preds = model.predict(X)
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        mae = float(mean_absolute_error(y, preds))
        mape = float(np.mean(np.abs((y - preds) / y)) * 100)
        results[name] = {"rmse": rmse, "mae": mae, "mape": mape}
        print(f"\n📊 {name.upper()}: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.1f}%")

    # Save model
    local_model_path = "/tmp/model.joblib"
    joblib.dump(model, local_model_path)

    # Upload to GCS
    gcs_model_path = args.model_uri.replace("gs://", "")
    bucket_name = gcs_model_path.split("/")[0]
    blob_path = "/".join(gcs_model_path.split("/")[1:]) + "/model.joblib"

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_model_path)
    print(f"\n✅ Model saved to gs://{bucket_name}/{blob_path}")

    # Save metrics
    metrics_path = "/".join(gcs_model_path.split("/")[1:]) + "/metrics.json"
    blob_metrics = bucket.blob(metrics_path)
    blob_metrics.upload_from_string(json.dumps(results, indent=2))
    print(f"✅ Metrics saved to gs://{bucket_name}/{metrics_path}")

    # Save feature names
    features_path = "/".join(gcs_model_path.split("/")[1:]) + "/feature_cols.json"
    blob_features = bucket.blob(features_path)
    blob_features.upload_from_string(json.dumps(feature_cols))
    print(f"✅ Feature columns saved to gs://{bucket_name}/{features_path}")

if __name__ == "__main__":
    main()
