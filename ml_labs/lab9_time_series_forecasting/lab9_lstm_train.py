"""
Lab 9 — LSTM Training Script for Vertex AI
Trains a single-layer LSTM on JFK temperature data.
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from google.cloud import storage

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-uri", type=str, required=True)
    parser.add_argument("--model-uri", type=str, required=True)
    parser.add_argument("--window-size", type=int, default=14)
    parser.add_argument("--lstm-units", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    return parser.parse_args()

LSTM_FEATURES = [
    "temp", "dewpoint", "wind_speed", "precipitation",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
]

def create_windows(data, window_size, target_col_idx=0):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i, target_col_idx])
    return np.array(X), np.array(y)

def main():
    args = parse_args()
    print(f"\n{'='*60}")
    print(f"Lab 9 — LSTM Training")
    print(f"{'='*60}")
    print(f"  window_size: {args.window_size}")
    print(f"  lstm_units: {args.lstm_units}")
    print(f"  epochs: {args.epochs}")
    print(f"  learning_rate: {args.learning_rate}")

    # Load data
    df_train = pd.read_csv(f"{args.data_uri}/jfk_weather_train.csv")
    df_val = pd.read_csv(f"{args.data_uri}/jfk_weather_val.csv")
    df_test = pd.read_csv(f"{args.data_uri}/jfk_weather_test.csv")
    print(f"  Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # Scale
    scaler = StandardScaler()
    scaler.fit(df_train[LSTM_FEATURES])
    train_scaled = scaler.transform(df_train[LSTM_FEATURES])
    val_scaled = scaler.transform(df_val[LSTM_FEATURES])
    test_scaled = scaler.transform(df_test[LSTM_FEATURES])

    # Create windows
    X_train, y_train = create_windows(train_scaled, args.window_size)
    X_val, y_val = create_windows(val_scaled, args.window_size)
    X_test, y_test = create_windows(test_scaled, args.window_size)
    print(f"  X_train: {X_train.shape}")

    # Build model
    model = keras.Sequential([
        layers.LSTM(args.lstm_units, input_shape=(args.window_size, len(LSTM_FEATURES))),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
                  loss="mse", metrics=["mae"])

    # Train
    print("\n🏋️ Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
        ],
        verbose=2
    )

    # Evaluate in °F
    def to_fahrenheit(scaled, scaler_obj):
        return scaled * scaler_obj.scale_[0] + scaler_obj.mean_[0]

    results = {}
    for name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        preds = model.predict(X).flatten()
        actual_f = to_fahrenheit(y, scaler)
        pred_f = to_fahrenheit(preds, scaler)
        rmse = float(np.sqrt(mean_squared_error(actual_f, pred_f)))
        mae = float(mean_absolute_error(actual_f, pred_f))
        mape = float(np.mean(np.abs((actual_f - pred_f) / actual_f)) * 100)
        results[name] = {"rmse": rmse, "mae": mae, "mape": mape}
        print(f"\n📊 {name.upper()}: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.1f}%")

    # Save model as SavedModel
    local_model_dir = "/tmp/lstm_model"
    model.save(local_model_dir)

    # Upload to GCS using gsutil (avoids blob permission issues)
    import subprocess
    gcs_model_dest = f"{args.model_uri}/saved_model/"
    subprocess.run(["gsutil", "-m", "cp", "-r", f"{local_model_dir}/*", gcs_model_dest], check=True)
    print(f"\n✅ SavedModel uploaded to {gcs_model_dest}")

    # Save metrics using gsutil too
    metrics_local = "/tmp/metrics.json"
    with open(metrics_local, "w") as f:
        json.dump(results, f, indent=2)
    subprocess.run(["gsutil", "cp", metrics_local, f"{args.model_uri}/metrics.json"], check=True)

    scaler_local = "/tmp/scaler.json"
    scaler_info = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                   "features": LSTM_FEATURES}
    with open(scaler_local, "w") as f:
        json.dump(scaler_info, f, indent=2)
    subprocess.run(["gsutil", "cp", scaler_local, f"{args.model_uri}/scaler.json"], check=True)

    print(f"✅ Metrics and scaler saved")

if __name__ == "__main__":
    main()
