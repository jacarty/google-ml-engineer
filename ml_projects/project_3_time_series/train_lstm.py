"""Train an LSTM on Bitcoin volatility windows.

Reads numpy data from GCS, trains a Keras LSTM, saves the SavedModel back to GCS.
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from google.cloud import storage


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True,
                   help="GCS path holding train/val/test .npy files (e.g. gs://bucket/path)")
    p.add_argument("--output-dir", required=True,
                   help="GCS path to write the trained SavedModel (e.g. gs://bucket/path)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lstm-units", type=int, default=32)
    p.add_argument("--dense-units", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def download_npy_from_gcs(gcs_uri, local_path):
    """Download a single GCS object to a local file."""
    assert gcs_uri.startswith("gs://"), gcs_uri
    bucket_name, _, blob_path = gcs_uri[len("gs://"):].partition("/")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)


def load_inputs(input_dir):
    """Download all training inputs to /tmp and load into memory."""
    files = ["train_X.npy", "train_y.npy", "val_X.npy", "val_y.npy", "scaler_params.json"]
    data = {}
    for fname in files:
        local_path = f"/tmp/{fname}"
        download_npy_from_gcs(f"{input_dir}/{fname}", local_path)
        if fname.endswith(".npy"):
            data[fname.replace(".npy", "")] = np.load(local_path)
        elif fname.endswith(".json"):
            with open(local_path) as f:
                data["scaler_params"] = json.load(f)
    return data


def build_model(window_size, n_features, lstm_units, dense_units, learning_rate):
    """Define the LSTM architecture."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window_size, n_features)),
        tf.keras.layers.LSTM(lstm_units, return_sequences=False),
        tf.keras.layers.Dense(dense_units, activation="relu"),
        tf.keras.layers.Dense(1, activation="linear"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def main():
    args = parse_args()

    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Input dir:  {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Hyperparameters: epochs={args.epochs}, batch_size={args.batch_size}, "
          f"lstm_units={args.lstm_units}, dense_units={args.dense_units}, "
          f"lr={args.learning_rate}, patience={args.patience}")
    print("=" * 60)

    tf.keras.utils.set_random_seed(args.seed)

    # Load data
    print("\nLoading data from GCS...")
    data = load_inputs(args.input_dir)
    train_X, train_y = data["train_X"], data["train_y"]
    val_X, val_y = data["val_X"], data["val_y"]
    scaler = data["scaler_params"]

    print(f"  train_X: {train_X.shape}, train_y: {train_y.shape}")
    print(f"  val_X:   {val_X.shape}, val_y:   {val_y.shape}")
    print(f"  window_size: {scaler['window_size']}, n_features: {scaler['n_features']}")

    # Build model
    print("\nBuilding model...")
    model = build_model(
        window_size=scaler["window_size"],
        n_features=scaler["n_features"],
        lstm_units=args.lstm_units,
        dense_units=args.dense_units,
        learning_rate=args.learning_rate,
    )
    model.summary()

    # Train
    print("\nTraining...")
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        train_X, train_y,
        validation_data=(val_X, val_y),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop],
        verbose=2,
    )

    # Final metrics
    final_train_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]
    final_train_mae = history.history["mae"][-1]
    final_val_mae = history.history["val_mae"][-1]
    print(f"\nFinal train_loss={final_train_loss:.4f}, val_loss={final_val_loss:.4f}")
    print(f"Final train_mae ={final_train_mae:.4f}, val_mae ={final_val_mae:.4f}")
    print(f"Stopped at epoch {len(history.history['loss'])} of {args.epochs}")

    # Save model
    print(f"\nSaving model to {args.output_dir}...")
    model.save(args.output_dir, save_format="tf")
    print("Done.")


if __name__ == "__main__":
    main()
