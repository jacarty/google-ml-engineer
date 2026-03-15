"""Training script for Mini-Lab B: Census income Keras model.

Runs on Vertex AI with prebuilt TF 2.15 container.
"""
import argparse
import json
import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow import keras
from google.cloud import storage

print(f"TensorFlow version: {tf.__version__}")


def download_blob(bucket_name, source_blob, dest_file):
    """Download a file from GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob)
    blob.download_to_filename(dest_file)
    print(f"Downloaded gs://{bucket_name}/{source_blob} -> {dest_file}")


def upload_directory(local_dir, bucket_name, gcs_prefix):
    """Upload a local directory to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for root, dirs, files in os.walk(local_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, local_dir)
            blob_path = f"{gcs_prefix}/{relative_path}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            print(f"Uploaded {local_path} -> gs://{bucket_name}/{blob_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket-name", required=True)
    parser.add_argument("--data-dir", required=True, help="GCS prefix for data files")
    parser.add_argument("--model-dir", required=True, help="GCS prefix for model output")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()

    # --- Download data from GCS ---
    os.makedirs("/tmp/data", exist_ok=True)
    for fname in ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy", "feature_names.pkl"]:
        download_blob(args.bucket_name, f"{args.data_dir}/{fname}", f"/tmp/data/{fname}")

    X_train = np.load("/tmp/data/X_train.npy")
    X_test = np.load("/tmp/data/X_test.npy")
    y_train = np.load("/tmp/data/y_train.npy")
    y_test = np.load("/tmp/data/y_test.npy")

    with open("/tmp/data/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    n_features = X_train.shape[1]
    print(f"\nTraining data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Features: {n_features}")

    # --- Build model ---
    model = keras.Sequential([
        keras.layers.Input(shape=(n_features,), name="input_features"),
        keras.layers.Dense(64, activation="relu", name="hidden_1"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu", name="hidden_2"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid", name="output"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # --- Train ---
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        ],
        verbose=1,
    )

    # --- Evaluate ---
    y_pred_proba = model.predict(X_test).flatten()
    y_pred = (y_pred_proba >= 0.5).astype(int)
    accuracy = float((y_pred == y_test).mean())

    from sklearn.metrics import roc_auc_score
    roc_auc = float(roc_auc_score(y_test, y_pred_proba))

    print(f"\nTest accuracy: {accuracy:.4f}")
    print(f"Test ROC AUC:  {roc_auc:.4f}")

    # --- Save metrics ---
    metrics = {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "epochs_trained": len(history.history["loss"]),
    }
    with open("/tmp/metrics.json", "w") as f:
        json.dump(metrics, f)

    client = storage.Client()
    bucket = client.bucket(args.bucket_name)
    blob = bucket.blob(f"{args.model_dir}/metrics.json")
    blob.upload_from_filename("/tmp/metrics.json")
    print(f"Metrics saved to gs://{args.bucket_name}/{args.model_dir}/metrics.json")

    # --- Save model as SavedModel ---
    local_model_dir = "/tmp/model_output"
    model.save(local_model_dir)
    print(f"\nModel saved locally to {local_model_dir}")

    # Upload to GCS
    upload_directory(local_model_dir, args.bucket_name, args.model_dir)
    print(f"\nModel uploaded to gs://{args.bucket_name}/{args.model_dir}/")


if __name__ == "__main__":
    main()
