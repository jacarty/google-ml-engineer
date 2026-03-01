import argparse
import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--text-truncate", type=int, default=500)
    args = parser.parse_args()

    # AIP_MODEL_DIR is set automatically by Vertex AI
    # Saving here auto-registers the model in Model Registry
    model_dir = os.environ.get("AIP_MODEL_DIR")
    print(f"Args: {vars(args)}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"AIP_MODEL_DIR: {model_dir}")

    # --- Load data (GCS paths work natively in pre-built containers) ---
    print("Loading data...")
    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(args.data_dir, "val.csv"))

    train_df = train_df.dropna(subset=["text", "label"])
    val_df = val_df.dropna(subset=["text", "label"])

    # Truncate long text — critical for performance
    train_df["text"] = train_df["text"].str[:args.text_truncate]
    val_df["text"] = val_df["text"].str[:args.text_truncate]

    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # --- Encode labels ---
    labels = sorted(train_df["label"].unique())
    label_to_id = {l: i for i, l in enumerate(labels)}
    print(f"Labels: {label_to_id}")

    y_train = train_df["label"].map(label_to_id).values
    y_val = val_df["label"].map(label_to_id).values

    # --- Text vectorization ---
    print("Building text vectorizer...")
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=args.vocab_size,
        output_mode="int",
        output_sequence_length=args.max_length,
    )
    vectorizer.adapt(train_df["text"].values)
    print(f"Vocabulary size: {vectorizer.vocabulary_size()}")

    # --- Build model ---
    print("Building model...")
    model = tf.keras.Sequential([
        vectorizer,
        tf.keras.layers.Embedding(args.vocab_size, args.embedding_dim),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(labels), activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # --- Create tf.data datasets for efficient batching ---
    print("Creating datasets...")
    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_df["text"].values, y_train)
    ).shuffle(10000).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(
        (val_df["text"].values, y_val)
    ).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    # --- Train ---
    print("Training...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    # --- Evaluate ---
    val_loss, val_accuracy = model.evaluate(val_ds)
    print(f"\nValidation accuracy: {val_accuracy:.4f}")

    # --- Save as SavedModel (format expected by TF Serving prediction container) ---
    print(f"Saving model to {model_dir}...")
    model.save(model_dir)
    print("Model saved!")

    # --- Save metadata alongside model ---
    metadata = {
        "val_accuracy": float(val_accuracy),
        "val_loss": float(val_loss),
        "labels": labels,
        "label_to_id": label_to_id,
        "params": vars(args),
        "train_size": len(train_df),
        "val_size": len(val_df),
    }
    metadata_path = os.path.join(model_dir, "metadata.json")
    with tf.io.gfile.GFile(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {json.dumps(metadata, indent=2)}")


if __name__ == "__main__":
    main()
