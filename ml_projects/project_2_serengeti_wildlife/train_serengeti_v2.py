"""Transfer learning for Serengeti wildlife — v2 with correct preprocessing.

Changes from v1:
- Uses MobileNetV2 preprocess_input ([-1, 1]) instead of / 255.0
- Adds ReduceLROnPlateau callback
- Adds EarlyStopping to avoid wasting epochs
"""

import argparse
import json
import os
import tensorflow as tf


NUM_CLASSES = 10
IMAGE_SIZE = 224
SPECIES = sorted([
    "wildebeest", "zebra", "gazellethomsons", "buffalo", "hartebeest",
    "elephant", "gazellegrants", "giraffe", "impala", "warthog"
])

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecord-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--epochs-frozen", type=int, default=10)
    parser.add_argument("--epochs-finetune", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--finetune-lr", type=float, default=0.00005)
    parser.add_argument("--finetune-layers", type=int, default=50)
    return parser.parse_args()


def get_tfrecord_files(base_dir, split):
    pattern = f"{base_dir}/{split}/{split}-*.tfrecord"
    files = tf.io.gfile.glob(pattern)
    print(f"  {split}: {len(files)} shard(s)")
    return files


def parse_example(serialized):
    feature_description = {
        "image_raw": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(serialized, feature_description)

    image = tf.io.decode_jpeg(example["image_raw"], channels=3)
    image = tf.cast(image, tf.float32)
    # Correct preprocessing: scale to [-1, 1] for MobileNetV2
    image = preprocess_input(image)
    image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])

    return image, example["label"]


def augment(image, label):
    # Augment BEFORE preprocess_input would be ideal, but since our
    # images are already in [-1, 1], we apply augmentations that work
    # in that range
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    # Clip to [-1, 1] range
    image = tf.clip_by_value(image, -1.0, 1.0)
    return image, label


def build_dataset(files, batch_size, is_training=False):
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(buffer_size=5000)
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def build_model(learning_rate):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, base_model


def main():
    args = parse_args()

    print("Loading TFRecord files...")
    train_files = get_tfrecord_files(args.tfrecord_dir, "train")
    val_files = get_tfrecord_files(args.tfrecord_dir, "val")

    train_ds = build_dataset(train_files, args.batch_size, is_training=True)
    val_ds = build_dataset(val_files, args.batch_size, is_training=False)

    # Phase 1: Frozen base
    print(f"\n{'='*60}")
    print(f"Phase 1: Frozen base — {args.epochs_frozen} epochs, lr={args.learning_rate}")
    print(f"{'='*60}")

    model, base_model = build_model(args.learning_rate)
    model.summary()

    history_frozen = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_frozen,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, verbose=1
            ),
        ],
    )

    frozen_val_acc = max(history_frozen.history["val_accuracy"])
    print(f"\nFrozen phase best val accuracy: {frozen_val_acc:.4f}")

    # Phase 2: Fine-tune
    print(f"\n{'='*60}")
    print(f"Phase 2: Fine-tune top {args.finetune_layers} layers — "
          f"{args.epochs_finetune} epochs, lr={args.finetune_lr}")
    print(f"{'='*60}")

    base_model.trainable = True
    for layer in base_model.layers[:-args.finetune_layers]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"Base model: {len(base_model.layers)} total, {trainable_count} trainable")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.finetune_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_finetune,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, verbose=1
            ),
        ],
    )

    finetune_val_acc = max(history_finetune.history["val_accuracy"])
    final_val_acc = history_finetune.history["val_accuracy"][-1]
    print(f"\nFine-tune best val accuracy: {finetune_val_acc:.4f}")
    print(f"Fine-tune final val accuracy (restored): {final_val_acc:.4f}")

    # Save model
    print(f"\nSaving model to {args.model_dir}")
    export_path = os.path.join(args.model_dir, "saved_model")
    model.save(export_path)
    print(f"SavedModel saved to {export_path}")

    # Save label map
    label_map = {i: s for i, s in enumerate(SPECIES)}
    local_label = "/tmp/label_map.json"
    with open(local_label, "w") as f:
        json.dump(label_map, f, indent=2)
    tf.io.gfile.copy(local_label, os.path.join(args.model_dir, "label_map.json"), overwrite=True)

    # Save training history
    history = {
        "frozen_val_accuracy": [float(v) for v in history_frozen.history["val_accuracy"]],
        "frozen_val_loss": [float(v) for v in history_frozen.history["val_loss"]],
        "finetune_val_accuracy": [float(v) for v in history_finetune.history["val_accuracy"]],
        "finetune_val_loss": [float(v) for v in history_finetune.history["val_loss"]],
        "best_frozen_val_accuracy": float(frozen_val_acc),
        "best_finetune_val_accuracy": float(finetune_val_acc),
    }
    local_hist = "/tmp/training_history.json"
    with open(local_hist, "w") as f:
        json.dump(history, f, indent=2)
    tf.io.gfile.copy(local_hist, os.path.join(args.model_dir, "training_history.json"), overwrite=True)

    print(f"\nTraining complete!")
    print(f"  Frozen best val accuracy:    {frozen_val_acc:.4f}")
    print(f"  Fine-tune best val accuracy: {finetune_val_acc:.4f}")


if __name__ == "__main__":
    main()
