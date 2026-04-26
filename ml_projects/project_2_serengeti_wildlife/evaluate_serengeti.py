"""Evaluate saved model on test TFRecords, produce per-class metrics."""

import argparse
import json
import os
import numpy as np
import tensorflow as tf

NUM_CLASSES = 10
IMAGE_SIZE = 224
SPECIES = sorted([
    "wildebeest", "zebra", "gazellethomsons", "buffalo", "hartebeest",
    "elephant", "gazellegrants", "giraffe", "impala", "warthog"
])

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input


def parse_example(serialized):
    feature_description = {
        "image_raw": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(serialized, feature_description)
    image = tf.io.decode_jpeg(example["image_raw"], channels=3)
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
    return image, example["label"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tfrecord-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    # Load model
    model_path = os.path.join(args.model_dir, "saved_model")
    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Load test data
    test_pattern = f"{args.tfrecord_dir}/test/test-*.tfrecord"
    test_files = tf.io.gfile.glob(test_pattern)
    print(f"Test shards: {len(test_files)}")

    test_ds = tf.data.TFRecordDataset(test_files, num_parallel_reads=tf.data.AUTOTUNE)
    test_ds = test_ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    # Collect predictions
    all_labels = []
    all_preds = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        all_labels.extend(labels.numpy())
        all_preds.extend(np.argmax(preds, axis=1))

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Overall accuracy
    accuracy = np.mean(all_labels == all_preds)
    print(f"\nTest accuracy: {accuracy:.4f}")

    # Confusion matrix
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        confusion[true][pred] += 1

    # Per-class metrics
    results = {"test_accuracy": float(accuracy), "per_class": {}}
    print(f"\n{'Species':<20} {'Count':>6} {'Correct':>8} {'Accuracy':>9} {'Top Confusion':>30}")
    print("-" * 80)

    for i, species in enumerate(SPECIES):
        count = int(np.sum(confusion[i]))
        correct = int(confusion[i][i])
        acc = correct / count if count > 0 else 0

        # Find top confusion (most common misclassification)
        row = confusion[i].copy()
        row[i] = 0  # exclude correct class
        top_confused_idx = np.argmax(row)
        top_confused_count = row[top_confused_idx]
        top_confused = f"{SPECIES[top_confused_idx]} ({top_confused_count})"

        print(f"{species:<20} {count:>6} {correct:>8} {acc:>8.1%} {top_confused:>30}")

        results["per_class"][species] = {
            "count": count,
            "correct": correct,
            "accuracy": float(acc),
            "top_confusion": SPECIES[top_confused_idx],
            "top_confusion_count": int(top_confused_count),
        }

    # Save confusion matrix
    results["confusion_matrix"] = confusion.tolist()
    results["class_names"] = SPECIES

    local_path = "/tmp/eval_results.json"
    with open(local_path, "w") as f:
        json.dump(results, f, indent=2)
    output_path = os.path.join(args.output_dir, "eval_results.json")
    tf.io.gfile.copy(local_path, output_path, overwrite=True)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
