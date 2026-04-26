"""Export SavedModel with base64 serving signature.

Wraps the trained model with a @tf.function that:
1. Accepts raw JPEG bytes (base64-decoded by TF Serving automatically)
2. Decodes, resizes, preprocesses
3. Returns class probabilities + predicted species name
"""

import argparse
import json
import os
import tensorflow as tf

IMAGE_SIZE = 224
SPECIES = sorted([
    "wildebeest", "zebra", "gazellethomsons", "buffalo", "hartebeest",
    "elephant", "gazellegrants", "giraffe", "impala", "warthog"
])

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="GCS path to trained model")
    parser.add_argument("--export-dir", required=True, help="GCS path for serving model")
    args = parser.parse_args()

    # Load trained model
    model_path = os.path.join(args.model_dir, "saved_model")
    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded")

    # Species lookup table
    species_table = tf.constant(SPECIES)

    # Define serving function
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def serve(image_bytes):
        """Accept batch of raw JPEG bytes, return predictions."""
        def decode_and_preprocess(raw):
            image = tf.io.decode_jpeg(raw, channels=3)
            image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
            image = tf.cast(image, tf.float32)
            image = preprocess_input(image)
            return image

        images = tf.map_fn(decode_and_preprocess, image_bytes, fn_output_signature=tf.float32)
        probabilities = model(images, training=False)
        predicted_indices = tf.argmax(probabilities, axis=1)
        predicted_species = tf.gather(species_table, predicted_indices)
        confidences = tf.reduce_max(probabilities, axis=1)

        return {
            "species": predicted_species,
            "confidence": confidences,
            "probabilities": probabilities,
        }

    # Save with serving signature
    print(f"Exporting serving model to {args.export_dir}")
    tf.saved_model.save(
        model,
        args.export_dir,
        signatures={"serving_default": serve},
    )
    print("Export complete")

    # Verify
    loaded = tf.saved_model.load(args.export_dir)
    sig = loaded.signatures["serving_default"]
    print(f"Serving signature inputs: {sig.structured_input_signature}")
    print(f"Serving signature outputs: {list(sig.structured_outputs.keys())}")


if __name__ == "__main__":
    main()
