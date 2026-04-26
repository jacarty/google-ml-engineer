"""Create sharded TFRecords from images in GCS.

Runs as a Vertex AI CustomJob. Reads images from GCS, resizes to 224x224,
and writes sharded TFRecords back to GCS.

Usage:
    python create_tfrecords.py \
        --manifest-uri gs://bucket/manifest.csv \
        --output-dir gs://bucket/tfrecords \
        --image-size 224 \
        --num-shards 10
"""

import argparse
import io
import os
import math

import pandas as pd
import tensorflow as tf
from google.cloud import storage

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-uri", required=True, help="GCS path to subset_manifest.csv")
    parser.add_argument("--output-dir", required=True, help="GCS output directory for TFRecords")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-shards", type=int, default=10)
    return parser.parse_args()


def read_manifest_from_gcs(uri):
    """Read CSV manifest from GCS."""
    # Parse gs://bucket/path
    parts = uri.replace("gs://", "").split("/", 1)
    bucket_name, blob_path = parts[0], parts[1]
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    content = blob.download_as_text()
    
    return pd.read_csv(io.StringIO(content))


def read_and_resize_image(gcs_path, image_size):
    """Read an image from GCS and resize to target dimensions."""
    parts = gcs_path.replace("gs://", "").split("/", 1)
    bucket_name, blob_path = parts[0], parts[1]
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    image_bytes = blob.download_as_bytes()
    
    # Decode and resize
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, tf.uint8)
    
    # Re-encode as JPEG for compact storage
    encoded = tf.io.encode_jpeg(image)
    return encoded.numpy()


def make_example(image_bytes, label, species, file_name):
    """Create a tf.train.Example from image data."""
    return tf.train.Example(features=tf.train.Features(feature={
        "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        "species": tf.train.Feature(bytes_list=tf.train.BytesList(value=[species.encode()])),
        "file_name": tf.train.Feature(bytes_list=tf.train.BytesList(value=[file_name.encode()])),
    }))


def write_tfrecords_for_split(df, split, label_map, output_dir, image_size, num_shards):
    """Write sharded TFRecords for a single split."""
    split_df = df[df["split"] == split].reset_index(drop=True)
    
    if len(split_df) == 0:
        print(f"  No data for split '{split}', skipping")
        return
    
    # Shuffle to mix species across shards
    split_df = split_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    records_per_shard = math.ceil(len(split_df) / num_shards)
    total_written = 0
    errors = 0
    
    # Reuse storage client
    client = storage.Client()
    
    for shard_idx in range(num_shards):
        start = shard_idx * records_per_shard
        end = min(start + records_per_shard, len(split_df))
        shard_df = split_df.iloc[start:end]
        
        if len(shard_df) == 0:
            break
        
        shard_filename = f"{split}-{shard_idx:05d}-of-{num_shards:05d}.tfrecord"
        shard_path = f"{output_dir}/{split}/{shard_filename}"
        
        # Parse GCS path for writer
        parts = shard_path.replace("gs://", "").split("/", 1)
        bucket_name, blob_path = parts[0], parts[1]
        
        # Write to local temp, then upload
        local_path = f"/tmp/{shard_filename}"
        
        with tf.io.TFRecordWriter(local_path) as writer:
            for _, row in shard_df.iterrows():
                try:
                    image_bytes = read_and_resize_image(row["gcs_path"], image_size)
                    label = label_map[row["species"]]
                    example = make_example(image_bytes, label, row["species"], row["file_name"])
                    writer.write(example.SerializeToString())
                    total_written += 1
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        print(f"    Error processing {row['file_name']}: {e}")
        
        # Upload to GCS
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(local_path)
        os.remove(local_path)
        
        print(f"  Shard {shard_idx+1}/{num_shards}: {len(shard_df)} records → {shard_path}")
    
    print(f"  {split}: {total_written} written, {errors} errors")


def main():
    args = parse_args()
    
    # Label map (alphabetical)
    species_list = sorted([
        "wildebeest", "zebra", "gazellethomsons", "buffalo", "hartebeest",
        "elephant", "gazellegrants", "giraffe", "impala", "warthog"
    ])
    label_map = {s: i for i, s in enumerate(species_list)}
    print(f"Label map: {label_map}")
    
    # Read manifest
    print(f"\nReading manifest from {args.manifest_uri}")
    df = read_manifest_from_gcs(args.manifest_uri)
    print(f"Total images: {len(df):,}")
    
    # Process each split
    for split in ["train", "val", "test"]:
        n = len(df[df["split"] == split])
        print(f"\nProcessing {split} ({n:,} images)...")
        
        # Fewer shards for smaller splits
        split_shards = args.num_shards if split == "train" else max(2, args.num_shards // 5)
        
        write_tfrecords_for_split(
            df, split, label_map,
            args.output_dir, args.image_size, split_shards
        )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
