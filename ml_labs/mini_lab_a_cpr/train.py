"""
Minimal training script for Mini-Lab A.
Produces a model.joblib artifact compatible with CPR's SklearnPredictor.
"""

import argparse
import os
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('AIP_MODEL_DIR', '/tmp/model'))
    args = parser.parse_args()

    # Load data
    print(f"Loading data from: {args.data_path}")
    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df):,} rows")

    # Prepare features and target
    target = 'income_bracket'
    X = df.drop(columns=[target])
    y = (df[target].str.strip() == '>50K').astype(int)

    # One-hot encode categoricals
    X = pd.get_dummies(X, drop_first=True)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Train with Lab 3 best hyperparameters
    model = GradientBoostingClassifier(
        n_estimators=385,
        max_depth=9,
        learning_rate=0.028,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))

    # Save model artifact locally first, then upload to GCS
    local_dir = '/tmp/model_output'
    os.makedirs(local_dir, exist_ok=True)

    model_path = os.path.join(local_dir, 'model.joblib')
    joblib.dump(model, model_path)
    print(f"✅ Model saved locally: {model_path}")

    feature_path = os.path.join(local_dir, 'feature_columns.joblib')
    joblib.dump(list(X_train.columns), feature_path)
    print(f"✅ Feature columns saved locally: {feature_path}")

    # Upload to GCS using google-cloud-storage
    from google.cloud import storage

    # Parse bucket and prefix from gs:// path
    gcs_path = args.model_dir.replace('gs://', '')
    bucket_name = gcs_path.split('/')[0]
    prefix = '/'.join(gcs_path.split('/')[1:]).rstrip('/')

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for filename in ['model.joblib', 'feature_columns.joblib']:
        blob = bucket.blob(f"{prefix}/{filename}")
        blob.upload_from_filename(os.path.join(local_dir, filename))
        print(f"✅ Uploaded {filename} to gs://{bucket_name}/{prefix}/{filename}")

if __name__ == '__main__':
    main()
