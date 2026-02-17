"""
Custom training script for census income prediction.
This script runs in Docker containers on Vertex AI.
"""

import argparse
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import os

def load_data(data_path):
    """Load data from GCS or local path."""
    print(f"📂 Loading data from: {data_path}")
    
    if data_path.endswith('.csv'):
        if data_path.startswith('gs://'):
            # Use gcsfs for GCS paths
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            with fs.open(data_path, 'rb') as f:
                df = pd.read_csv(f)
        else:
            df = pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        if data_path.startswith('gs://'):
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            with fs.open(data_path, 'rb') as f:
                df = pd.read_parquet(f)
        else:
            df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df

def preprocess_data(df, target_column):
    """Prepare features and target."""
    print(f"\n🔧 Preprocessing data...")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Clean target column (handle whitespace and periods)
    if y.dtype == 'object':
        y = y.str.strip()
        y = y.str.replace('.', '', regex=False)
    
    # Convert to binary
    y = (y == '>50K').astype(int)
    
    # Verify both classes exist
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError(f"Target column has only one class! Check your data.")
    
    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)
    
    print(f"✅ Features: {X.shape[1]} columns")
    print(f"✅ Target distribution: {y.value_counts().to_dict()}")
    
    return X, y

def train_model(X_train, y_train, hyperparameters):
    """Train gradient boosting model."""
    print(f"\n🚀 Training model with hyperparameters:")
    for key, value in hyperparameters.items():
        print(f"   {key}: {value}")
    
    model = GradientBoostingClassifier(**hyperparameters, random_state=42)
    model.fit(X_train, y_train)
    
    print("✅ Training complete!")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    print(f"\n📊 Evaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n📈 Model Performance:")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   ROC AUC: {roc_auc:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }

def save_model(model, model_dir):
    """Save trained model to GCS or local path."""
    print(f"\n💾 Saving model to: {model_dir}")
    
    # Always create local copy first
    local_model_dir = "/tmp/model"
    os.makedirs(local_model_dir, exist_ok=True)
    local_model_path = os.path.join(local_model_dir, 'model.joblib')
    
    # Save model locally
    joblib.dump(model, local_model_path)
    print(f"✅ Model saved locally to: {local_model_path}")
    
    # If target is GCS, upload
    if model_dir.startswith('gs://'):
        try:
            from google.cloud import storage
            
            # Parse GCS path: gs://bucket/path/to/dir
            gcs_path = model_dir.replace('gs://', '')
            bucket_name = gcs_path.split('/')[0]
            blob_path = '/'.join(gcs_path.split('/')[1:]).rstrip('/')
            
            # Upload to GCS
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(f"{blob_path}/model.joblib")
            
            blob.upload_from_filename(local_model_path)
            print(f"✅ Model uploaded to GCS: gs://{bucket_name}/{blob_path}/model.joblib")
            
        except Exception as e:
            print(f"❌ Error uploading to GCS: {e}")
            print(f"   Model is still available locally at: {local_model_path}")
            raise
    else:
        # Save to local filesystem
        os.makedirs(model_dir, exist_ok=True)
        final_path = os.path.join(model_dir, 'model.joblib')
        
        if final_path != local_model_path:
            import shutil
            shutil.copy(local_model_path, final_path)
        
        print(f"✅ Model saved to: {final_path}")

def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser()
    
    # Data arguments
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to training data (GCS or local)')
    parser.add_argument('--target-column', type=str, default='income_bracket',
                       help='Name of target column')
    parser.add_argument('--model-dir', type=str, default='/tmp/model',
                       help='Directory to save trained model (GCS or local)')
    
    # Hyperparameters
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of boosting stages')
    parser.add_argument('--max-depth', type=int, default=5,
                       help='Maximum depth of trees')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                       help='Learning rate')
    parser.add_argument('--min-samples-split', type=int, default=2,
                       help='Minimum samples required to split a node')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CUSTOM TRAINING - CENSUS INCOME PREDICTION")
    print("=" * 60)
    
    # 1. Load data
    df = load_data(args.data_path)
    
    # 2. Preprocess
    X, y = preprocess_data(df, args.target_column)
    
    # 3. Split data
    print(f"\n✂️  Splitting data (80/20 train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training set: {len(X_train):,} samples")
    print(f"   Test set: {len(X_test):,} samples")
    
    # 4. Train model
    hyperparameters = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'min_samples_split': args.min_samples_split,
    }
    model = train_model(X_train, y_train, hyperparameters)
    
    # 5. Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # 6. Save model
    save_model(model, args.model_dir)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print(f"   Final Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"   Final ROC AUC: {metrics['roc_auc']:.4f}")
    print("=" * 60)

if __name__ == '__main__':
    main()