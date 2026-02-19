"""
Custom training script for census income prediction - PRODUCTION VERSION
Uses sklearn Pipeline to bundle preprocessing + model together.
This ensures predictions work with raw data (no manual preprocessing needed).
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df

def preprocess_target(df, target_column):
    """
    Separate features and target, clean target column.
    Returns X (features) and y (binary target).
    """
    print(f"\n🔧 Preprocessing target column...")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Clean target column (handle whitespace and periods)
    if y.dtype == 'object':
        y = y.str.strip()
        y = y.str.replace('.', '', regex=False)
    
    # Convert to binary: 1 for >50K, 0 for <=50K
    y = (y == '>50K').astype(int)
    
    # Verify both classes exist
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError(f"Target column has only one class! Check your data.")
    
    print(f"✅ Target distribution: {y.value_counts().to_dict()}")
    print(f"   Class 0 (<=50K): {(y == 0).sum():,} samples ({(y == 0).mean()*100:.1f}%)")
    print(f"   Class 1 (>50K): {(y == 1).sum():,} samples ({(y == 1).mean()*100:.1f}%)")
    
    return X, y

def create_preprocessing_pipeline(X):
    """
    Create preprocessing pipeline that handles categorical encoding.
    This pipeline will be saved WITH the model so predictions work on raw data.
    
    Key insight: By using a Pipeline, we don't need to manually preprocess
    prediction data - the model does it automatically!
    """
    print(f"\n🏗️  Building preprocessing pipeline...")
    
    # Identify categorical and numeric columns
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"   Categorical features ({len(categorical_features)}): {categorical_features}")
    print(f"   Numeric features ({len(numeric_features)}): {numeric_features}")
    
    # Create preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            # One-hot encode categorical features
            # handle_unknown='ignore' is CRITICAL for production:
            # - Handles new categories in prediction data that weren't in training
            # - Prevents errors when deploying to production
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
            
            # Keep numeric features as-is (could add StandardScaler if needed)
            ('num', 'passthrough', numeric_features)
        ],
        remainder='drop'  # Drop any columns not specified above
    )
    
    print(f"✅ Preprocessing pipeline created")
    return preprocessor

def create_model_pipeline(preprocessor, hyperparameters):
    """
    Create full pipeline: preprocessing → model.
    This is the object we'll save and deploy.
    """
    print(f"\n🔗 Creating full pipeline (preprocessing + model)...")
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(**hyperparameters, random_state=42))
    ])
    
    print(f"✅ Pipeline created with {len(pipeline.steps)} steps:")
    for i, (name, step) in enumerate(pipeline.steps, 1):
        print(f"   {i}. {name}: {type(step).__name__}")
    
    return pipeline

def train_model(pipeline, X_train, y_train, hyperparameters):
    """Train the full pipeline on training data."""
    print(f"\n🚀 Training pipeline with hyperparameters:")
    for key, value in hyperparameters.items():
        print(f"   {key}: {value}")
    
    print(f"\n   Training on {len(X_train):,} samples...")
    print(f"   Input shape: {X_train.shape}")
    
    # Fit the entire pipeline (preprocessing + model)
    pipeline.fit(X_train, y_train)
    
    print("✅ Training complete!")
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    """
    Evaluate pipeline performance.
    Note: We pass RAW data - the pipeline handles preprocessing!
    """
    print(f"\n📊 Evaluating pipeline on {len(X_test):,} test samples...")
    
    # Predictions (pipeline handles preprocessing internally)
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n📈 Model Performance:")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   ROC AUC: {roc_auc:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))
    
    # Feature importance (from the classifier inside the pipeline)
    classifier = pipeline.named_steps['classifier']
    print(f"\n🔍 Top 10 Most Important Features:")
    
    # Get feature names after preprocessing
    preprocessor = pipeline.named_steps['preprocessor']
    feature_names = preprocessor.get_feature_names_out()
    
    # Get feature importances
    importances = classifier.feature_importances_
    
    # Sort and display top 10
    indices = np.argsort(importances)[-10:][::-1]
    for i, idx in enumerate(indices, 1):
        print(f"   {i}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }

def save_model(pipeline, model_dir):
    """
    Save trained pipeline to GCS or local path.
    
    CRITICAL: We save the ENTIRE pipeline (preprocessing + model).
    This means predictions can work on raw data without manual preprocessing!
    """
    print(f"\n💾 Saving pipeline to: {model_dir}")
    
    # Always create local copy first
    local_model_dir = "/tmp/model"
    os.makedirs(local_model_dir, exist_ok=True)
    local_model_path = os.path.join(local_model_dir, 'model.joblib')
    
    # Save pipeline locally
    joblib.dump(pipeline, local_model_path)
    print(f"✅ Pipeline saved locally to: {local_model_path}")
    
    # Get file size
    size_mb = os.path.getsize(local_model_path) / (1024 * 1024)
    print(f"   File size: {size_mb:.2f} MB")
    
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
            print(f"✅ Pipeline uploaded to GCS: gs://{bucket_name}/{blob_path}/model.joblib")
            
        except Exception as e:
            print(f"❌ Error uploading to GCS: {e}")
            print(f"   Pipeline is still available locally at: {local_model_path}")
            raise
    else:
        # Save to local filesystem
        os.makedirs(model_dir, exist_ok=True)
        final_path = os.path.join(model_dir, 'model.joblib')
        
        if final_path != local_model_path:
            import shutil
            shutil.copy(local_model_path, final_path)
        
        print(f"✅ Pipeline saved to: {final_path}")

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
    print("PRODUCTION VERSION (with sklearn Pipeline)")
    print("=" * 60)
    
    # 1. Load data
    df = load_data(args.data_path)
    
    # 2. Preprocess target (only the target, not features!)
    X, y = preprocess_target(df, args.target_column)
    
    # 3. Split data BEFORE creating pipeline
    # Important: Split on RAW data, then pipeline handles preprocessing
    print(f"\n✂️  Splitting data (80/20 train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training set: {len(X_train):,} samples")
    print(f"   Test set: {len(X_test):,} samples")
    
    # 4. Create preprocessing pipeline (learns from training data only!)
    preprocessor = create_preprocessing_pipeline(X_train)
    
    # 5. Create full pipeline (preprocessing + model)
    hyperparameters = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'min_samples_split': args.min_samples_split,
    }
    pipeline = create_model_pipeline(preprocessor, hyperparameters)
    
    # 6. Train pipeline
    pipeline = train_model(pipeline, X_train, y_train, hyperparameters)
    
    # 7. Evaluate (pass RAW test data - pipeline handles preprocessing!)
    metrics = evaluate_model(pipeline, X_test, y_test)
    
    # 8. Save pipeline
    # Use Vertex AI's AIP_MODEL_DIR if available (set automatically by base_output_dir)
    model_dir = os.environ.get('AIP_MODEL_DIR', args.model_dir)
    save_model(pipeline, model_dir)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print(f"   Final Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"   Final ROC AUC: {metrics['roc_auc']:.4f}")
    print("\n💡 Pipeline saved - ready for deployment!")
    print("   Predictions will work on RAW data (no manual preprocessing needed)")
    print("=" * 60)

if __name__ == '__main__':
    main()