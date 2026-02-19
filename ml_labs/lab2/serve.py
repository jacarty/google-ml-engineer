"""
Flask serving script for Vertex AI custom prediction container.
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import logging
logging.getLogger('werkzeug').setLevel(logging.WARNING)

app = Flask(__name__)

model = None

def load_model():
    """Load model from AIP_STORAGE_URI (GCS path set by Vertex AI)."""
    global model
    storage_uri = os.environ.get('AIP_STORAGE_URI', '/tmp/model')
    local_model_path = '/tmp/model/model.joblib'
    os.makedirs('/tmp/model', exist_ok=True)

    if storage_uri.startswith('gs://'):
        from google.cloud import storage as gcs
        
        # Parse gs://bucket/path
        gcs_path = storage_uri.replace('gs://', '')
        bucket_name = gcs_path.split('/')[0]
        blob_prefix = '/'.join(gcs_path.split('/')[1:]).rstrip('/')
        blob_name = f"{blob_prefix}/model.joblib"
        
        print(f"Downloading gs://{bucket_name}/{blob_name}")
        client = gcs.Client()
        bucket = client.bucket(bucket_name)
        bucket.blob(blob_name).download_to_filename(local_model_path)
    else:
        local_model_path = os.path.join(storage_uri, 'model.joblib')

    print(f"Loading model from: {local_model_path}")
    model = joblib.load(local_model_path)
    print(f"Model loaded: {type(model).__name__}")

@app.route('/health', methods=['GET'])
def health():
    if model is not None:
        return jsonify({'status': 'healthy'}), 200
    return jsonify({'status': 'not ready'}), 503

@app.route('/predict', methods=['POST'])
def predict():
    try:
        content = request.get_json()
        instances = content.get('instances', [])
        df = pd.DataFrame(instances)
        predictions = model.predict(df).tolist()
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    port = int(os.environ.get('AIP_HTTP_PORT', 8080))
    print(f"Starting prediction server on port {port}")
    app.run(host='0.0.0.0', port=port)