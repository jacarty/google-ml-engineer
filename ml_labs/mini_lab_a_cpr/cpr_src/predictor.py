"""
Custom Prediction Routine for census income classification.

Replaces the Lab 2 Flask serve.py with a structured Predictor class.
Vertex AI handles HTTP server, routing, health checks — we only write ML logic.
"""

import os
import joblib
import numpy as np
import pandas as pd
from google.cloud.aiplatform.prediction.sklearn.predictor import SklearnPredictor


class CensusPredictor(SklearnPredictor):
    """Census income predictor using CPR.

    Compared to Lab 2's serve.py:
    - No Flask, no routes, no health endpoint — Vertex AI handles all of that
    - Same preprocessing logic, structured into dedicated methods
    - Postprocessing adds class labels and probabilities (Lab 2 only returned raw predictions)
    """

    def __init__(self):
        super().__init__()
        self._feature_columns = None

    def load(self, artifacts_uri: str) -> None:
        """Load model and feature columns from GCS.

        SklearnPredictor.load() handles joblib model loading.
        We extend it to also load the feature column list saved during training.
        """
        super().load(artifacts_uri)

        # artifacts_uri could be GCS or local — download_model_artifacts handles both
        from google.cloud.aiplatform.utils import prediction_utils
        prediction_utils.download_model_artifacts(artifacts_uri)

        # After download, files are in the current working directory
        self._feature_columns = joblib.load("feature_columns.joblib")
        print(f"✅ Loaded {len(self._feature_columns)} feature columns")

    def preprocess(self, prediction_input: dict) -> pd.DataFrame:
        """Transform raw JSON input into model-ready DataFrame.

        In Lab 2's serve.py, this was embedded inside the /predict route:
            content = request.get_json()
            instances = content.get('instances', [])
            df = pd.DataFrame(instances)

        Here it's a dedicated method with proper validation and encoding.

        Args:
            prediction_input: Dict with 'instances' key containing list of dicts.
                Each dict has raw feature names/values (before one-hot encoding).

        Returns:
            DataFrame with one-hot encoded features matching training columns.
        """
        # Extract instances from the prediction input
        instances = prediction_input.get("instances", prediction_input)

        # Convert to DataFrame
        df = pd.DataFrame(instances)

        # One-hot encode categorical features (same as training)
        df = pd.get_dummies(df, drop_first=True)

        # Align columns with training features
        # - Add missing columns (set to 0) — handles unseen categories
        # - Remove extra columns — handles categories not in training data
        df = df.reindex(columns=self._feature_columns, fill_value=0)

        return df

    def predict(self, instances: pd.DataFrame) -> np.ndarray:
        """Run inference. Inherited from SklearnPredictor — returns model.predict().

        We override to return predict_proba instead, giving us class probabilities
        that postprocess() can format nicely.
        """
        return self._model.predict_proba(instances)

    def postprocess(self, prediction_results: np.ndarray) -> dict:
        """Format prediction output with class labels and probabilities.

        In Lab 2's serve.py, this was just:
            return jsonify({'predictions': predictions})

        Here we add structured output with class labels and confidence scores.

        Args:
            prediction_results: Array of shape (n_samples, 2) with class probabilities.

        Returns:
            Dict with predictions in Vertex AI's expected format.
        """
        predictions = []
        for probs in prediction_results:
            predicted_class = int(np.argmax(probs))
            predictions.append({
                "predicted_label": ">50K" if predicted_class == 1 else "<=50K",
                "confidence": float(probs[predicted_class]),
                "probabilities": {
                    "<=50K": float(probs[0]),
                    ">50K": float(probs[1]),
                },
            })

        return {"predictions": predictions}
