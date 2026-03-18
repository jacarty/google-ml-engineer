# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

This is a study repository for the **Google Cloud Professional Machine Learning Engineer Certification**. It contains hands-on labs using GCP services (BigQuery ML, Vertex AI, Cloud Storage) and a reference cheat sheet. There are no automated tests or build steps — the primary workflow is running Jupyter notebooks.

## Environment

- Python 3.13 via `.venv/` (created with `python3.13 -m venv .venv`)
- Docker containers for Vertex AI training jobs use **Python 3.10-slim**
- Activate venv: `source .venv/bin/activate`

## Repository Structure

```
ml_crash_course/     # Foundational ML notebooks (linear regression, classification, fairness)
ml_labs/
  ml_labs_plan.md                  # Master plan: all 10 labs + 5 mini-labs (ALL COMPLETE)
  lab1_bigquery_ml/                # BigQuery ML feature engineering
  lab2_vertex_ai/                  # Vertex AI end-to-end pipeline: AutoML + custom training + serving
  lab3_hyperparameter_tuning/      # Hyperparameter tuning with Vertex AI Vizier
  lab4_monitoring/                 # Model monitoring and drift detection
  lab5_mlops_services/             # Feature Store, Experiments, Metadata
  lab6_agents/                     # Vertex AI Agent Builder (RAG)
  lab7_text_classification/        # Text classification (Stack Overflow)
  lab8_image_classification/       # Image classification (EuroSAT satellite)
  lab9_time_series_forecasting/    # Time series forecasting (NOAA weather)
  lab10_vertex_ai_pipelines/       # KFP pipelines with conditional deployment
  mini_lab_a_cpr/                  # Custom Prediction Routine (CPR)
  mini_lab_b_explainability/       # Sampled Shapley + Integrated Gradients
  mini_lab_c_tfrecord/             # TFRecord pipeline and benchmarking
  mini_lab_d_traffic_splitting/    # Champion/challenger traffic splitting
  mini_lab_e_dataflow/             # Dataflow + RunInference
ml_projects/
  ml_projects_plan.md              # Reinforcement projects plan (3 projects, not yet started)
general_notes.md                   # GCP ML exam weak area analysis and decision frameworks
```

## Common GCP Commands

Audit running GCP resources to avoid unexpected billing:
```bash
gcloud compute instances list
gcloud ai models list --region=us-central1
gcloud ai endpoints list --region=us-central1
```

Always delete resources after lab sessions to control costs.

## Custom Training Container Pattern

Labs 2 and 3 use the same Docker-based pattern for Vertex AI custom training jobs:

1. **Training script** (`lab2_vertex_train.py`, `lab3_train.py`) — accepts hyperparameters as CLI args, reads data from GCS (`gs://...` paths), saves `model.joblib` to GCS via `google-cloud-storage`
2. **Dockerfile** — `FROM python:3.10-slim`, installs deps, copies script, sets ENTRYPOINT
3. **Serving script** (`serve.py`, Lab 2) — Flask app on port 8080, reads model from `AIP_STORAGE_URI` env var (set by Vertex AI), exposes `/health` and `/predict` endpoints

Build and push containers to Artifact Registry before submitting Vertex AI jobs. The notebooks contain the `docker build` / `docker push` and `aiplatform` SDK calls.

## Data

- Dataset: UCI Census Income (predict whether income >50K)
- Source file: `ml_labs/lab2/census_income.csv`
- GCS bucket used across labs for data and model artifacts
- Target column: `income_bracket` (binary: `>50K` or `<=50K`)
- Preprocessing: one-hot encode categoricals, strip whitespace from target

## Key GCP Services Used

| Lab | Primary Services |
|-----|-----------------|
| Lab 1 | BigQuery ML — `CREATE MODEL`, `ML.EVALUATE`, `ML.PREDICT`, `TRANSFORM` |
| Lab 2 | Vertex AI Datasets, AutoML Tabular, Custom Training Jobs, Endpoints, Batch Prediction |
| Lab 3 | Vertex AI Hyperparameter Tuning (`HyperparameterTuningJob`), `cloudml-hypertune` for reporting metrics |
| Lab 4 | Vertex AI Model Monitoring, drift/skew detection, `ModelDeploymentMonitoringJob` |
| Lab 5 | Vertex AI Feature Store, Experiments, Metadata — lineage tracking |
| Lab 6 | Vertex AI Agent Builder, Vertex AI Search, RAG, grounding |
| Lab 7 | Vertex AI Custom Training (text), TF-IDF, embedding models |
| Lab 8 | Vertex AI Custom Training (images), MobileNetV2 transfer learning, AutoML Vision |
| Lab 9 | BigQuery ML `ARIMA_PLUS`, GBT, LSTM time series comparison |
| Lab 10 | KFP pipelines, `dsl.Condition`, lightweight components, pipeline scheduling |
| Mini-Lab A | Custom Prediction Routine (CPR), `SklearnPredictor`, Artifact Registry |
| Mini-Lab B | Vertex AI Explainability, Sampled Shapley, Integrated Gradients |
| Mini-Lab C | TFRecord format, `tf.data`, sharding, read performance benchmarking |
| Mini-Lab D | Vertex AI traffic splitting, champion/challenger, prediction logging to BigQuery |
| Mini-Lab E | Dataflow, Apache Beam, `RunInference`, DirectRunner vs DataflowRunner |

## Hyperparameter Tuning (Lab 3)

Training scripts use `cloudml-hypertune` to report metrics back to Vizier:
```python
import hypertune
hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag='accuracy',
    metric_value=accuracy
)
```
The metric tag must match `metric_spec` in the `HyperparameterTuningJob` config.

## Exam Prep Context

`general_notes.md` contains the most important decision frameworks for the certification exam — particularly the serving architecture selection table, the abstraction ladder (Pre-trained APIs → AutoML → CPR → Custom Container → GKE), and the MLOps deployment patterns (shadow, A/B, canary, blue/green).

`ml_projects/ml_projects_plan.md` describes 3 reinforcement projects (Support Ticket Routing, Wildlife Species ID, Bitcoin Volatility Forecasting) that apply all labs + mini-labs to new domains.
