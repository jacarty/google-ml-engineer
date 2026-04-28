# GCP ML Engineer Certification Study

Hands-on labs and notes for the [Google Cloud Professional Machine Learning Engineer](https://cloud.google.com/learn/certification/machine-learning-engineer) certification.

## Labs

| Lab | Topic | Status |
|-----|-------|--------|
| Lab 1 | Feature Engineering with BigQuery ML | ✅ Complete |
| Lab 2 | End-to-End Pipeline in Vertex AI (AutoML + Custom Training + Serving) | ✅ Complete |
| Lab 3 | Hyperparameter Tuning with Vertex AI Vizier | ✅ Complete |
| Lab 4 | Model Monitoring & Drift Detection | ✅ Complete |
| Lab 5 | MLOps Services | ✅ Complete |
| Lab 6 | Vertex AI Agent Builder (RAG) | ✅ Complete |
| Lab 7 | Text Classification with Stack Overflow Data | ✅ Complete |
| Lab 8 | Image Classification with Satellite Data | ✅ Complete |
| Lab 9 | Time Series Forecasting | ✅ Complete |
| Lab 10 | Vertex AI Pipelines | ✅ Complete |
| Mini-Lab A | Custom Prediction Routine (CPR) | ✅ Complete |
| Mini-Lab B | Explainability (Sampled Shapley) | ✅ Complete |
| Mini-Lab C | TFRecord Pipeline | ✅ Complete |
| Mini-Lab D | Shadow Deployment | ✅ Complete |
| Mini-Lab E | Dataflow + RunInference | ✅ Complete |

| Project | Domain | Status | Key Outcome |
|---------|--------|--------|-------------|
| Project 1: Support Ticket Routing | Text / NLP | ✅ Complete | KFP pipeline with conditional deploy, CPR endpoint, Weighted F1 0.5156 |
| Project 2: Serengeti Wildlife Species ID | Image / Vision | ✅ Complete | MobileNetV2 transfer learning, 72.6% test accuracy (10 species), occlusion sensitivity explainability |
| Project 3: Bitcoin Volatility Forecasting | Time Series / Finance | ✅ Complete | ARIMA_PLUS + GBT + LSTM ensemble; 26% RMSE improvement over climatology baseline; ARIMA_PLUS_XREG with exogenous regressors |


See [`ml_labs/ml_labs_plan.md`](ml_labs/ml_labs_plan.md) for the full study plan.

## Reference
- [`ml_crash_course/`](ml_crash_course/) — Foundational ML notebooks (linear regression, classification, fairness)

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install jupyter pandas scikit-learn google-cloud-aiplatform
```

> **Cost tip:** Delete Vertex AI endpoints, models, and monitoring jobs after each lab session. Run `gcloud ai endpoints list --region=us-central1` to audit running resources.
