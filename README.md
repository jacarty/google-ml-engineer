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
| Lab 6 | AI Agents | 🔄 In Progress |

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
