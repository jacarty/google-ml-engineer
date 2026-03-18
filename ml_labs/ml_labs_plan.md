# GCP ML Engineer Certification — Complete Plan

**Goal:** Google Cloud Professional Machine Learning Engineer Certification  
**Approach:** Hands-on labs first, then theory reinforcement  
**Estimated Total Cost:** ~$290-370 (including exam fee)

---

## Progress Summary

| Lab | Status | Key Outcome |
|-----|--------|-------------|
| Lab 1: BigQuery ML (Feature Engineering) | ✅ Complete | Boosted trees: 86.23% accuracy |
| Lab 2: Vertex AI Pipeline (End-to-End) | ✅ Complete | Custom beat AutoML at 0.27% of the cost |
| Lab 3: Hyperparameter Tuning | ✅ Complete | Manual vs Bayesian optimization; best 87.59% |
| Lab 4: Monitoring & Drift Detection | ✅ Complete | Drift detection + response runbook |
| Lab 5: MLOps Services | ✅ Complete | Feature Store, Experiments, Metadata |
| Lab 6: Agent Builder (RAG) | ✅ Complete | Working RAG agent with citations |
| Lab 7: Text Classification | ✅ Complete | TF-IDF baseline (83.8%) beat neural (81%) |
| Lab 8: Image Classification | ✅ Complete | MobileNetV2 transfer learning; AutoML ~96.4%, Custom ~97.83% |
| Lab 9: Time Series Forecasting | ✅ Complete | ARIMA_PLUS vs GBT vs LSTM comparison |
| Lab 10: Vertex AI Pipelines | ✅ Complete | KFP conditional deploy; lightweight components; scheduling |
| Mini-Lab A: Custom Prediction Routine | ✅ Complete | CPR with SklearnPredictor, arm64/amd64 fix |
| Mini-Lab B: Explainability | ✅ Complete | Sampled Shapley + Integrated Gradients + local SHAP |
| Mini-Lab C: TFRecord Pipeline | ✅ Complete | CSV vs TFRecord benchmarking; 1.6x read speedup; sharding |
| Mini-Lab D: Traffic Splitting | ✅ Complete | Champion/challenger 80/20 split; BQ prediction logging; canary rollout |
| Mini-Lab E: Dataflow + RunInference | ✅ Complete | Beam RunInference on Dataflow; DirectRunner vs DataflowRunner; BQ output |

## Suggested Lab Order (Remaining)

All labs and mini-labs complete. Next: Phase 3 (Theory Reinforcement) and Phase 4 (Certification-Specific Prep).

---

## Design Principles

Every lab follows this pattern:

1. **Data lives in BigQuery** (public dataset) — explore and prepare with SQL
2. **Export to Cloud Storage** (CSV, TFRecord, or image files)
3. **Train in Vertex AI** (custom training with containers)
4. **Log to Experiments** (parameters, metrics, artifacts)
5. **Evaluate** (appropriate metrics for the problem type)
6. **Serve** (online endpoint or batch prediction — varies by lab)
7. **Track lineage** via Metadata (data → model → endpoint)

### Container Strategy

Default to Google's prebuilt training containers unless there's a specific reason to build custom. Each lab begins with a container decision.

| Scenario | Use |
|----------|-----|
| Standard TF/scikit-learn training, pip extras needed | **Prebuilt** + `requirements` param |
| Need custom system packages or non-standard base | **Custom Dockerfile** |
| Need identical train + serve environment | **Custom Dockerfile** (same image for both) |
| Using CPR for serving | **Prebuilt training** + CPR-built serving image |

### Standalone Labs

Every lab assumes all resources were deleted after the previous lab. Each lab includes setup from scratch.

---

## Phase 1: Foundation Labs (Labs 1–6)

### Lab 1 — Feature Engineering with BigQuery ML ✅

**Dataset:** Census income (BigQuery)  
**Focus:** BigQuery ML training, TRANSFORM pattern, algorithm selection  
**Key Skills:** Feature engineering, model evaluation (precision/recall/ROC AUC), train-serve skew prevention

### Lab 2 — End-to-End Pipeline in Vertex AI ✅

**Dataset:** Census income (exported to GCS)  
**Focus:** Full ML lifecycle — data → training → deployment → prediction  
**Key Skills:** AutoML vs custom training, Docker containerization, endpoint deployment, cost optimization

### Lab 3 — Hyperparameter Tuning ✅

**Dataset:** Census income  
**Focus:** Manual tuning vs Vertex AI Bayesian optimization  
**Key Skills:** Search space definition, parallel trials, cost-benefit analysis of tuning

### Lab 4 — Model Monitoring & Drift Detection ✅

**Dataset:** Census income + synthetic drift data  
**Focus:** Production monitoring, drift simulation, response procedures  
**Key Skills:** Skew/drift detection, alert configuration, drift response runbook

### Lab 5 — MLOps Services ✅

**Dataset:** Census income  
**Focus:** Retrofit existing pipeline with Vertex AI Experiments, Feature Store, Metadata  
**Key Skills:** Experiment tracking, feature serving (online/offline), ML lineage

### Lab 6 — Vertex AI Agent Builder ✅

**Dataset:** Certification notes and lab documentation  
**Focus:** RAG-based agent for document Q&A  
**Key Skills:** Datastore configuration, grounding vs fine-tuning, chunking strategy, citation

### Lab 7 — Text Classification with Stack Overflow Data ✅

**Dataset:** `bigquery-public-data.stackoverflow.posts_questions`  
**Task:** Predict tag (python, javascript, java, c#, html) from question text  
**Container:** Google prebuilt TF 2.15  
**Exam Relevance:** Text preprocessing, NLP pipelines, TF/Keras custom training, batch vs online serving

**Parts:**
- Setup — GCS bucket, prebuilt container selection
- Data exploration and preparation in BigQuery — top 5 tags, text cleaning, hash-based splits
- Text preprocessing and feature engineering — TF-IDF baseline vs TextVectorization
- Custom training with TensorFlow — Embedding + GlobalAveragePooling1D, Experiments logging
- Batch prediction — Vertex AI batch prediction on test set, JSONL format
- Metadata and lineage — trace BQ → GCS → training → model → predictions

**New Skills:** Text pipelines, TF/Keras for NLP, batch prediction, serving architecture decisions

---

### Lab 8 — Image Classification with Satellite Data ✅

**Dataset:** EuroSAT via TensorFlow Datasets (10 land use classes, 27k satellite images)  
**Task:** Multi-class satellite image classification  
**Container:** Prebuilt TF + custom prediction container  
**Exam Relevance:** Vision API vs AutoML vs custom, transfer learning, tf.data pipelines

**Results:**
- AutoML: ~96.4% accuracy
- Custom MobileNetV2 fine-tuned: ~97.83% accuracy
- Base64 serving pattern to solve gRPC/REST payload limits

**Key Learnings:**
- Baking preprocessing into SavedModel graph avoids payload size issues
- `tf.io.decode_base64` requires URL-safe base64 without padding
- Transfer learning (freeze → fine-tune) is highly effective for domain-specific image tasks

**New Skills:** Image pipelines, transfer learning, tf.data, base64 serving, pre-trained vs custom decision making

---

### Lab 9 — Time Series Forecasting with NOAA Weather Data ✅

**Dataset:** `bigquery-public-data.noaa_gsod` (Global Surface Summary of the Day)  
**Task:** Forecast daily temperature for JFK Airport  
**Container:** Prebuilt sklearn 1.0 for GBT, prebuilt TF 2.15 for LSTM, none for BQML  
**Exam Relevance:** ARIMA_PLUS, temporal features, BQML vs custom, ML.FORECAST, ML.EXPLAIN_FORECAST, time-aware splitting, regression metrics

**Parts:**
- Setup + Data Exploration — NOAA GSOD query, cleaning (9999.9 missing indicators), temporal feature engineering
- BigQuery ML ARIMA_PLUS — ML.FORECAST with confidence intervals, ML.EXPLAIN_FORECAST decomposition
- Custom GBT — time series as tabular regression, prebuilt sklearn container on Vertex AI
- Custom LSTM — 14-day sliding windows, prebuilt TF 2.15 container on Vertex AI
- Comparison + ML.PREDICT + Metadata + Cleanup

**Results:**

| Model | Test RMSE (°F) | Test MAE (°F) | Test MAPE (%) |
|-------|---------------|--------------|--------------|
| ARIMA_PLUS (BQML) | 7.07 | 5.46 | 11.2 |
| Gradient Boosted Trees | 2.36 | 1.79 | 3.5 |
| LSTM | 3.95 | 3.05 | 6.3 |

**Key Learnings:**
- GBT dominates when lag features provide yesterday's actual temperature (one-step-ahead)
- ARIMA_PLUS does true multi-step forecasting with just date + temp — 7°F RMSE over 4 years is respectable
- LSTM learns temporal patterns from sequences without manual lag features, but can't beat GBT with explicit lags
- ML.PREDICT does NOT work with ARIMA models — use ML.FORECAST instead
- ML.EXPLAIN_FORECAST decomposes predictions into trend, seasonal, and holiday components (unique to BQML)
- Time-aware splitting is mandatory — random splits cause data leakage
- Cyclical encoding (sin/cos) essential for periodic features
- Prebuilt sklearn 1.0 container works; sklearn 1.3+ has broken google-cloud-bigquery dependency
- TF SavedModel format from Vertex AI (Keras 2) requires TFSMLayer to load locally in Keras 3
- TensorFlow on Apple Silicon M4 is extremely slow for training — submit to Vertex AI instead
- Python 3.13 breaks google-auth REST transport; use gRPC client directly for Metadata operations

**New Skills:** Temporal feature engineering (lag, rolling avg, cyclical encoding), ARIMA_PLUS, ML.FORECAST, ML.EXPLAIN_FORECAST, LSTM windowed data preparation, regression metrics (RMSE, MAE, MAPE), time-aware splitting

---

### Lab 10 — Vertex AI Pipelines: Orchestrating End-to-End Workflow ✅

**Dataset:** `bigquery-public-data.ml_datasets.penguins` (Palmer Penguins — 3 species, ~344 rows)  
**Task:** Build a Kubeflow pipeline: data prep → training → evaluation → conditional deployment  
**Container:** `python:3.10` base with pinned sklearn for lightweight KFP components; `sklearn-cpu.1-3` for serving  
**Exam Relevance:** Very high — Vertex AI Pipelines, conditional deployment, pipeline components, scheduling

**Parts:**
- Setup + Data Exploration — GCS bucket, KFP/pipeline-components install, penguins EDA
- Pipeline Components — 4 lightweight `@component` functions: data_prep, train_model, evaluate_model, deploy_model
- Pipeline Definition + Execution — `@dsl.pipeline` wiring, `dsl.Condition` for conditional deploy, two runs (deploy vs skip)
- Scheduling + Cleanup — `PipelineJobSchedule` with cron syntax, pause/resume/delete, resource cleanup

**Key Learnings:**
- Lightweight components are surprisingly powerful for small models — no separate training VM needed
- `dsl.Condition` is the exam's favorite: any "deploy only if X" question → `dsl.Condition`
- Imports must be inside components — each runs in an isolated container with no shared state
- Pipeline parameters make pipelines reusable — same pipeline, different thresholds/hyperparameters
- Scheduling ≠ event-triggered: `PipelineJobSchedule` (cron) vs Cloud Functions → `PipelineJob.submit()` (event)
- Google's sklearn prediction containers have old Python/pip that can't install KFP; use `python:3.10` + pinned deps instead
- Multi-output components: reference by name (`outputs["Output"]`) not `.output`
- Container consistency (again!): pin sklearn versions in training components to match the serving container

**New Skills:** Kubeflow Pipelines, `@component` decorator, `dsl.Condition`, pipeline parameters, `PipelineJobSchedule`, lightweight vs heavyweight component patterns

---

## Mini-Labs

Target specific gaps identified from practice questions. Each is 1-3 hours.

| Mini-Lab | Targets | Summary |
|----------|---------|---------|
| **A: Custom Prediction Routine (CPR)** | Abstraction level selection | Rebuild Lab 2 serving with CPR Predictor class instead of Flask |
| **B: Explainability (Sampled Shapley)** | Explainability methods | Add explanations to census model, compare Shapley vs Integrated Gradients vs XRAI |
| **C: TFRecord Pipeline** | Data formats | Convert census CSV → TFRecord, benchmark read speed, add sharding |
| **D: Shadow Deployment** | Deployment patterns | Traffic splitting with champion/challenger, document shadow vs A/B vs canary |
| **E: Dataflow + RunInference** | In-pipeline inference | Beam pipeline with RunInference on Dataflow, DirectRunner vs DataflowRunner |

---

### Mini-Lab A — Custom Prediction Routine (CPR) ✅

**Dataset:** `bigquery-public-data.ml_datasets.census_adult_income` (32,561 rows, binary income classification)  
**Task:** Serve a GradientBoostingClassifier using CPR instead of a custom Flask container  
**Model:** sklearn GradientBoostingClassifier with Lab 3's best hyperparameters  
**Environment:** Custom training on Vertex AI (`python:3.10-slim`), CPR serving via `SklearnPredictor`  
**Exam Relevance:** Very high — "custom model + preprocessing + minimize code" → CPR

**Parts:**
- Setup + Model Training — Census data from BigQuery, GBT training on Vertex AI via `CustomJob`
- Build the CPR Predictor — `CensusPredictor(SklearnPredictor)` with preprocess/postprocess overrides
- Build CPR Container, Deploy & Test — `LocalModel.build_cpr_model()`, push to Artifact Registry, deploy to endpoint
- Comparison, Reflection & Cleanup — Side-by-side Lab 2 (Flask) vs CPR, exam pattern cheat sheet

**Side-by-Side: Lab 2 (Flask) vs CPR:**

| Dimension | Lab 2 (Flask Container) | Mini-Lab A (CPR) |
|-----------|------------------------|------------------|
| Serving code | `serve.py` (~45 lines) + Dockerfile (~15 lines) | `predictor.py` (~70 lines incl. docstrings) |
| Files you wrote | `serve.py`, `Dockerfile`, `requirements.txt` | `predictor.py`, `requirements.txt` |
| HTTP server | Flask (you wrote + configured) | Vertex AI model server (auto-generated) |
| Dockerfile | Hand-written | Auto-generated by `build_cpr_model()` |
| Model loading | Custom `gsutil cp` + `joblib.load()` | `SklearnPredictor.load()` (inherited) |
| Output format | Raw predictions only | Labels + confidence + probabilities |

**Key Learnings:**
- CPR eliminates HTTP plumbing, not ML logic — same preprocessing, structured better
- `SklearnPredictor` handles model loading; override `load()` to add extra artifacts
- `base_image` parameter overrides CPR's default Python 3.7 — essential for version compatibility
- `build_cpr_model()` generates Dockerfile + builds image in one call
- `CustomJob` (not `CustomContainerTrainingJob`) when you don't have a serving container yet at submission time
- CPR sits at the sweet spot on the abstraction spectrum: Pre-trained APIs → AutoML → **CPR** → Custom Container → Self-managed (GKE)
- Apple Silicon (arm64) images must use `--platform linux/amd64` for Vertex AI compatibility
- `joblib` cannot write directly to `gs://` — save to `/tmp` then upload via GCS client

**New Skills:** CPR `SklearnPredictor` interface (load/preprocess/predict/postprocess), `LocalModel.build_cpr_model()`, `push_image()`, CPR vs custom container decision framework

---

### Mini-Lab B — Model Explainability: SHAP Analysis & Decision Framework ✅

**Dataset:** `bigquery-public-data.ml_datasets.census_adult_income` (32,561 rows, binary income classification)  
**Task:** Train a Keras NN, run local SHAP analysis, build exam-ready explainability decision framework  
**Model:** Keras Sequential (99 input features → 128 → 64 → 1 sigmoid)  
**Training:** Vertex AI CustomJob with prebuilt TF 2.15 container  
**Exam Relevance:** Very high — ExplanationSpec configuration, method selection (Sampled Shapley vs Integrated Gradients vs XRAI), feature attributions

**Parts:**
- Setup + TF Model Training — Preprocess locally (one-hot + standardize → 99 features), train on Vertex AI
- Local SHAP Analysis — KernelExplainer with force plots, summary plots, dependence plots
- Decision Framework + Cleanup — ExplanationSpec reference, method selection guide, exam patterns

**Results:**

| Metric | Value |
|--------|-------|
| Test Accuracy | 85.83% |
| ROC AUC | 0.9145 |
| Precision (>50K) | 75% |
| Recall (>50K) | 62% |

Top SHAP features (global importance): capital_gain, age, education_num, hours_per_week, marital status

**Explainability Method Decision Framework:**

| Method | Model Types | Best For |
|--------|------------|----------|
| Sampled Shapley | Any (sklearn, TF, XGBoost) | Tabular, model-agnostic |
| Integrated Gradients | Differentiable only (TF/Keras, PyTorch) | Neural nets, faster for TF |
| XRAI | Image models only | Image classification (region-level) |

**Key Learnings:**
- SHAP KernelExplainer is model-agnostic — conceptually identical to Vertex AI's Sampled Shapley
- Feature attributions are local (per-instance); summary plots aggregate to global importance
- ExplanationSpec is set at model upload time, not deploy time
- `ExplanationMetadata` (input/output tensor mapping) is optional for TF2 SavedModels (auto-detected)
- Sampled Shapley = universal (any model); Integrated Gradients = neural net specialist (faster, gradient-based); XRAI = image specialist (region segmentation)
- TF 2.15/Keras 2 SavedModels require `tf.saved_model.load()` locally under Keras 3, not `load_model()`
- Pickle protocol compatibility: models saved with Python 3.13 require `protocol=4` to be loadable by Python 3.10 serving containers

**New Skills:** SHAP KernelExplainer, force plots, summary plots, dependence plots, ExplanationSpec configuration, ExplanationMetadata, Sampled Shapley vs Integrated Gradients vs XRAI decision framework

---

### Mini-Lab C — TFRecord Pipeline & Benchmarking ✅

**Dataset:** `bigquery-public-data.chicago_taxi_trips.taxi_trips` (~750k rows, credit card payments only)  
**Task:** Binary classification — predict whether trip receives a tip (tip > 0)  
**Environment:** Local only (no Vertex AI)  
**Exam Relevance:** TFRecord format, tf.data pipeline, sharding strategy, data pipeline optimization

**Parts:**
- Setup + Data Preparation — Query Chicago Taxi from BigQuery, hash-based train/test split, CSV baseline
- TFRecord Fundamentals — tf.train.Example, three Feature types (FloatList, Int64List, BytesList), helper functions, write/read/verify
- Sharding — 10-shard write, count verification, file size comparison
- tf.data Pipelines — Three pipelines: CSV (make_csv_dataset), single TFRecord, sharded TFRecord with interleave
- Benchmark — Pipeline throughput comparison (3 runs each), speedup calculations
- Train & Compare — Same Keras model trained on CSV vs TFRecord pipelines, compare training time and accuracy

**Results:**

| Pipeline | Throughput | vs CSV |
|----------|-----------|--------|
| CSV (tf.data) | ~52,500 rec/s | baseline |
| TFRecord (single) | ~83,100 rec/s | 1.6x faster |
| TFRecord (sharded) | ~60,750 rec/s | 1.2x faster |

Training comparison: TFRecord 1.1x faster than CSV (51.6s vs 58.3s over 5 epochs). Identical accuracy (0.9441).

**Key Learnings:**
- TFRecord speed gain comes from binary read/parse efficiency, not file size reduction (TFRecords can be same size or larger than CSV)
- Feature description (schema) is mandatory at read time — no self-describing header like CSV; lose the schema, lose your data
- Sharded TFRecords are slower than single file on local NVMe SSD — the interleave coordination overhead exceeds the parallelism benefit when I/O is already fast
- Sharding's real payoff is distributed training (multiple workers reading different shards) and network storage (parallel GCS reads)
- On local SSD with data fitting in memory, the CSV→TFRecord speedup is real but modest; the gap widens dramatically at scale
- `tf.data.AUTOTUNE` and `prefetch(AUTOTUNE)` should always be included — they let TensorFlow optimize parallelism and overlap I/O with compute
- `tf.strings.to_hash_bucket_fast` is a quick-and-dirty way to handle string features in a dense model; production pipelines use embeddings or proper encoding
- Hash-based splitting provides deterministic reproducible train/test splits

**New Skills:** tf.train.Example, tf.train.Feature types, TFRecordWriter, TFRecordDataset, feature_description schema, sharded writes, tf.data.Dataset.interleave, pipeline benchmarking

---

### Mini-Lab D — Traffic Splitting: Champion/Challenger Deployment ✅

**Dataset:** `bigquery-public-data.ml_datasets.census_adult_income` (32,561 rows, binary income classification)  
**Task:** Deploy two model versions to a single endpoint with traffic splitting, verify via BQ prediction logging, simulate canary rollout  
**Environment:** Local training (sklearn), Vertex AI serving (prebuilt `sklearn-cpu.1-3:latest`)  
**Exam Relevance:** Traffic splitting, canary vs A/B vs shadow deployment, prediction request-response logging, gradual rollout patterns

**Parts:**
- Setup + Data Prep — Census data from BigQuery, label encoding, train/test split
- Train Two Model Versions — Champion RF (100 trees, depth 10) vs Challenger RF (300 trees, depth 20), local evaluation
- Deploy with Traffic Split + Prediction Logging — Single endpoint, 80/20 split, BQ request-response logging enabled
- Send Predictions + Verify Split — 150 requests, query BQ logs to confirm per-model routing
- Canary Rollout — 80/20 → 50/50 → 0/100, send requests at each stage, verify in BQ
- Deployment Strategy Decision Framework — Canary vs A/B vs shadow comparison with exam patterns
- Cleanup — Undeploy (0% traffic first), delete endpoint, models, GCS artifacts, BQ dataset

**Results:**

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Champion (100 trees, depth 10) | 85.66% | 81.11% | 54.03% | 64.85% |
| Challenger (300 trees, depth 20) | 86.17% | 77.42% | 61.45% | 68.51% |

Prediction disagreement: 5.7% (369/6491 test instances)

Traffic split verification (80/20 stage, 150 requests): Champion 84.0% (126), Challenger 16.0% (24) — within expected variance.

Overall (all stages, 300 requests): Champion 243, Challenger 57.

**Key Learnings:**
- `enable_request_response_logging=True` requires `request_response_logging_bq_destination_table` — the BQ destination URI cannot be empty
- Prediction logs include `deployed_model_id` column — this is how you verify which model served each request
- Log table name is auto-created as `request_response_logging` in the specified BQ dataset
- Undeploy order matters: must undeploy 0% traffic models first, otherwise Vertex AI rejects it (remaining traffic can't be 0%)
- `traffic_split={"0": 20, champion_id: 80}` — the `"0"` placeholder assigns traffic to the model being deployed in the same call
- Canary vs A/B vs shadow use the same exam decision framework: "minimize risk" → canary, "measure business impact" → A/B, "validate without affecting users" → shadow
- Canary and A/B use the same Vertex AI mechanism (`traffic_split`); the difference is intent, not infrastructure
- Shadow deployment is NOT native to Vertex AI — requires custom proxy (Cloud Function/Cloud Run)
- Prediction request-response logging has a delay (minutes) before logs appear in BQ — batch write behavior

**New Skills:** Vertex AI traffic splitting, `traffic_split` configuration, prediction request-response logging to BigQuery, canary rollout simulation, deployed model undeploy ordering

---

### Mini-Lab E — Dataflow + RunInference ✅

**Dataset:** `bigquery-public-data.ml_datasets.census_adult_income` (32,561 rows, binary income classification)  
**Task:** Build a Beam pipeline with in-pipeline ML inference, run locally then deploy to Dataflow  
**Model:** sklearn GradientBoostingClassifier (87.42% accuracy)  
**Environment:** Local Python 3.13 for DirectRunner, Google Cloud Dataflow for DataflowRunner  
**Exam Relevance:** Very high — RunInference, Beam I/O, DirectRunner vs DataflowRunner, in-pipeline inference pattern

**Parts:**
- Setup + Model Prep — Install `apache-beam[gcp]`, train sklearn GBT on census, upload pickle to GCS
- Beam Pipeline (DirectRunner) — Read CSV from GCS → parse → RunInference → write results; local execution
- Deploy to Dataflow (DataflowRunner) — Same pipeline code, swap runner; write predictions to BigQuery
- Compare & Analyse — Verify local and Dataflow predictions match; document latency patterns
- Cleanup — Delete BQ dataset, GCS artifacts

**Results:**

| Runner | Predictions | <=50K | >50K | Match |
|--------|------------|-------|------|-------|
| DirectRunner (local) | 6,456 | 5,232 | 1,224 | baseline |
| DataflowRunner (cloud) | 6,456 | 5,232 | 1,224 | ✅ identical |

**Key Learnings:**
- `beam.Map(lambda x: list.append(x))` does NOT reliably collect results with DirectRunner — use `beam.io.WriteToText` or `WriteToBigQuery` as proper sinks instead
- `SklearnModelHandlerNumpy` loads model once per worker via `setup()`, then reuses for all elements — no per-prediction overhead
- `ModelFileType.PICKLE` is the default; also supports `JOBLIB`
- DirectRunner → DataflowRunner is a one-line change (`runner='DataflowRunner'` + pipeline options) — same pipeline code runs anywhere
- Dataflow workers use the same Python version as the Beam SDK, so no pickle protocol mismatch (unlike Vertex AI serving containers which may run older Python)
- In-pipeline inference adds ~1-5ms latency vs 30-100ms+ for external endpoint calls — the exam answer for "minimal latency in a pipeline"
- `PredictionResult` has three fields: `.example` (input), `.inference` (prediction), `.model_id` (model path)
- Beam's `ReadFromText` reads from GCS transparently — same API for local files and cloud storage
- For CSV parsing in Beam, simple `line.split(',')` is more reliable than `csv.reader` when data is purely numeric with no embedded commas
- The exam pattern: "Bring the model TO the data, don't send data TO the model" — if data flows through Dataflow, load the model into workers

**New Skills:** Apache Beam programming model (PCollections, PTransforms), `RunInference` API, `SklearnModelHandlerNumpy`, `DirectRunner` vs `DataflowRunner`, Beam I/O connectors (ReadFromText, WriteToBigQuery), Dataflow job submission and monitoring, `PipelineOptions` configuration

---

## Phase 3: Theory Reinforcement

### Feature Engineering Deep Dive

**Topics:**
- Feature scaling and normalization, handling categorical variables, missing data strategies
- Feature selection: filter vs wrapper vs embedded methods, curse of dimensionality
- Domain-specific techniques: temporal features, geospatial features, text features

**Exercise:** Apply 3 new feature engineering techniques to the Lab 1 census model and measure impact.

### Algorithm Deep Dives

**Topics:**
- Logistic regression: mathematical foundations, MLE, regularization (L1/L2)
- Decision trees and random forests: information gain, Gini impurity, pruning, ensemble methods
- Gradient boosting: additive modeling, gradient descent in function space, XGBoost/LightGBM/CatBoost

**Exercise:** Implement logistic regression from scratch. Compare to scikit-learn. Explain why gradient boosting beats random forests on the census dataset.

### Model Evaluation & Selection

**Topics:**
- Evaluation metrics deep dive (precision, recall, F1, ROC AUC, when to use which)
- Cross-validation: k-fold, stratified, time series, nested
- Bias-variance tradeoff, learning curves

**Exercise:** Create a model selection framework for 5 business problems — fraud detection, churn prediction, recommendations, medical diagnosis, dynamic pricing. Document algorithm choice, evaluation metric, hyperparameters, and key features for each.

---

## Phase 4: Certification-Specific Prep

### GCP Services Deep Dive

**Service Comparison Matrix:**

| Use Case | BigQuery ML | AutoML | Vertex AI Custom | Pre-trained APIs | Agent Builder |
|----------|-------------|---------|------------------|------------------|---------------|
| Tabular data, SQL users | ✅ Best | ⚠️ Okay | ❌ Overkill | N/A | N/A |
| Complex deep learning | ❌ No | ⚠️ Limited | ✅ Best | N/A | N/A |
| No ML expertise | ⚠️ Need SQL | ✅ Best | ❌ Too hard | ✅ Best | ✅ Best |
| Custom architecture | ❌ No | ❌ No | ✅ Best | N/A | N/A |
| Time-to-market priority | ✅ Fast | ✅ Fast | ❌ Slow | ✅ Fastest | ✅ Fast |
| Q&A over internal docs | N/A | N/A | ❌ Overkill | N/A | ✅ Best |
| Time series forecasting | ✅ ARIMA_PLUS | ⚠️ Limited | ✅ LSTM/GBT | N/A | N/A |

**Additional Topics:**
- MLOps patterns: CI/CD for ML, A/B testing, shadow deployments, champion/challenger, Feature Store integration
- Specialized services: Vision AI, Natural Language API, Speech APIs, Recommendations AI, Document AI, Vector Search
- Cost optimization: preemptible VMs, batch vs online serving costs, reserved capacity
- Ethical AI and fairness: What-If Tool, fairness indicators, model cards, bias detection

---

## GCP Services Coverage Map

| Service | Lab(s) | Status |
|---------|--------|--------|
| BigQuery ML | 1, 9 | ✅ |
| Vertex AI Datasets | 2 | ✅ |
| AutoML Training | 2, 8 | ✅ |
| Custom Training | 2, 7, 8, 9 | ✅ |
| Model Deployment / Endpoints | 2, 8, 10 | ✅ |
| Hyperparameter Tuning | 3 | ✅ |
| Model Monitoring | 4 | ✅ |
| Feature Store | 5 | ✅ |
| Experiments | 5, 7, 8, 9 | ✅ |
| Metadata / Lineage | 5, 7, 8, 9, 10 | ✅ |
| Agent Builder / RAG | 6 | ✅ |
| Batch Prediction | 7 | ✅ |
| Text / NLP Pipeline | 7 | ✅ |
| Image / Vision Pipeline | 8 | ✅ |
| Transfer Learning | 8 | ✅ |
| Time Series / ARIMA_PLUS | 9 | ✅ |
| ML.FORECAST / ML.EXPLAIN_FORECAST | 9 | ✅ |
| LSTM / Sequence Models | 9 | ✅ |
| Vertex AI Pipelines | 10 | ✅ |
| Conditional Deployment | 10 | ✅ |
| Pipeline Scheduling | 10 | ✅ |
| CPR | Mini-Lab A | ✅ |
| Explainability (Shapley) | Mini-Lab B | ✅ |
| TFRecord Pipeline / tf.data | Mini-Lab C | ✅ |
| Shadow / Canary Deployment | Mini-Lab D | ✅ |
| Traffic Splitting | Mini-Lab D | ✅ |
| Prediction Request-Response Logging | Mini-Lab D | ✅ |
| Dataflow RunInference | Mini-Lab E | ✅ |
| Beam I/O (GCS, BigQuery) | Mini-Lab E | ✅ |

---

## Success Criteria

**Technical Skills:**
- Build production ML pipelines on GCP across tabular, text, image, and time series data
- Choose appropriate GCP ML services for different scenarios
- Implement proper MLOps practices (Feature Store, Experiments, Monitoring, Pipelines)
- Debug and optimize model performance
- Build RAG-based agents with Agent Builder

**Theoretical Knowledge:**
- Explain bias-variance tradeoff
- Understand when to use different algorithms
- Design feature engineering strategies
- Select appropriate evaluation metrics

**Certification Readiness:**
- Score 85%+ on practice exams
- Confidently answer scenario-based questions
- Understand GCP pricing and optimization
- Know ethical AI best practices
- Distinguish between Agent Builder, fine-tuning, and custom RAG scenarios

---

## Key Learnings So Far

- **Algorithm selection >> feature engineering:** Boosted trees gave 9.7x more improvement than manual feature engineering (Lab 1)
- **Custom can beat managed:** Custom training outperformed AutoML at 0.27% of the cost (Lab 2)
- **Container consistency matters:** Python/TF version mismatches between training and serving cause silent accuracy drops (Labs 2, 7, 8, 9)
- **Drift ≠ degradation:** Statistical drift doesn't always correlate with performance impact (Lab 4)
- **Simpler models can win:** TF-IDF + logistic regression outperformed neural embeddings for keyword-driven classification (Lab 7)
- **Prebuilt-first:** Google prebuilt containers reduce Dockerfile maintenance but require version awareness (Labs 7, 9)
- **Transfer learning is powerful:** MobileNetV2 fine-tuning hit 97.83% on satellite imagery with minimal data (Lab 8)
- **Base64 serving solves payload limits:** Baking decode+preprocess into SavedModel avoids gRPC/REST size limits (Lab 8)
- **ARIMA_PLUS for SQL users:** Zero-code time series forecasting with decomposition — exam favorite (Lab 9)
- **ML.FORECAST ≠ ML.PREDICT:** ARIMA models use ML.FORECAST; supervised BQML models use ML.PREDICT (Lab 9)
- **GBT beats LSTM with good features:** Tabular regression with lag features outperformed sequence model for one-step-ahead prediction (Lab 9)
- **TF on Apple Silicon is slow:** Submit TF training to Vertex AI rather than running locally on Mac (Lab 9)
- **Python 3.13 compatibility:** google-auth REST transport breaks; use gRPC client directly (Lab 9)
- **Lightweight KFP components work for small models:** No separate training VM needed for sklearn on tabular data (Lab 10)
- **`dsl.Condition` is the exam favorite:** "Deploy only if X" → conditional branching in pipelines (Lab 10)
- **Prediction images ≠ component images:** Google's sklearn prediction containers have old Python/pip; use `python:3.10` + pinned deps for KFP components (Lab 10)
- **Pipeline parameters enable reusability:** Same pipeline, different thresholds/hyperparameters per run (Lab 10)
- **TFRecord speed is about read/parse, not size:** Binary serialization eliminates text parsing overhead; files may be same size or larger than CSV (Mini-Lab C)
- **CPR eliminates HTTP plumbing, not ML logic:** Same preprocessing code, but no Flask server, routing, health checks, or hand-written Dockerfile — Vertex AI handles all of it (Mini-Lab A)
- **CPR is the exam's serving sweet spot:** "Custom model + preprocessing + minimize code" → CPR; sits between AutoML and custom container on the abstraction spectrum (Mini-Lab A)
- **`build_cpr_model()` auto-generates everything:** Dockerfile, image build, and model server from just a Predictor class and requirements.txt (Mini-Lab A)
- **SHAP KernelExplainer ≈ Vertex AI Sampled Shapley:** Both approximate Shapley values by sampling feature subsets; KernelExplainer gives richer local visualizations (Mini-Lab B)
- **ExplanationSpec is set at model upload, not deploy:** `ExplanationMetadata` + `ExplanationParameters` attach to the Model resource; optional for TF2 SavedModels (auto-detected) (Mini-Lab B)
- **Sampled Shapley = universal, IG = neural nets, XRAI = images:** Three explainability methods with clear decision boundaries; exam pattern is "any model" → Shapley, "TF model" → IG, "image model" → XRAI (Mini-Lab B)
- **TFRecord schema is not self-describing:** Feature description must be maintained in code; no header row like CSV (Mini-Lab C)
- **Sharding helps at scale, not locally:** On local NVMe, interleave overhead slows reads vs single file; payoff comes with distributed training and GCS (Mini-Lab C)
- **Traffic splitting is the Vertex AI deployment strategy mechanism:** One endpoint, multiple deployed models, percentage-based routing — covers both canary and A/B (Mini-Lab D)
- **Canary vs A/B vs shadow — same infra, different intent:** Canary = safe rollout, A/B = statistical comparison, shadow = risk-free validation; shadow requires custom proxy (Mini-Lab D)
- **Prediction logging requires explicit BQ destination:** `enable_request_response_logging=True` alone fails; must provide `request_response_logging_bq_destination_table` (Mini-Lab D)
- **Undeploy order matters:** Must undeploy 0% traffic models first; Vertex AI rejects undeploying a model if remaining traffic would sum to 0% (Mini-Lab D)
- **RunInference loads model once per worker:** `SklearnModelHandlerNumpy` calls `setup()` once, then reuses — no per-prediction overhead; in-pipeline inference adds ~1-5ms vs 30-100ms+ for external endpoints (Mini-Lab E)
- **DirectRunner → DataflowRunner is a one-line change:** Same Beam pipeline code runs locally or on cloud — the "write once, run anywhere" value proposition (Mini-Lab E)
- **Don't collect Beam results with list.append:** `beam.Map(lambda x: list.append(x))` is unreliable with DirectRunner; use proper sinks like `WriteToText` or `WriteToBigQuery` (Mini-Lab E)
- **Bring the model TO the data:** If data flows through Dataflow, load the model into workers with RunInference; if data is in BigQuery, use BigQuery ML; shared endpoint only when multiple consumers need it (Mini-Lab E)

---

## Learning Resources

**Primary:** [GCP ML Documentation](https://cloud.google.com/ai-platform/docs) · [ML Crash Course](https://developers.google.com/machine-learning/crash-course) · [Machine Learning Mastery](https://machinelearningmastery.com/)

**Supplementary:** [Vertex AI Code Samples](https://github.com/GoogleCloudPlatform/vertex-ai-samples) · [Kaggle Learn](https://www.kaggle.com/learn) · [StatQuest YouTube](https://www.youtube.com/c/joshstarmer) · [Coursera MLOps Specialization](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops)

---

## Post-Certification Path

- Build and deploy 2-3 portfolio ML systems on GitHub
- Consider: Google Professional Data Engineer, TensorFlow Developer Certificate, AWS ML Specialty
- Specialize: Computer Vision, NLP/LLMs, Recommendation Systems, or MLOps Engineering