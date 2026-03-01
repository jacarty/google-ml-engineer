# GCP ML Engineer Certification — Complete Plan

**Goal:** Google Cloud Professional Machine Learning Engineer Certification  
**Approach:** Hands-on labs first, then theory reinforcement  

---

## Progress Summary

| Lab | Status | Key Outcome |
|-----|--------|-------------|
| Lab 1: BigQuery ML (Feature Engineering) | ✅ Complete | Boosted trees: 86.23% accuracy |
| Lab 2: Vertex AI Pipeline (End-to-End) | ✅ Complete | Custom beat AutoML at 0.27% of the cost |
| Lab 3: Hyperparameter Tuning | ✅ Complete | Manual vs Bayesian optimization |
| Lab 4: Monitoring & Drift Detection | ✅ Complete | Drift detection + response runbook |
| Lab 5: MLOps Services | ✅ Complete | Feature Store, Experiments, Metadata |
| Lab 6: Agent Builder (RAG) | ✅ Complete | Working RAG agent with citations |
| Lab 7: Text Classification | ✅ Complete | TF-IDF baseline + batch prediction |
| Lab 8: Image Classification | 📋 Planned | — |
| Lab 9: Time Series Forecasting | 📋 Planned | — |
| Lab 10: Vertex AI Pipelines | 📋 Planned | — |
| Mini-Labs: A-E | 📋 Planned | — |

## Suggested Lab Order (Remaining)

1. **Mini-Lab A (CPR)** — Quick win, directly addresses exam gap on abstraction levels
2. **Lab 9 (Time Series + Feature Store)** — BQML vs custom, ML.PREDICT, Feature Store second touchpoint
3. **Lab 8 (Image)** — New modality, transfer learning, Vision API
4. **Lab 10 (Pipelines)** — Ties everything together with orchestration
5. **Mini-Labs B–E** — Fill remaining gaps as time allows

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

### Lab 8 — Image Classification with Satellite Data 📋

**Dataset:** EuroSAT via TensorFlow Datasets (10 land use classes, 27k satellite images)  
**Task:** Multi-class satellite image classification  
**Container:** Discuss prebuilt vs custom (likely prebuilt TF + prediction container)  
**Exam Relevance:** Vision API vs AutoML vs custom, transfer learning, tf.data pipelines

**Parts:**
- Setup — GCS bucket, container strategy discussion, enable Vision API
- Data sourcing and preparation — load via TFDS, upload to GCS, BQ metadata table
- Pre-trained API baseline — evaluate Vision API on satellite classes, document accuracy/cost/latency
- Transfer learning with custom training — MobileNetV2 frozen → fine-tuned, 3+ Experiment runs
- Online prediction with endpoint — deploy best model, compare to Vision API
- Metadata, lineage, and architecture doc — decision matrix: Vision API vs AutoML vs custom

**New Skills:** Image pipelines, Vision API evaluation, transfer learning, tf.data, pre-trained vs custom decision making

---

### Lab 9 — Time Series Forecasting with NOAA Weather Data 📋

**Dataset:** `bigquery-public-data.noaa_gsod` (Global Surface Summary of the Day)  
**Task:** Forecast daily temperature for JFK Airport  
**Container:** Discuss per model type (prebuilt TF for LSTM, prebuilt/custom sklearn for GBT, none for BQML)  
**Exam Relevance:** ARIMA_PLUS, temporal features, BQML vs custom, Feature Store, ML.PREDICT

**Parts:**
- Setup — GCS bucket, container strategy for each model type
- Data exploration in BigQuery — temporal feature engineering (lag, rolling averages, cyclical encoding), time-based splits
- BigQuery ML forecasting — ARIMA_PLUS model, ML.FORECAST, ML.EXPLAIN_FORECAST
- Custom training comparison — Gradient boosting vs LSTM, all three models in Experiments
- Batch prediction with BQML — ML.PREDICT in BigQuery, compare to Vertex AI batch
- Feature Store for temporal features — register features, Online Store, train from Feature Store vs CSV
- Metadata and lineage — trace NOAA → BQ features → Feature Store → training → predictions

**New Skills:** Time series features, BQML ARIMA_PLUS, LSTM, time-aware splitting, ML.PREDICT, Feature Store with temporal data

---

### Lab 10 — Vertex AI Pipelines: Orchestrating End-to-End Workflow 📋

**Dataset:** `bigquery-public-data.stackoverflow.posts_questions` (same as Lab 7, queried fresh)  
**Task:** Build a Kubeflow pipeline: data prep → training → evaluation → conditional deployment  
**Container:** Discuss before starting (likely prebuilt TF, reusing Lab 7 pattern)  
**Exam Relevance:** Very high — Vertex AI Pipelines, conditional deployment, pipeline components

**Parts:**
- Setup — GCS bucket, container strategy, pip install kfp
- Pipeline components — data_prep, train_model, evaluate_model, conditional_deploy as KFP components
- Pipeline definition and conditional logic — wire components, dsl.Condition for deploy-if-good, submit to Vertex AI
- Pipeline execution and analysis — explore pipeline graph, verify both branches, schedule recurring runs

**New Skills:** Kubeflow Pipelines, component design, conditional deployment, pipeline scheduling, automatic lineage

---

## Mini-Labs

Target specific gaps identified from practice questions. Each is 1-3 hours.

| Mini-Lab | Targets | Summary |
|----------|---------|---------|
| **A: Custom Prediction Routine (CPR)** | Abstraction level selection | Rebuild Lab 2 serving with CPR Predictor class instead of Flask |
| **B: Explainability (Sampled Shapley)** | Explainability methods | Add explanations to census model, compare Shapley vs Integrated Gradients vs XRAI |
| **C: TFRecord Pipeline** | Data formats | Convert census CSV → TFRecord, benchmark read speed, add sharding |
| **D: Shadow Deployment** | Deployment patterns | Traffic splitting with champion/challenger, document shadow vs A/B vs canary |
| **E: Dataflow + RunInference** | Streaming serving | Build Beam pipeline with model in Dataflow workers, add Pub/Sub trigger |

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

**Additional Topics:**
- MLOps patterns: CI/CD for ML, A/B testing, shadow deployments, champion/challenger, Feature Store integration
- Specialized services: Vision AI, Natural Language API, Speech APIs, Recommendations AI, Document AI, Vector Search
- Cost optimization: preemptible VMs, batch vs online serving costs, reserved capacity
- Ethical AI and fairness: What-If Tool, fairness indicators, model cards, bias detection

---

## GCP Services Coverage Map

| Service | Lab(s) | Status |
|---------|--------|--------|
| BigQuery ML | 1, 9 | ✅ / 📋 |
| Vertex AI Datasets | 2 | ✅ |
| AutoML Training | 2 | ✅ |
| Custom Training | 2, 7, 8, 9 | ✅ / 📋 |
| Model Deployment / Endpoints | 2, 8 | ✅ / 📋 |
| Hyperparameter Tuning | 3 | ✅ |
| Model Monitoring | 4 | ✅ |
| Feature Store | 5, 9 | ✅ / 📋 |
| Experiments | 5, 7, 8, 9 | ✅ / 📋 |
| Metadata / Lineage | 5, 7, 8, 9, 10 | ✅ / 📋 |
| Agent Builder / RAG | 6 | ✅ |
| Batch Prediction | 7, 9 | ✅ / 📋 |
| Text / NLP Pipeline | 7 | ✅ |
| Image / Vision Pipeline | 8 | 📋 |
| Vision API (pre-trained) | 8 | 📋 |
| Transfer Learning | 8 | 📋 |
| Time Series / ARIMA_PLUS | 9 | 📋 |
| LSTM / Sequence Models | 9 | 📋 |
| Vertex AI Pipelines | 10 | 📋 |
| Conditional Deployment | 10 | 📋 |
| CPR | Mini-Lab A | 📋 |
| Explainability (Shapley) | Mini-Lab B | 📋 |
| TFRecord Pipeline | Mini-Lab C | 📋 |
| Shadow / Canary Deployment | Mini-Lab D | 📋 |
| Dataflow RunInference | Mini-Lab E | 📋 |
| Pub/Sub + Streaming Inference | Mini-Lab E | 📋 |

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
- **Container consistency matters:** Python/TF version mismatches between training and serving cause silent accuracy drops (Labs 2, 7)
- **Drift ≠ degradation:** Statistical drift doesn't always correlate with performance impact (Lab 4)
- **Simpler models can win:** TF-IDF + logistic regression outperformed neural embeddings for keyword-driven classification (Lab 7)
- **Prebuilt-first:** Google prebuilt containers reduce Dockerfile maintenance but require version awareness (Lab 7)

---

## Learning Resources

**Primary:** [GCP ML Documentation](https://cloud.google.com/ai-platform/docs) · [ML Crash Course](https://developers.google.com/machine-learning/crash-course) · [Machine Learning Mastery](https://machinelearningmastery.com/)

**Supplementary:** [Vertex AI Code Samples](https://github.com/GoogleCloudPlatform/vertex-ai-samples) · [Kaggle Learn](https://www.kaggle.com/learn) · [StatQuest YouTube](https://www.youtube.com/c/joshstarmer) · [Coursera MLOps Specialization](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops)

---

## Post-Certification Path

- Build and deploy 2-3 portfolio ML systems on GitHub
- Consider: Google Professional Data Engineer, TensorFlow Developer Certificate, AWS ML Specialty
- Specialize: Computer Vision, NLP/LLMs, Recommendation Systems, or MLOps Engineering