# GCP ML Engineer Certification — Study Notes

## Part 1: Key Concepts & Decision Frameworks

---

### 1. Serving Architecture Selection

Choosing **where to run inference** is a critical decision — loading models into Dataflow vs. deploying to endpoints vs. using BigQuery ML vs. Cloud Run.

**Decision framework:**

| Scenario | Best Choice | Why |
|----------|------------|-----|
| Data already in BigQuery, batch predictions | BigQuery ML (`ML.PREDICT`) | Zero data movement |
| Streaming pipeline (Dataflow/Beam), low latency | Load model directly into Dataflow workers | No network calls, ~1-5ms |
| Multiple consumers need the same model | Vertex AI Endpoint | Shared serving infrastructure |
| Need full control, strict SLAs | GKE with TFServing | Fine-grained tuning |
| Serverless, bursty traffic | Cloud Run | Scales to zero |

**Key insight:** If a Dataflow pipeline needs predictions, load the model into the pipeline — avoid calling an external endpoint.

---

### 2. Abstraction Level Selection

Picking the **right level of abstraction** is essential — use managed options when possible, custom solutions when needed.

**The abstraction ladder:**

```
Most managed (least code)     →     Most control (most code)
─────────────────────────────────────────────────────────
Pre-trained APIs  →  AutoML  →  CPR  →  Custom Container  →  Self-managed (GKE)
```

**Common scenarios and recommended approaches:**

| Scenario | Recommended Approach |
|---|---|
| "minimize code" | CPR or AutoML |
| "minimize infrastructure" | Vertex AI managed services |
| "full control" | Custom container or GKE |
| "no ML expertise" | Pre-trained APIs or AutoML |
| "minimize computational overhead" | BigQuery ML (in-place) |
| "minimize latency" | In-process inference or Memorystore |

---

### 3. Loss Functions & Metrics Selection

Choosing the right loss function and evaluation metric depends on the problem type and class distribution.

**Quick reference:**

| Problem Type | Loss Function | Primary Metric |
|---|---|---|
| Binary classification | Binary cross-entropy | F1, AUC-ROC, AUC-PR |
| Multi-class (one label) | Categorical cross-entropy | Accuracy, macro F1 |
| Multi-class (integer labels) | Sparse categorical cross-entropy | Same as above |
| Multi-label (multiple labels) | Binary cross-entropy (per label) | Hamming loss, micro F1 |
| Regression | MSE or MAE | RMSE, MAE, R² |
| Ranking | Pairwise/listwise loss | NDCG, MRR |
| Imbalanced classification | Weighted cross-entropy or focal loss | **F1, AUC-PR** (never accuracy) |

**Imbalanced data rule:** If classes are imbalanced (>80/20 split), accuracy is misleading. Use F1 or AUC-PR instead.

---

### 4. Data Format & Pipeline Optimization

Choosing the right data format significantly impacts training performance.

**When to use each format:**

| Format | Best For | Key Advantage |
|---|---|---|
| CSV | Quick prototyping, small data | Human-readable |
| Parquet | BigQuery export, columnar analytics | Schema-preserving, compressed |
| TFRecord | TensorFlow training pipelines | Native binary, fastest I/O |
| Avro | Streaming data (Kafka/Pub/Sub) | Schema evolution |
| JSONL | API ingestion, logging | Flexible schema |

**Pipeline optimization order:**
1. Convert CSV → TFRecord (biggest single improvement)
2. Shard into multiple files + parallel interleave
3. Prefetch and cache
4. Use `tf.data` service for distributed preprocessing

---

### 5. Distributed Training Strategies

Choosing the right distribution strategy depends on hardware and model size.

| Strategy | When to Use | Machines |
|---|---|---|
| `MirroredStrategy` | Multi-GPU, single machine | 1 |
| `MultiWorkerMirroredStrategy` | Multi-GPU, one or more machines | 1+ |
| `ParameterServerStrategy` | Very large models, async training | Multiple (with PS) |
| `TPUStrategy` | TPU pods | TPU only |

**GPU vs. TPU decision:**
- Default to GPU for: custom ops, debugging-heavy research, small-medium models
- Use TPU for: very large models, standard TF ops only, production training at scale
- TPU gotchas: limited custom op support, harder debugging, XLA compilation quirks

---

### 6. Explainability Methods

Different explainability methods work best for different model types:

| Method | Works With | Use Case |
|---|---|---|
| Sampled Shapley | Any model type | Tabular data, general purpose |
| Integrated Gradients | Neural networks only | DNNs, requires differentiable model |
| XRAI | Image models | Region-based image explanations |

**Selection guide:** For any model type or custom models → Sampled Shapley. For neural networks → Integrated Gradients is also valid.

---

### 7. MLOps & Production Deployment Patterns

Common deployment patterns for production ML systems:

| Pattern | What It Is | When to Use |
|---|---|---|
| Shadow deployment | New model gets same traffic, predictions logged but not served | High-risk models, need confidence before switching |
| A/B testing | Split traffic between old and new model | Measuring business impact |
| Champion/challenger | Current model (champion) vs. candidate (challenger) | Continuous improvement cycle |
| Canary deployment | Gradually increase traffic to new model (10% → 50% → 100%) | Safe rollout with rollback option |
| Blue/green | Instant switch between two full deployments | Zero-downtime deployment |

---

## Part 2: Model Selection Cheat Sheet

### Algorithm Selection Decision Tree

```
Is your data tabular?
├── YES → How much data?
│   ├── < 1000 rows → Logistic Regression or SVM
│   ├── 1K - 100K rows → Gradient Boosted Trees (XGBoost/LightGBM)
│   └── > 100K rows → Gradient Boosted Trees or Neural Network
│
├── Is it images?
│   ├── Classification → CNN (ResNet, EfficientNet)
│   ├── Object detection → YOLO, Faster R-CNN, or AutoML Vision
│   └── Segmentation → U-Net, Mask R-CNN
│
├── Is it text?
│   ├── Classification → BERT, or simpler: TF-IDF + Logistic Regression
│   ├── Generation → GPT/T5/PaLM
│   ├── NER → BERT + token classification
│   └── Embeddings → Sentence-BERT, Word2Vec
│
├── Is it time series?
│   ├── Forecasting → LSTM/GRU, Prophet, or ARIMA
│   ├── Anomaly detection → Autoencoders, Isolation Forest
│   └── Classification → 1D-CNN or LSTM
│
├── Is it a recommendation problem?
│   ├── Collaborative filtering → Matrix factorization, ALS
│   ├── Content-based → Feature similarity
│   └── Hybrid → Two-tower neural network
│
└── Is it a decision/action problem?
    └── Reinforcement Learning (Q-learning, policy gradient)
```

### Classical ML Models

| Model | Type | Strengths | Weaknesses | Hyperparameters |
|---|---|---|---|---|
| **Logistic Regression** | Classification | Interpretable, fast, good baseline | Linear boundaries only | C (regularization), penalty (L1/L2) |
| **Linear Regression** | Regression | Simple, interpretable | Assumes linearity | alpha (regularization) |
| **Decision Tree** | Both | Interpretable, handles mixed types | Overfits easily | max_depth, min_samples_split |
| **Random Forest** | Both | Robust, less overfitting than trees | Slower, less interpretable | n_estimators, max_depth, max_features |
| **Gradient Boosted Trees** | Both | Best for tabular data, wins competitions | Can overfit, slower to train | n_estimators, learning_rate, max_depth |
| **SVM** | Classification | Good for high-dimensional, small data | Slow on large data, kernel choice | C, kernel, gamma |
| **K-Nearest Neighbors** | Both | Simple, no training | Slow prediction, curse of dimensionality | k, distance metric |
| **Naive Bayes** | Classification | Fast, good for text | Assumes feature independence | smoothing (alpha) |
| **K-Means** | Clustering | Simple, scalable | Must specify k, assumes spherical clusters | n_clusters, init method |

### Deep Learning Models

| Model | Use Case | Key Feature | GCP Service |
|---|---|---|---|
| **DNN (Dense)** | Tabular, structured data | Fully connected layers | Vertex AI, BigQuery ML |
| **CNN** | Images, spatial data | Convolutional filters detect local patterns | Vertex AI, AutoML Vision |
| **RNN/LSTM/GRU** | Sequences, time series, text | Memory of previous inputs | Vertex AI Custom Training |
| **Transformer** | Text, translation, generation | Self-attention, parallel processing | Vertex AI with pre-trained models |
| **Autoencoder** | Anomaly detection, compression | Learns compressed representation | Custom training |
| **GAN** | Image generation, data augmentation | Generator vs discriminator | Custom training |

### When to Use What on GCP

| Scenario | Service | Model Type |
|---|---|---|
| SQL users, data in BigQuery | BigQuery ML | Logistic Reg, Boosted Trees, DNN, ARIMA |
| No ML expertise, any data type | AutoML | Automatic selection |
| Need specific algorithm/architecture | Vertex AI Custom Training | Anything |
| Standard vision/NLP/speech task | Pre-trained APIs | Google's pre-trained models |
| Need embeddings or LLM | Vertex AI Model Garden | Foundation models |

### Regularization Quick Reference

| Technique | What It Does | When to Use |
|---|---|---|
| **L1 (Lasso)** | Drives weights to zero (feature selection) | Too many features, want sparsity |
| **L2 (Ridge)** | Shrinks weights toward zero (doesn't eliminate) | Multicollinearity, want stability |
| **Elastic Net** | Combination of L1 + L2 | When unsure, or correlated features |
| **Dropout** | Randomly drops neurons during training | Neural networks overfitting |
| **Early stopping** | Stop training when validation loss increases | Any iterative model |
| **Data augmentation** | Create transformed copies of training data | Small datasets, especially images |
| **Batch normalization** | Normalizes layer inputs | Deep networks, faster training |

### Evaluation Metrics Decision Guide

```
What are you optimizing for?

Binary Classification:
├── Balanced classes → Accuracy or AUC-ROC
├── Imbalanced classes → F1, AUC-PR
├── Cost of false positives high (spam filter) → Precision
├── Cost of false negatives high (cancer detection) → Recall
└── Need threshold-independent metric → AUC-ROC

Multi-class:
├── All classes equally important → Macro F1
├── More common classes matter more → Weighted F1
└── Overall correctness → Accuracy (only if balanced)

Regression:
├── Want to penalize large errors more → RMSE
├── Want robust to outliers → MAE
├── Want relative performance → R²
└── Want percentage error → MAPE

Ranking:
├── Order matters → NDCG
└── First result matters most → MRR
```

### Feature Engineering Patterns

| Pattern | When to Use | Example |
|---|---|---|
| **Bucketizing** | Non-linear numeric relationships | age → [young, mid, senior] |
| **Feature crossing** | Interaction effects | education × occupation |
| **Log transform** | Skewed distributions | log(income), log(page_views) |
| **One-hot encoding** | Low-cardinality categorical | workclass → [private, govt, self-emp] |
| **Embedding** | High-cardinality categorical | zip_code → learned vector |
| **Target encoding** | High-cardinality + tabular | Replace category with mean target |
| **Time features** | Temporal data | hour_of_day, day_of_week, is_weekend |
| **Text: TF-IDF** | Simple text features | Term frequency weighting |
| **Text: Embeddings** | Semantic text features | Word2Vec, BERT embeddings |

### Key Formulas

```
Precision = TP / (TP + FP)        "Of predicted positive, how many are correct?"
Recall    = TP / (TP + FN)        "Of actual positive, how many did we find?"
F1        = 2 × (P × R) / (P + R) Harmonic mean of precision and recall
Accuracy  = (TP + TN) / Total     "Overall correctness" (misleading if imbalanced)

AUC-ROC   = Area under TPR vs FPR curve (threshold-independent)
AUC-PR    = Area under Precision vs Recall curve (better for imbalanced)

RMSE = sqrt(mean((y_pred - y_actual)²))
MAE  = mean(|y_pred - y_actual|)
R²   = 1 - (SS_res / SS_tot)
```

---

## Part 3: Supplementary Lab Ideas

Additional hands-on exercises to reinforce key concepts:

### Lab A: Serving Architecture Comparison (2-3 hours)
**Focus:** Serving selection

Deploy a model three different ways and measure latency/cost:
1. **BigQuery ML** — Export model to BigQuery ML, run `ML.PREDICT`
2. **Vertex AI Endpoint** — Standard managed endpoint
3. **Embedded in Dataflow** — Load model into a Beam pipeline, make predictions in-process

Compare: latency, throughput, cost per 1000 predictions, operational complexity.

### Lab B: Custom Prediction Routine (CPR) (1-2 hours)
**Focus:** Abstraction level

Build serving without Flask using CPR:
1. Write a `Predictor` class with `preprocess()`, `predict()`, `postprocess()`
2. Build the CPR container using Vertex AI's tooling
3. Deploy and compare to a custom Flask container

### Lab C: TFRecord Pipeline (1-2 hours)
**Focus:** Data formats

Convert CSV to TFRecord format and build a `tf.data` pipeline:
1. Write a TFRecord conversion script
2. Build a `tf.data.TFRecordDataset` pipeline with parsing, batching, prefetching
3. Benchmark CSV vs TFRecord read speed
4. Add parallel interleave with sharded files

### Lab D: Vertex AI Pipelines (2-3 hours)
**Focus:** Pipeline orchestration

Build a simple 3-step Kubeflow pipeline on Vertex AI:
1. Data loading component (reads from GCS)
2. Training component (trains a classifier)
3. Evaluation component (computes metrics, decides if model is good enough)

### Lab E: Explainability & Fairness (1-2 hours)
**Focus:** Model interpretability

Add explainability to a deployed model:
1. Configure Sampled Shapley on a Vertex AI model
2. Request explanations alongside predictions
3. Use the What-If Tool to explore fairness across demographic groups
4. Document which features drive predictions for different segments
