# GCP ML Engineer — Reinforcement Projects

**Purpose:** Apply skills from Labs 1–10 and Mini-Labs A–E to new domains and datasets, reinforcing core competencies while adding one new exam-relevant skill per project.

**Approach:** Each project follows the full GCP pipeline (BigQuery → GCS → Vertex AI → Endpoint) and is designed to be executed in a single chat session.

**Sequencing:** Projects are ordered by priority. Each project is self-contained.

---

## Progress Summary

| Project | Domain | Status | Key Outcome |
|---------|--------|--------|-------------|
| Project 1: Support Ticket Routing | Text / NLP | ✅ Complete | KFP pipeline with conditional deploy, CPR endpoint, Weighted F1 0.5156 |
| Project 2: Serengeti Wildlife Species ID | Image / Vision | ✅ Complete | MobileNetV2 transfer learning, 72.6% test accuracy (10 species), occlusion sensitivity explainability |
| Project 3: Bitcoin Volatility Forecasting | Time Series / Finance | ⬜ Not Started | — |

---

## Project 1 — Support Ticket Routing (Text Classification)

### Overview

| Attribute | Detail |
|-----------|--------|
| **Dataset** | Multilingual Customer Support Tickets (Kaggle/HuggingFace, ~20k English tickets) |
| **Task** | Multi-class classification — route ticket to correct department (7-8 classes: Technical Support, Customer Service, Billing, Product Support, IT Support, Returns, Sales) |
| **Pipeline** | CSV → BigQuery → GCS → TF-IDF baseline + neural → Vertex AI custom training → CPR endpoint |
| **Estimated Cost** | ~$8-12 |
| **Estimated Time** | 6-8 hours |

### Skills Reinforced

| Skill | Original Lab | How It's Applied Here |
|-------|-------------|----------------------|
| Text preprocessing & TF-IDF baseline | Lab 7 | Email body text → cleaned features; TF-IDF + LogReg as baseline |
| Neural text classification | Lab 7 | Embedding + pooling architecture on ticket text |
| Custom training on Vertex AI | Labs 2, 7, 8 | Prebuilt TF container, Experiments logging |
| CPR serving | Mini-Lab A | `SklearnPredictor` or custom `Predictor` with confidence routing |
| TFRecord pipeline | Mini-Lab C | Convert preprocessed text to TFRecords for efficient training |
| Experiment tracking | Lab 5 | Compare TF-IDF vs neural across metrics |

### New Skill: KFP Pipeline on Real Data

Lab 10 built a pipeline on the 344-row penguins toy dataset. This project wires the full workflow as a Kubeflow pipeline on a real-world text classification problem:

- **Data prep component** — query BigQuery, clean text, split, export to GCS
- **TF-IDF training component** — train baseline, log metrics
- **Neural training component** — train embedding model, log metrics
- **Evaluation component** — compare models, select champion
- **Conditional deployment** — `dsl.Condition` deploys only if champion beats threshold
- **CPR serving** — deploy with confidence-based routing (high confidence → auto-route, low confidence → flag for human review)

This also covers the exam pattern: "route low-confidence predictions to human review" — a common scenario question.

### Parts

| Part | Focus | Estimated Time |
|------|-------|---------------|
| 1 | Setup + Data Exploration | ~30 min |
| 2 | Data Preparation (BigQuery → GCS → TFRecord) | ~45 min |
| 3 | TF-IDF Baseline Training | ~30 min |
| 4 | Neural Model Training | ~45 min |
| 5 | Model Comparison + Experiment Analysis | ~30 min |
| 6 | KFP Pipeline (orchestrate Parts 2-5) | ~1.5 hr |
| 7 | CPR Endpoint + Confidence Routing | ~1 hr |
| 8 | Cleanup | ~15 min |

### Success Criteria

- TF-IDF baseline accuracy established
- Neural model trained and compared via Experiments
- Full KFP pipeline runs end-to-end with conditional deployment
- CPR endpoint returns department prediction + confidence + human-review flag
- All resources cleaned up

### Dataset Notes

- Source: https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets (also on Kaggle)
- Fields: subject, body, queue (department), priority, type, language
- Filter to English tickets only
- Target variable: `queue` (department routing)
- Text input: concatenation of `subject` + `body`
- Synthetic dataset — good for learning, not production-representative

---

## Project 2 — Serengeti Wildlife Species ID (Image Classification) ✅

### Overview

| Attribute | Detail |
|-----------|--------|
| **Dataset** | Snapshot Serengeti via LILA BC (29,880 images, top 10 species, one image per sequence) |
| **Task** | 10-class wildlife species classification from camera trap images |
| **Pipeline** | COCO JSON metadata → GCS image subset → BigQuery metadata → sharded TFRecords (cloud) → MobileNetV2 transfer learning → base64 serving endpoint → occlusion sensitivity explainability |
| **Actual Cost** | ~$12-15 |
| **Actual Time** | ~10 hours (including ~6 hours of CPU training due to GPU quota limits) |

### Results

| Metric | Value |
|--------|-------|
| Test Accuracy (overall) | 72.6% |
| Best Species | giraffe (80.5%), impala (80.3%), warthog (80.6%) |
| Worst Species | Grant's gazelle (49.3%), hartebeest (63.4%), wildebeest (67.9%) |
| Main Confusion Pairs | Grant's ↔ Thomson's gazelle, wildebeest ↔ buffalo |
| Training (v1, wrong preprocessing) | 68.4% val accuracy |
| Training (v2, corrected preprocessing) | 76.8% val accuracy |

### Skills Reinforced

| Skill | Original Lab | How It's Applied Here |
|-------|-------------|----------------------|
| Transfer learning (MobileNetV2) | Lab 8 | Freeze → fine-tune on wildlife species |
| TFRecord sharded pipeline | Mini-Lab C | Cloud-based TFRecord creation via CustomJob |
| tf.data pipeline | Lab 8, Mini-Lab C | Image augmentation (flip, brightness, contrast), prefetch |
| Base64 serving pattern | Lab 8 | Bake decode+preprocess into SavedModel graph via export script |
| Custom training on Vertex AI | Labs 2, 7, 8 | Prebuilt TF 2.15 CPU container (GPU quota unavailable) |

### New Skill: Image Explainability (XRAI → Occlusion Sensitivity Fallback)

Attempted XRAI via Vertex AI ExplanationSpec — failed with metadata mismatch (`Explain metadata output not in output_signature of the model function`). ExplanationSpec requires exact match between metadata input/output names and the serving signature's tensor names. Fell back to occlusion sensitivity (perturbation-based, model-agnostic), which produced region-level attribution heatmaps showing the model focuses on animal bodies.

Exam knowledge confirmed: XRAI = image specialist, configured at model upload time via `ExplanationSpec` with `XraiAttribution(step_count=N)`. Sampled Shapley = any model. Integrated Gradients = differentiable models.

### Parts (As Executed)

| Part | Focus | Actual Time |
|------|-------|-------------|
| 1 | Setup + Metadata Download (LILA BC JSON) + Species EDA | ~45 min |
| 2 | Subset Selection (3k/class, one per sequence) + GCS Copy | ~30 min (copy ~20 min with 32 threads) |
| 3 | BigQuery Metadata Table + Location-Based Splits | ~30 min |
| 4 | TFRecord Pipeline (sharded, cloud-based CustomJob) | ~30 min job time |
| 5 | Transfer Learning — v1 (wrong preprocessing) then v2 (corrected) | ~3 hrs each on CPU |
| 6 | Evaluation (per-class confusion matrix via CustomJob) | ~10 min |
| 7 | Model Export (base64 serving wrapper) + Endpoint Deployment | ~30 min |
| 7b | Endpoint Testing (zebra, elephant, giraffe — all correct, >99% confidence) | ~5 min |
| 8 | XRAI attempt (failed) → Occlusion Sensitivity (CustomJob) | ~15 min |
| 9 | Cleanup | ~10 min |

### Key Learnings

- **MobileNetV2 `preprocess_input` expects [-1, 1], not [0, 1]** — wrong normalisation cost ~10% accuracy. The ImageNet weights are calibrated for a specific input range; feeding the wrong range means every activation in the base model is shifted. Always use the model-specific `preprocess_input` function.
- **Location-based splitting prevents data leakage from shared camera backgrounds** — if the same camera appears in train and val, the model can memorise background features (specific trees, camera angle) instead of learning animal features. Hash-based split on camera location ensures the model has never seen a camera's background during validation.
- **Camera trap images are genuinely hard** — night IR images, motion blur, visually similar species (Grant's vs Thomson's gazelle). 72.6% on 10 species with these challenges is reasonable for MobileNetV2 with 3k images/class.
- **Server-side GCS copy (`blob.rewrite`) is much faster than download+upload** — data never touches your machine when copying between GCS buckets.
- **Run everything on Vertex AI when TF is involved** — TF 2.15 SavedModels + Python 3.13 + Apple Silicon = unreliable locally. TFRecord reading, model loading, and inference all hang or crash. Use CustomJobs for anything TF-related.
- **TF 2.15 SavedModels require `tf.saved_model.load` or `TFSMLayer` under Keras 3** — `keras.models.load_model()` rejects Keras 2 SavedModel format.
- **ExplanationSpec metadata must match exact serving signature input/output names** — the error message is clear but debugging requires knowing your signature's tensor names precisely.
- **Occlusion sensitivity is a viable model-agnostic alternative to GradCAM/XRAI** — systematically block patches, measure confidence drop. Slower but works with any model, any framework, any saved format.
- **CPU training on Vertex AI works but is ~10x slower than GPU** — ~3 hrs vs ~20 min for 20 epochs on 21k images. Request GPU quota increase before your next project.
- **EarlyStopping with `restore_best_weights=True` prevents overfitting degradation** — v2 peaked at epoch 6 of fine-tuning, and EarlyStopping restored the best weights rather than keeping the degraded final epoch.
- **`ReduceLROnPlateau` helps stabilise fine-tuning** — automatically halves learning rate when val loss stalls, preventing the oscillation seen in v1.
- **`sync=False` on CustomJob means the resource isn't immediately available** — accessing `job.display_name` right after `job.run(sync=False)` raises `RuntimeError` because the API call is still in flight. Add a `time.sleep(15)` or check status in a separate cell.

---

## Project 3 — Bitcoin Volatility Forecasting (Time Series)

### Overview

| Attribute | Detail |
|-----------|--------|
| **Dataset** | BigQuery public crypto datasets (`bigquery-public-data.crypto_bitcoin`) + derived daily features |
| **Task** | Forecast daily Bitcoin price volatility (realized variance or absolute log returns) |
| **Pipeline** | BigQuery feature engineering → GCS export → ARIMA_PLUS baseline → GBT → LSTM → Vertex AI custom training → batch prediction |
| **Estimated Cost** | ~$8-12 |
| **Estimated Time** | 6-8 hours |

### Skills Reinforced

| Skill | Original Lab | How It's Applied Here |
|-------|-------------|----------------------|
| BigQuery feature engineering | Lab 1, Lab 9 | Daily aggregations, rolling windows, on-chain metrics |
| ARIMA_PLUS in BQML | Lab 9 | Baseline volatility forecast |
| GBT with temporal features | Lab 9 | Lag features, rolling stats, cyclical encoding |
| LSTM on windowed sequences | Lab 9 | Sequence-to-one volatility prediction |
| Time-aware splitting | Lab 9 | Chronological train/val/test — no future leakage |
| Traffic splitting for comparison | Mini-Lab D | Deploy two models, compare live predictions |

### New Skill: ARIMA_PLUS with Exogenous Variables

Lab 9's ARIMA_PLUS used only date + temperature (univariate). Bitcoin data enables multivariate forecasting:

- Feed external regressors into ARIMA_PLUS: transaction volume, active addresses, average transaction value, hash rate
- Compare univariate ARIMA_PLUS (price only) vs multivariate (price + on-chain features)
- Document when exogenous variables help vs hurt forecast accuracy
- Exam pattern: "forecast with external features in BQML" → ARIMA_PLUS with additional columns

This is a frequently tested topic that the lab series hasn't covered yet.

### Parts

| Part | Focus | Estimated Time |
|------|-------|---------------|
| 1 | Setup + Data Exploration (BigQuery crypto tables) | ~45 min |
| 2 | Feature Engineering (daily OHLCV + on-chain metrics) | ~1 hr |
| 3 | ARIMA_PLUS Baseline (univariate + multivariate) | ~45 min |
| 4 | GBT with Engineered Features | ~1 hr |
| 5 | LSTM on Windowed Sequences | ~1 hr |
| 6 | Model Comparison + Batch Prediction | ~45 min |
| 7 | Traffic Split Deployment (champion vs challenger) | ~45 min |
| 8 | Cleanup | ~15 min |

### Success Criteria

- Daily Bitcoin features derived from BigQuery blockchain data
- Univariate vs multivariate ARIMA_PLUS comparison documented
- GBT and LSTM models trained and compared on same test period
- Batch predictions generated for held-out period
- Traffic splitting deployed with champion/challenger
- All resources cleaned up

### Dataset Notes

- BigQuery tables: `bigquery-public-data.crypto_bitcoin.transactions`, `bigquery-public-data.crypto_bitcoin.blocks`
- Need to derive daily OHLCV from transaction-level data, or supplement with external price source
- On-chain features: daily transaction count, total value transferred, unique addresses, average fee, block count
- Bitcoin's extreme volatility makes this a harder forecasting problem than Lab 9's temperature data — expect higher RMSE and more model uncertainty
- Time-aware splitting is critical — crypto markets have distinct regime changes (bull/bear cycles)
- Important learning: "no single best method" for crypto volatility — reinforces the model comparison discipline

---

## Cross-Project Exam Coverage

These three projects together touch the following exam topics not yet covered by the labs:

| New Topic | Project | Exam Pattern |
|-----------|---------|-------------|
| KFP pipeline on real data (not toy) | Project 1 | "Orchestrate ML workflow with conditional logic" |
| Low-confidence → human review routing | Project 1 | "Handle uncertain predictions in production" |
| Image explainability (XRAI/occlusion sensitivity) | Project 2 | "Explain image classification decisions" |
| ExplanationSpec configuration gotchas | Project 2 | "Configure explainability for deployed model" |
| ARIMA_PLUS with exogenous variables | Project 3 | "Forecast with external features in BQML" |
| Multi-model traffic splitting (real comparison) | Project 3 | "Compare models in production" |

---

## Design Principles

All projects follow the established lab patterns:

1. **Data lives in BigQuery** — explore and prepare with SQL
2. **Export to Cloud Storage** — CSV, TFRecord, or image files
3. **Train in Vertex AI** — custom training with prebuilt containers
4. **Log to Experiments** — parameters, metrics, artifacts
5. **Evaluate** — appropriate metrics for the problem type
6. **Serve** — online endpoint or batch prediction
7. **Track lineage** — data → model → endpoint
8. **Clean up** — delete all resources after each project

### Container Strategy

Same as the labs — prebuilt first unless there's a specific reason for custom:

| Project | Training Container | Serving Container |
|---------|-------------------|-------------------|
| Project 1 (Text) | Prebuilt TF 2.15 | CPR (custom predictor) |
| Project 2 (Image) | Prebuilt TF 2.15 (CPU — GPU quota unavailable) | Prebuilt TF 2.15 (base64 serving) |
| Project 3 (Time Series) | Prebuilt sklearn 1.0 / TF 2.15 | Prebuilt sklearn / TF |