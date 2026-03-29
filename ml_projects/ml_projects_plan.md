# GCP ML Engineer — Reinforcement Projects

**Purpose:** Apply skills from Labs 1–10 and Mini-Labs A–E to new domains and datasets, reinforcing core competencies while adding one new exam-relevant skill per project.

**Approach:** Each project follows the full GCP pipeline (BigQuery → GCS → Vertex AI → Endpoint) and is designed to be executed in a single chat session.

**Sequencing:** Projects are ordered by priority. Each project is self-contained.

---

## Progress Summary

| Project | Domain | Status | Key Outcome |
|---------|--------|--------|-------------|
| Project 1: Support Ticket Routing | Text / NLP | ✅ Complete | Pipeline (balanced weights) | Weighted F1 | 0.5156 |
| Project 2: Serengeti Wildlife Species ID | Image / Vision | ⬜ Not Started | — |
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

## Project 2 — Serengeti Wildlife Species ID (Image Classification)

### Overview

| Attribute | Detail |
|-----------|--------|
| **Dataset** | Snapshot Serengeti via LILA BC (subset: top 10 species + empty class, ~50-80k images) |
| **Task** | Multi-class image classification with empty frame filtering |
| **Pipeline** | Download subset → BigQuery metadata → TFRecord sharded pipeline → transfer learning → Vertex AI custom training → AutoML comparison → endpoint with base64 serving |
| **Estimated Cost** | ~$10-15 |
| **Estimated Time** | 8-10 hours |

### Skills Reinforced

| Skill | Original Lab | How It's Applied Here |
|-------|-------------|----------------------|
| Transfer learning (MobileNetV2/EfficientNet) | Lab 8 | Fine-tune on wildlife species with freeze → unfreeze |
| TFRecord sharded pipeline | Mini-Lab C | Convert images to sharded TFRecords for efficient I/O |
| tf.data pipeline | Lab 8, Mini-Lab C | Image augmentation, prefetch, interleave for sharded reads |
| Base64 serving pattern | Lab 8 | Bake decode+preprocess into SavedModel graph |
| AutoML comparison | Lab 2, Lab 8 | AutoML Vision vs custom model cost/accuracy tradeoff |
| Experiment tracking | Lab 5 | Log per-class metrics across model variants |

### New Skill: XRAI Explainability on Image Model

Mini-Lab B covered Sampled Shapley (tabular, any model) and Integrated Gradients (tabular, neural nets) but never used XRAI — the image-specific explainability method. This project completes the trifecta:

- Deploy model with `ExplanationSpec` using XRAI parameters
- Request explanations via the endpoint
- Visualize region-level attributions (e.g. "model focused on stripes → zebra")
- Compare XRAI vs Integrated Gradients on the same image model
- Document the full explainability decision framework with all three methods tested hands-on

Exam pattern: "Explain why the image was classified as X" → XRAI (region-level) or IG (pixel-level).

### Parts

| Part | Focus | Estimated Time |
|------|-------|---------------|
| 1 | Setup + Data Download (LILA BC subset) | ~1 hr |
| 2 | Data Exploration + BigQuery Metadata Table | ~30 min |
| 3 | TFRecord Pipeline (sharded, with augmentation) | ~1 hr |
| 4 | Transfer Learning Model (MobileNetV2 fine-tune) | ~1.5 hr |
| 5 | AutoML Vision Comparison | ~1 hr (mostly waiting) |
| 6 | Model Deployment (base64 serving + XRAI) | ~1.5 hr |
| 7 | XRAI Explainability Analysis | ~1 hr |
| 8 | Cleanup | ~15 min |

### Success Criteria

- Subset of 50-80k images across 10+ species loaded and processed
- Custom model achieves >90% accuracy on held-out test set
- AutoML comparison documented with cost/accuracy tradeoff
- XRAI explanations visualized showing meaningful region attributions
- Base64 serving pattern working end-to-end
- All resources cleaned up

### Dataset Notes

- Source: https://lila.science/datasets/snapshot-serengeti/
- Full dataset: ~7.1M images, 61 categories — far too large; subset required
- Subsetting strategy: top 10 species by image count + empty class, one image per sequence (avoid near-duplicates)
- Challenges: class imbalance, night images (IR), motion blur, empty frames (~76% of full dataset)
- Images are JPEG, varying resolution
- COCO Camera Traps JSON format for annotations

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
| XRAI explainability (hands-on) | Project 2 | "Explain image classification decisions" |
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
| Project 2 (Image) | Prebuilt TF 2.15 | Prebuilt TF 2.15 (base64 serving) |
| Project 3 (Time Series) | Prebuilt sklearn 1.0 / TF 2.15 | Prebuilt sklearn / TF |