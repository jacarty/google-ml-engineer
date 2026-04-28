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

## Project 3 — Bitcoin Volatility Forecasting (Time Series) ✅

### Overview

| Attribute | Detail |
|-----------|--------|
| **Dataset** | Binance public archives BTCUSDT 5-minute bars (905k rows, 2017-08-17 to 2026-03-31) + BigQuery `bigquery-public-data.crypto_bitcoin.blocks` for on-chain features |
| **Task** | One-step-ahead forecasting of daily realized variance from intraday data; primary target is log(RV) |
| **Pipeline** | Binance archive → GCS → BigQuery raw load → typed/partitioned table → daily RV + on-chain features → 4 models + ensemble → batch evaluation |
| **Actual Cost** | ~$1-2 (well under the $8-12 budget) |
| **Actual Time** | ~9-10 hours |

### Results

**Test period:** 2025-01-01 to 2026-03-31, 455 days. **Target:** log of daily realized variance, computed from 5-minute log returns. **Baseline target std:** 1.219; baseline target mean: -7.04 (training period).

| Model | RMSE log_rv | MAE log_rv | RMSE rv | MAE rv | QLIKE |
|-------|------------:|-----------:|--------:|-------:|------:|
| ARIMA_PLUS univariate | 0.872 | 0.704 | 0.000796 | 0.000414 | 0.409 |
| ARIMA_PLUS_XREG multivariate | 1.061 | 0.887 | 0.000990 | 0.000622 | 0.430 |
| GBT (one-step-ahead) | 0.829 | 0.645 | 0.000728 | 0.000376 | 0.297 |
| LSTM (one-step-ahead) | 0.812 | 0.623 | 0.000746 | 0.000329 | 0.627 |
| **Ensemble (GBT + LSTM avg)** | **0.642** | **0.487** | **0.000711** | **0.000300** | **0.288** |

The ensemble beat all individual models on every metric. 26% RMSE improvement over the ARIMA univariate baseline, 30% QLIKE improvement.

**Bias decomposition:**

| Model | Mean residual | Std residual |
|-------|--------------:|-------------:|
| GBT | +0.439 | 0.704 |
| LSTM | -0.354 | 0.732 |
| **Ensemble** | **+0.043** | **0.641** |

GBT and LSTM had opposite-direction biases of similar magnitude. Averaging cancelled them, dropping the ensemble residual mean to near-zero.

### Skills Reinforced

| Skill | Original Lab | How It's Applied Here |
|-------|-------------|----------------------|
| BigQuery feature engineering | Lab 1, Lab 9 | Daily aggregations from intraday bars; on-chain metric aggregation |
| ARIMA_PLUS in BQML | Lab 9 | Univariate + multivariate (XREG variant); ML.EXPLAIN_FORECAST decomposition |
| GBT with temporal features | Lab 9 | HAR-style lag and rolling-mean features in BOOSTED_TREE_REGRESSOR |
| LSTM on windowed sequences | Lab 9 | 30-day windows, custom training script on Vertex AI |
| Time-aware splitting | Lab 9 | Strict chronological train/val/test (no random splits) |
| Custom training on Vertex AI | Labs 7-9 | LSTM via prebuilt TF 2.15 container, CustomJob submission |
| Container consistency | Labs 2, 7-9 | TFSMLayer used to load Keras 2 SavedModel under Keras 3 locally |

### New Skill: ARIMA_PLUS with Exogenous Regressors (XREG)

Executed via `ARIMA_PLUS_XREG` model_type with two lagged exogenous features (`tx_count_lag1`, `quote_volume_lag1`). The implementation surfaced two BQML-specific gotchas worth knowing:

- `ARIMA_PLUS_XREG` does NOT support `decompose_time_series=TRUE` — that option is exclusive to plain ARIMA_PLUS. Trade-off: gain exogenous regressors, lose the trend/seasonal decomposition view.
- `ML.FORECAST` for XREG models requires the exogenous values for the forecast period to be passed as a third argument table. Default `horizon` is 3 if not specified explicitly in the STRUCT — caused a confusing "only 3 forecasts returned" failure.

Exam pattern confirmed: "forecast with external features in BQML" → `ARIMA_PLUS_XREG` with exogenous time series passed at training and prediction time.

### Parts (As Executed)

| Part | Focus | Actual Time |
|------|-------|-------------|
| 1 | Setup + ingest 5-min BTCUSDT bars from Binance to GCS to BigQuery | ~1 hour |
| 2 | Daily realized variance + on-chain feature aggregation | ~45 min |
| 3 | EDA: stationarity (ADF/KPSS), autocorrelation (ACF/PACF), exogenous correlation analysis | ~1 hour |
| 4 | ARIMA_PLUS univariate + multivariate XREG; ML.FORECAST + ML.EXPLAIN_FORECAST | ~1.5 hours |
| 5 | GBT training and evaluation (after debugging a stuck CREATE MODEL job) | ~1.5 hours |
| 6 | LSTM via Vertex AI CustomJob; SavedModel inference via TFSMLayer | ~1.5 hours |
| 7 | Ensemble (equal-weight average of GBT + LSTM); five-way comparison | ~30 min |

### Key Learnings

**Modelling**

- **Volatility forecasting is genuinely tractable.** ACF showed long-memory persistence (0.23 at lag 80 days), PACF dominated by lag 1 with significant lag 7 / 14 spikes (weekly seasonality). The signal is real and forecastable.
- **ARIMA_PLUS converges to flat-line forecasts at long horizons.** ML.EXPLAIN_FORECAST decomposition revealed that beyond the first ~30 days of forecasting, only the weekly seasonal component contributes dynamics — the trend collapses to the long-run mean. This is the AR(1) decay-to-mean property in action.
- **Tree models are conservative; neural models are aggressive.** GBT clusters predictions near the training distribution mean (cannot extrapolate). LSTM extrapolates freely along learned patterns and can dramatically under- or over-predict regime extremes. These are inductive biases of the model families, not bugs.
- **The ensemble of two adequate models beats either individually.** Bias cancellation is the mechanism: GBT residual mean +0.44, LSTM residual mean -0.35, ensemble residual mean +0.04. Variance also reduces from averaging. Equal-weight averaging is often within 1-2% of more sophisticated stacking — try simple averages first.
- **Different metrics produce different "winners".** LSTM was best on RMSE/MAE log_rv but worst on QLIKE (the canonical vol metric that asymmetrically punishes underestimation). Don't trust a single number for model selection.

**Design discipline**

- **EDA findings have to translate to model design choices, not just be acknowledged.** EDA showed that on-chain feature *levels* correlated with log_rv only because both shared a non-stationary trend; *changes* had near-zero correlation. The first ARIMA_PLUS_XREG attempt used levels and overfit to the trend (1.06 RMSE on test, *worse* than the univariate baseline at 0.87). The GBT and LSTM models used differenced exogenous features and avoided the trap.
- **In-sample improvement does not guarantee out-of-sample improvement.** ARIMA_PLUS_XREG's AIC dropped 913 points and in-sample variance fell 29% relative to the univariate model — yet test RMSE got *worse*. With 3,000+ observations, almost anything is statistically significant; effect size and out-of-sample stability matter more than p-values.
- **Statistical significance is not effect size, especially with non-stationary data.** Two trending series will spuriously correlate even when there's no real predictive relationship; differencing the inputs is the canonical defence.

**BQML / Vertex AI gotchas**

- **`ARIMA_PLUS_XREG` lacks `decompose_time_series` support** — this option is for plain `ARIMA_PLUS` only.
- **`ML.FORECAST` default `horizon` is 3** — must be set explicitly inside the STRUCT, even for XREG models that take an exogenous-values table as the third argument.
- **`BOOSTED_TREE_REGRESSOR` with `data_split_method='CUSTOM'` + `early_stop=TRUE` can hang indefinitely.** First training attempt consumed 6h 46m of slot time without completing a single iteration. Stripping to `WHERE split = 'train'` only and removing early_stop made it train in ~5 minutes. Lesson: start with the simplest CREATE MODEL config and add features one at a time.
- **GCS directory marker blobs cause `IsADirectoryError` on local download.** Filter blobs whose names end in `/` when downloading SavedModel directories — TF saves an empty `assets/` blob that breaks naive recursive downloads.
- **Keras 2 SavedModels need TFSMLayer, not load_model, under Keras 3.** Same lesson as Lab 9 / Project 2. The signature can also appear corrupted (`unknown:0` tensor names) but TFSMLayer handles it correctly.

**Data engineering**

- **Public datasets are tied to their region.** `bigquery-public-data` is in `US` multi-region; queries that read from it must write to a dataset in `US`. Initial dataset created in `us-central1` had to be dropped and recreated.
- **Binance changed timestamp units mid-2024.** Pre-2024 monthly archives have timestamps in milliseconds (13 digits); 2025+ archives use microseconds (16 digits). Magnitude-based CASE expression handled both cleanly without re-ingesting any data.
- **The "raw → typed/partitioned" two-step pattern is industry standard for a reason.** Letting raw data land as-is means re-running the transform doesn't require re-ingesting. The transform CTAS handled the timestamp-unit conversion in pure SQL.
- **Partition filtering is the cheap-query enabler.** Querying 16 years of `crypto_bitcoin.blocks` with partition filter `timestamp_month >= '2017-08-01'` cost $0.0001. Without the filter, the same query would have scanned the full table.

**Cost discipline**

- **`storage_client.batch()` for bulk GCS deletes** — packs up to 100 deletions into one HTTP round trip, ~50x faster than serial deletes for cleanup.
- **`total_bytes_billed` after every BQ query** — printing the actual cost surfaces problem queries before they hit a monthly bill.
- **Ensemble was free improvement.** Computing GBT and LSTM predictions individually was the expensive part; averaging them cost nothing and produced the best model.

### Departures from Original Plan

- **Skipped traffic-split deployment (champion vs challenger).** The original Part 7 was "deploy two models, compare live predictions" via Mini-Lab D pattern. Instead, built batch predictions and an ensemble — which produced a better evaluation outcome and didn't incur ongoing endpoint costs. The ensemble result made champion/challenger framing redundant.
- **Realized variance from intraday data instead of daily OHLCV.** The original plan suggested deriving daily features from `bigquery-public-data.crypto_bitcoin.transactions` or external price sources. Instead, used Binance's public 5-minute bar archives, which gave a higher-quality realized variance target than daily-only data could provide.
- **Used blocks table only, not transactions table.** Cost discipline — the transactions table is several TB and would have added significant query cost for marginal benefit. Block-level on-chain features (block count, total transaction count, average block size) captured most of the predictive signal.

---