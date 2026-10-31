# Hotel Customer No-Show Prediction

## Objective

This project predicts whether a hotel booking will become a no-show (`no_show = 1`). It is designed to support targeted reminders, confirmation calls, deposits, or overbooking review rather than automatic booking rejection.

The dataset contains 119,391 records. One row without a target is removed, leaving 119,390 usable bookings. No-shows account for 37.04%, giving a moderate majority:minority ratio of approximately 1.70:1.

## Project structure

```text
hotel_no_show_final/
├── eda.ipynb
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── data/
│   └── data.csv
├── src/
│   ├── data_preparation.py
│   ├── model_training.py
│   └── config.yaml
├── tests/
│   └── test_pipeline.py
└── main.py
```

Training creates a timestamped directory under `artifacts/` containing the model bundle, configuration snapshot, run metadata, logs, comparison results, threshold analysis, calibration results, feature importance, and final test metrics. Generated artifacts are excluded from Git because they can be reproduced from the tracked code and data.

## Exploratory analysis

The EDA notebook includes:

- explicit full-data and stratified-sample decisions;
- schema, target, duplicate, category, and missingness checks;
- missingness relationships with the target;
- chronological month and approximate lead-time analysis;
- ordered and cyclical temporal reasoning;
- subgroup and interaction analysis across branch, country, and first-time status;
- chi-square diagnostic counts and Cramér's V effect sizes;
- an exploratory model-based feature-importance screen;
- an explicit imbalance and evaluation plan; and
- a consolidated limitations section.

The full dataset is used for summaries, tests, and modelling. A reproducible 30,000-row stratified sample is used only for repeated plots and the quick exploratory importance model.

## Data preparation and feature engineering

The pipeline validates the schema and binary target before transformation. It then:

1. removes the single row without a target;
2. standardises text categories and month capitalisation;
3. maps adult counts written as words or digits to numeric values;
4. corrects negative checkout-day sign errors after confirming their absolute values are valid days;
5. separates price amount and currency and converts USD to SGD using the configured fixed rate;
6. preserves missingness through fold-local imputation indicators;
7. removes the unique `booking_id` identifier;
8. creates approximate booking lead months and length of stay;
9. creates cyclical sine/cosine month features;
10. creates total guests and SGD price per guest; and
11. creates branch-country and first-time interaction features.

The target and identifier are explicitly asserted to be absent from model inputs. Preprocessing is unfitted when returned by data preparation and is fitted only inside each model's cross-validation or training pipeline.

## Class imbalance strategy

Accuracy alone is unsuitable because a model could favour the majority class. The same five stratified folds compare:

- Dummy prior baseline;
- unweighted Logistic Regression;
- class-weighted Logistic Regression;
- class-weighted Random Forest;
- class-weighted Histogram Gradient Boosting; and
- Histogram Gradient Boosting with random oversampling.

Random oversampling occurs inside the estimator's `fit` method after fold-local preprocessing. Validation and test rows are never resampled. This allows a direct comparison of no adjustment, class weighting, and resampling without leakage.

## Model selection and evaluation

The primary model-selection metric is average precision (PR-AUC), which assesses minority-class probability ranking without assuming a classification threshold. Supporting metrics include precision, recall, F1, F2, balanced accuracy, ROC-AUC, MCC, Brier score, log loss, specificity, and confusion-matrix counts.

Five-fold cross-validation results:

| Model | Precision | Recall | F2 | Balanced accuracy | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest — class weighted | **0.707** | 0.604 | 0.622 | **0.728** | **0.814** | **0.764** |
| Histogram Gradient Boosting — class weighted | 0.646 | 0.649 | 0.648 | 0.720 | 0.797 | 0.727 |
| Histogram Gradient Boosting — oversampled | 0.648 | **0.650** | **0.649** | 0.721 | 0.797 | 0.727 |
| Logistic Regression — unweighted | 0.662 | 0.506 | 0.531 | 0.677 | 0.745 | 0.628 |
| Logistic Regression — class weighted | 0.595 | 0.644 | 0.633 | 0.693 | 0.745 | 0.626 |
| Dummy prior | 0.000 | 0.000 | 0.000 | 0.500 | 0.500 | 0.370 |

Class-weighted Random Forest is selected because it has the strongest PR-AUC, ROC-AUC, and balanced accuracy. The result also demonstrates that imbalance handling is not selected by assumption: weighting and oversampling are both evaluated, and weighting performs better for probability ranking in this dataset.

## Calibration and cost-aware threshold

After tuning, the selected model is probability-calibrated using sigmoid calibration. A separate stratified validation partition from the training data is used for threshold selection. The test set is not used to select the model, hyperparameters, calibration method, or threshold.

The configured scenario assigns a false negative five times the cost of a false positive. Thresholds from 0.10 to 0.90 are evaluated. The selected threshold minimises validation cost while requiring precision of at least 0.50, preventing the model from achieving high recall by flagging almost every booking.

This 5:1 ratio is a transparent scenario, not a claimed monetary estimate. Hotel operations should replace it with actual empty-room and intervention costs.

## Final untouched-test results

The calibrated class-weighted Random Forest selects a threshold of **0.20** and achieves:

| Metric | Result |
|---|---:|
| Precision | 0.504 |
| Recall | **0.881** |
| Specificity | 0.490 |
| F1 | 0.641 |
| F2 | **0.767** |
| Balanced accuracy | 0.686 |
| MCC | 0.375 |
| ROC-AUC | 0.821 |
| PR-AUC | **0.778** |
| Brier score | 0.158 |
| Log loss | 0.482 |

Confusion matrix counts:

- True positives: 7,796
- False negatives: 1,049
- True negatives: 7,363
- False positives: 7,670

Under the configured 5:1 cost scenario, the selected threshold has an estimated test cost of 12,915 units. This is 8,034 units lower than using the default 0.50 threshold and 31,310 units lower than predicting every booking as a show. These values compare policies under the stated assumptions; they are not currency amounts.

Validation permutation importance identifies `country_first_time`, length of stay, booking lead time, checkout day, arrival day, cyclical arrival month, room, and country as leading predictors. Importance is predictive rather than causal.

## Installation

The project was validated with Python 3.12.

```bash
conda activate hdbenv
python -m pip install -r requirements.txt
```

For notebook and development tools:

```bash
python -m pip install -r requirements-dev.txt
```

## Run training

From the project root:

```bash
python main.py train
```

`python main.py` remains equivalent to `python main.py train`.

## Run tests

```bash
python -m pytest -q
```

The tests cover configuration validation, missing schema fields, cleaning edge cases, preprocessing compatibility, feature engineering, reproducible stratification, leakage boundaries, random oversampling, and cost-aware threshold selection.

## Batch prediction

After training, locate the latest model bundle under the timestamped artifact directory and run:

```bash
python main.py predict \
  --model artifacts/run_YYYYMMDD_HHMMSS/model_bundle.joblib \
  --input path/to/new_bookings.csv \
  --output predictions.csv
```

The input must contain the original booking feature columns but does not need `no_show`. The output contains booking ID, calibrated no-show probability, and the thresholded prediction.

## Limitations and responsible use

- The dataset contains months but no year, so exact lead times, stay lengths, and temporal holdout evaluation are unavailable.
- The fixed USD-to-SGD conversion rate is an assumption because transaction dates are unavailable.
- Country is strongly predictive but may encode demographic or structural differences. It requires fairness review, subgroup monitoring, and human-governed policy use.
- There is no free-text field, so text-derived analysis is not applicable.
- The cost ratio must be validated with real hotel economics.
- Performance and calibration must be checked on newer bookings and monitored for drift before deployment.

The model should prioritise proportionate outreach or review. It should not automatically reject customers or impose punitive policies based solely on a prediction.
