# Hotel Customer No-Show Prediction

## Project overview

This project predicts whether a hotel booking will become a no-show (`no_show = 1`). The result can help the hotel prioritise reminders, confirmations or deposit policies while avoiding a policy that treats every booking as high risk.

This is a binary classification problem because the target has two outcomes: show (`0`) and no-show (`1`). The dataset contains 119,391 records. One record without a valid target is removed, leaving 119,390 usable observations; 37.04% are no-shows.

## Project structure

```text
hotel_no_show_prediction/
├── eda.ipynb
├── README.md
├── requirements.txt
├── data/
│   └── data.csv
├── src/
│   ├── data_preparation.py
│   ├── model_training.py
│   └── config.yaml
└── main.py
```

Running `main.py` also creates `best_model.joblib`, `model_comparison.csv` and `final_test_results.json`. These reproducible outputs are excluded from Git by default.

## Data investigation

The EDA notebook checks the schema, data types, duplicates, missing values, category consistency, distributions, no-show rates, correlations, statistical significance and effect sizes. Important findings include:

- `booking_id` is unique and is excluded from modelling.
- One row has no valid target and is removed.
- `room` and `price` have 18.10% and 20.84% missing values respectively.
- `arrival_month` has inconsistent capitalisation, which is standardised.
- `num_adults` mixes words and digits, which are mapped to numeric values.
- 5,696 checkout days have negative signs. Their absolute values are valid days from 1 to 31, so the sign is corrected.
- Price contains SGD or USD prefixes. Currency and numeric amount are separated rather than treating the full string as a category.
- Country has the strongest standalone association with no-show status, followed by branch. Numerical features have weak individual effects, supporting multivariate model comparison.

Missing-value imputation is learned only from each training fold. Missingness indicators preserve potentially useful information about absent `room` and `price` values.

## Modelling approach

The data uses an 80/20 stratified train-test split. All preprocessing is contained inside scikit-learn pipelines to prevent information from validation or test data leaking into training.

The pipeline:

1. validates the required schema and target;
2. standardises categories and repairs documented formatting errors;
3. engineers booking lead months, approximate stay length and total guests;
4. median-imputes and scales numerical features;
5. mode-imputes and one-hot encodes categorical features;
6. compares three suitable predictive models plus a naive benchmark using the same five stratified folds;
7. tunes the best model using training data only;
8. selects a probability threshold from out-of-fold training predictions; and
9. evaluates once on the untouched test set.

The three predictive models are Logistic Regression, Random Forest and Histogram Gradient Boosting. Logistic Regression provides an interpretable linear baseline. Random Forest captures nonlinearities and interactions. Histogram Gradient Boosting builds trees sequentially to correct prior errors and is efficient for this tabular dataset.

## Evaluation choice

Missing a likely no-show may leave a room unused, so recall is important. However, excessive false alerts could inconvenience reliable customers. The primary selection metric is therefore **F2**, which gives recall twice the weight of precision. ROC-AUC and average precision assess probability ranking independently of the final classification threshold.

The final threshold is selected using out-of-fold predictions from the training set, maximising F2 while requiring precision of at least 0.55. The test set is not used for model or threshold selection.

## Cross-validation results

| Model | Precision | Recall | F1 | F2 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| Histogram Gradient Boosting | 0.644 | 0.647 | 0.645 | **0.646** | 0.795 | 0.724 |
| Logistic Regression | 0.589 | **0.652** | 0.619 | 0.639 | 0.744 | 0.624 |
| Random Forest | **0.714** | 0.598 | **0.651** | 0.618 | **0.816** | **0.768** |
| Dummy prior | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 | 0.370 |

Values are five-fold means at the models' default 0.50 thresholds. Histogram Gradient Boosting has the highest F2 and a better precision-recall balance than Logistic Regression. Random Forest ranks risks particularly well, as shown by its ROC-AUC and average precision, but its lower recall makes its default operating point less suitable for the stated objective.

## Selected model and final test result

Histogram Gradient Boosting was selected, tuned and assigned a probability threshold of 0.40. On the untouched test set it achieved:

| Metric | Result |
|---|---:|
| Precision | 0.559 |
| Recall | **0.785** |
| Specificity | 0.636 |
| F1 | 0.653 |
| F2 | **0.726** |
| ROC-AUC | 0.804 |
| Average precision | 0.738 |

The test confusion matrix contains 6,942 correctly identified no-shows, 1,903 missed no-shows, 9,559 correctly identified shows and 5,474 false alerts. The threshold intentionally increases recall to catch more no-shows while retaining the configured minimum precision.

The model should support targeted operational action rather than automatic rejection. Country is predictive in this historical dataset, but it should be monitored for fairness and policy implications. Before deployment, the hotel should attach real monetary costs to false negatives and false positives, validate performance on newer time periods and monitor drift. The available month fields do not include years, so exact lead time and stay duration cannot be reconstructed.

## How to run

Open a terminal in this project folder, then run:

```bash
conda activate hdbenv
python -m pip install -r requirements.txt
python main.py
```

If `hdbenv` is not available, create a project-specific environment instead:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

For the notebook, select the same Python environment as the Jupyter kernel and run all cells in `eda.ipynb` from top to bottom. Run commands from the project root so the relative path `data/data.csv` resolves correctly.

## Reproducibility

The random seed, split ratio, feature lists, cross-validation settings, threshold rule and file paths are centralised in `src/config.yaml`. The complete pipeline can be reproduced with one command, and the saved model contains both preprocessing and the trained classifier.
