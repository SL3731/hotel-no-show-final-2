"""Imbalance-aware model comparison, calibration, thresholding, and evaluation."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

LOGGER = logging.getLogger(__name__)


class RandomOversamplingClassifier(ClassifierMixin, BaseEstimator):
    """Clone an estimator and oversample only the training fold inside ``fit``."""

    def __init__(self, estimator: Any, random_state: int = 42):
        self.estimator = estimator
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomOversamplingClassifier":
        """Randomly duplicate minority observations until classes are balanced."""
        X_array, y_array = np.asarray(X), np.asarray(y)
        classes, counts = np.unique(y_array, return_counts=True)
        if len(classes) != 2:
            raise ValueError("Random oversampling requires exactly two target classes.")
        majority_size = int(counts.max())
        generator = np.random.default_rng(self.random_state)
        sampled_indices = []
        for label, count in zip(classes, counts):
            indices = np.flatnonzero(y_array == label)
            sampled_indices.extend(indices.tolist())
            if count < majority_size:
                sampled_indices.extend(
                    generator.choice(indices, size=majority_size - count, replace=True).tolist()
                )
        sampled_indices = generator.permutation(sampled_indices)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X_array[sampled_indices], y_array[sampled_indices])
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions from the fitted wrapped estimator."""
        return self.estimator_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities from the fitted wrapped estimator."""
        return self.estimator_.predict_proba(X)


class ModelTraining:
    """Compare imbalance strategies and tune, calibrate, and evaluate one model."""

    def __init__(self, config: dict[str, Any], preprocessor: Any):
        self.config = config
        self.preprocessor = preprocessor
        self.random_state = int(config["random_state"])

    def _models(self) -> dict[str, Any]:
        unweighted_hgb = HistGradientBoostingClassifier(
            max_iter=220, learning_rate=0.08, l2_regularization=1.0,
            random_state=self.random_state,
        )
        return {
            "Dummy Prior": DummyClassifier(strategy="prior"),
            "Logistic Regression - Unweighted": LogisticRegression(
                max_iter=2_000, random_state=self.random_state
            ),
            "Logistic Regression - Class Weighted": LogisticRegression(
                max_iter=2_000, class_weight="balanced", random_state=self.random_state
            ),
            "Random Forest - Class Weighted": RandomForestClassifier(
                n_estimators=180, min_samples_leaf=2, max_features="sqrt",
                class_weight="balanced_subsample", random_state=self.random_state, n_jobs=2,
            ),
            "Histogram Gradient Boosting - Class Weighted": HistGradientBoostingClassifier(
                max_iter=220, learning_rate=0.08, l2_regularization=1.0,
                class_weight="balanced", random_state=self.random_state,
            ),
            "Histogram Gradient Boosting - Random Oversampling": RandomOversamplingClassifier(
                estimator=unweighted_hgb, random_state=self.random_state
            ),
        }

    def _pipeline(self, model: Any) -> Pipeline:
        """Keep preprocessing inside every fit and cross-validation fold."""
        return Pipeline([
            ("preprocessor", clone(self.preprocessor)),
            ("model", model),
        ])

    def compare_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        """Compare all models on identical stratified folds using rich metrics."""
        cv = StratifiedKFold(
            n_splits=int(self.config["cv_folds"]), shuffle=True, random_state=self.random_state
        )
        scoring = {
            "precision": make_scorer(precision_score, zero_division=0),
            "recall": make_scorer(recall_score, zero_division=0),
            "f1": make_scorer(f1_score, zero_division=0),
            "f2": make_scorer(fbeta_score, beta=2),
            "balanced_accuracy": "balanced_accuracy",
            "roc_auc": "roc_auc",
            "average_precision": "average_precision",
        }
        rows = []
        for name, model in self._models().items():
            LOGGER.info("Cross-validating %s", name)
            scores = cross_validate(
                self._pipeline(model), X_train, y_train, cv=cv, scoring=scoring, n_jobs=1
            )
            row = {"model": name}
            for metric in scoring:
                row[f"cv_{metric}_mean"] = float(scores[f"test_{metric}"].mean())
                row[f"cv_{metric}_std"] = float(scores[f"test_{metric}"].std())
            rows.append(row)
        return pd.DataFrame(rows).sort_values(
            ["cv_average_precision_mean", "cv_f2_mean"], ascending=False
        ).reset_index(drop=True)

    def tune_selected_model(self, selected_name: str, X_fit: pd.DataFrame, y_fit: pd.Series) -> RandomizedSearchCV:
        """Tune the selected model using config-owned ranges and training data only."""
        spaces = self.config["hyperparameters"]
        if selected_name not in spaces:
            raise ValueError(f"No tuning space configured for {selected_name!r}.")
        cv = StratifiedKFold(
            n_splits=int(self.config["tuning_cv_folds"]), shuffle=True, random_state=self.random_state
        )
        search = RandomizedSearchCV(
            self._pipeline(self._models()[selected_name]),
            param_distributions=spaces[selected_name],
            n_iter=int(self.config["tuning_iterations"]),
            scoring=self.config["primary_metric"],
            cv=cv,
            random_state=self.random_state,
            n_jobs=1,
            refit=True,
        )
        LOGGER.info("Tuning %s on fit partition", selected_name)
        search.fit(X_fit, y_fit)
        return search

    def threshold_table(self, y_true: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
        """Evaluate validation thresholds using F2 and explicit business costs."""
        settings = self.config["threshold"]
        thresholds = np.arange(
            float(settings["minimum"]),
            float(settings["maximum"]) + float(settings["step"]) / 2,
            float(settings["step"]),
        )
        rows = []
        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
            cost = fp * float(settings["false_positive_cost"]) + fn * float(settings["false_negative_cost"])
            rows.append({
                "threshold": float(threshold),
                "precision": float(precision_score(y_true, predictions, zero_division=0)),
                "recall": float(recall_score(y_true, predictions, zero_division=0)),
                "f1": float(f1_score(y_true, predictions, zero_division=0)),
                "f2": float(fbeta_score(y_true, predictions, beta=float(settings["beta"]), zero_division=0)),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "estimated_cost": float(cost),
                "cost_per_booking": float(cost / len(y_true)),
                "true_negative": int(tn),
                "true_positive": int(tp),
            })
        return pd.DataFrame(rows)

    def select_threshold(self, table: pd.DataFrame) -> tuple[float, dict[str, float]]:
        """Minimise cost subject to a precision guardrail, with F2 as tie-breaker."""
        minimum_precision = float(self.config["threshold"]["minimum_precision"])
        eligible = table.loc[table["precision"] >= minimum_precision]
        if eligible.empty:
            raise ValueError("No validation threshold satisfies the configured precision guardrail.")
        selected = eligible.sort_values(["estimated_cost", "f2"], ascending=[True, False]).iloc[0]
        f2_best = table.sort_values("f2", ascending=False).iloc[0]
        return float(selected["threshold"]), {
            "validation_cost": float(selected["estimated_cost"]),
            "validation_f2": float(selected["f2"]),
            "validation_precision": float(selected["precision"]),
            "validation_recall": float(selected["recall"]),
            "f2_optimal_threshold": float(f2_best["threshold"]),
        }

    def evaluate_model(self, model: Any, threshold: float, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
        """Evaluate once on the untouched test set, including calibration and cost."""
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        default_predictions = (probabilities >= 0.50).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
        _, default_fp, default_fn, _ = confusion_matrix(y_test, default_predictions, labels=[0, 1]).ravel()
        settings = self.config["threshold"]
        fp_cost, fn_cost = float(settings["false_positive_cost"]), float(settings["false_negative_cost"])
        selected_cost = fp * fp_cost + fn * fn_cost
        default_cost = default_fp * fp_cost + default_fn * fn_cost
        all_show_cost = int(y_test.sum()) * fn_cost
        return {
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(y_test, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "specificity": float(tn / (tn + fp)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "f2": float(fbeta_score(y_test, predictions, beta=2, zero_division=0)),
            "matthews_correlation": float(matthews_corrcoef(y_test, predictions)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "average_precision": float(average_precision_score(y_test, probabilities)),
            "brier_score": float(brier_score_loss(y_test, probabilities)),
            "log_loss": float(log_loss(y_test, probabilities)),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
            "estimated_cost": float(selected_cost),
            "estimated_cost_per_booking": float(selected_cost / len(y_test)),
            "default_threshold_cost": float(default_cost),
            "all_show_baseline_cost": float(all_show_cost),
            "cost_reduction_vs_default": float(default_cost - selected_cost),
            "cost_reduction_vs_all_show": float(all_show_cost - selected_cost),
        }

    def run(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
        """Run comparison, tuning, calibration, threshold selection, and final test."""
        comparison = self.compare_models(X_train, y_train)
        predictive = comparison.loc[comparison["model"] != "Dummy Prior"]
        selected_name = str(predictive.iloc[0]["model"])
        X_fit, X_validation, y_fit, y_validation = train_test_split(
            X_train,
            y_train,
            test_size=float(self.config["validation_size"]),
            random_state=self.random_state,
            stratify=y_train,
        )
        search = self.tune_selected_model(selected_name, X_fit, y_fit)
        calibration_model = CalibratedClassifierCV(
            estimator=clone(search.best_estimator_),
            method=self.config["calibration_method"],
            cv=int(self.config["calibration_cv_folds"]),
        )
        calibration_model.fit(X_fit, y_fit)
        validation_probabilities = calibration_model.predict_proba(X_validation)[:, 1]
        thresholds = self.threshold_table(y_validation, validation_probabilities)
        selected_threshold, threshold_summary = self.select_threshold(thresholds)

        importance = permutation_importance(
            calibration_model,
            X_validation,
            y_validation,
            scoring="average_precision",
            n_repeats=2,
            random_state=self.random_state,
            n_jobs=1,
        )
        feature_importance = pd.DataFrame({
            "feature": X_validation.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }).sort_values("importance_mean", ascending=False)

        final_model = CalibratedClassifierCV(
            estimator=clone(search.best_estimator_),
            method=self.config["calibration_method"],
            cv=int(self.config["calibration_cv_folds"]),
        )
        final_model.fit(X_train, y_train)
        results = self.evaluate_model(final_model, selected_threshold, X_test, y_test)
        probabilities = final_model.predict_proba(X_test)[:, 1]
        observed, predicted = calibration_curve(y_test, probabilities, n_bins=10, strategy="quantile")
        calibration = pd.DataFrame({"mean_predicted_probability": predicted, "observed_no_show_rate": observed})
        results.update({
            "selected_model": selected_name,
            "selection_metric": self.config["primary_metric"],
            "best_tuning_score": float(search.best_score_),
            "best_parameters": search.best_params_,
            "calibration_method": self.config["calibration_method"],
            "threshold_selection_source": "stratified validation partition from training data",
            "business_cost_ratio_fn_to_fp": float(self.config["threshold"]["false_negative_cost"] / self.config["threshold"]["false_positive_cost"]),
            **threshold_summary,
        })
        return comparison, final_model, results, thresholds, feature_importance, calibration
