"""Classification model comparison, tuning, threshold selection, and evaluation."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    fbeta_score, make_scorer, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline

LOGGER = logging.getLogger(__name__)


class ModelTraining:
    """Compare three predictive classifiers plus a naive benchmark."""

    def __init__(self, config: dict[str, Any], preprocessor: Any):
        self.config = config
        self.preprocessor = preprocessor
        self.random_state = int(config["random_state"])

    def _models(self) -> dict[str, Any]:
        return {
            "Dummy Prior": DummyClassifier(strategy="prior"),
            "Logistic Regression": LogisticRegression(
                max_iter=2_000, class_weight="balanced", random_state=self.random_state
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=150, min_samples_leaf=2, max_features="sqrt",
                class_weight="balanced_subsample", random_state=self.random_state, n_jobs=1,
            ),
            "Histogram Gradient Boosting": HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.08, l2_regularization=1.0,
                class_weight="balanced", random_state=self.random_state,
            ),
        }

    def _pipeline(self, model: Any) -> Pipeline:
        return Pipeline([
            ("preprocessor", clone(self.preprocessor)),
            ("model", model),
        ])

    def compare_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        cv = StratifiedKFold(
            n_splits=int(self.config["cv_folds"]), shuffle=True, random_state=self.random_state
        )
        scoring = {
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "f2": make_scorer(fbeta_score, beta=2),
            "roc_auc": "roc_auc",
            "average_precision": "average_precision",
        }
        rows = []
        for name, model in self._models().items():
            LOGGER.info("Cross-validating %s", name)
            scores = cross_validate(
                self._pipeline(model), X_train, y_train,
                cv=cv, scoring=scoring, n_jobs=1,
            )
            row = {"model": name}
            for metric in scoring:
                row[f"cv_{metric}_mean"] = scores[f"test_{metric}"].mean()
                row[f"cv_{metric}_std"] = scores[f"test_{metric}"].std()
            rows.append(row)
        return pd.DataFrame(rows).sort_values(
            ["cv_f2_mean", "cv_recall_mean"], ascending=False
        ).reset_index(drop=True)

    def tune_selected_model(self, selected_name: str, X_train, y_train) -> RandomizedSearchCV:
        spaces = {
            "Logistic Regression": {
                "model__C": np.logspace(-2, 2, 20),
                "model__solver": ["lbfgs", "liblinear"],
            },
            "Random Forest": {
                "model__n_estimators": [100, 150, 250],
                "model__max_depth": [10, 18, 26, None],
                "model__min_samples_leaf": [1, 2, 5, 10],
                "model__max_features": ["sqrt", 0.6, 1.0],
            },
            "Histogram Gradient Boosting": {
                "model__learning_rate": [0.03, 0.05, 0.08, 0.12],
                "model__max_iter": [150, 220, 300],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__min_samples_leaf": [15, 30, 60],
                "model__l2_regularization": [0.1, 0.5, 1.0, 3.0],
            },
        }
        if selected_name not in spaces:
            raise ValueError("Dummy model cannot be selected for tuning.")
        cv = StratifiedKFold(
            n_splits=int(self.config["tuning_cv_folds"]), shuffle=True,
            random_state=self.random_state,
        )
        search = RandomizedSearchCV(
            self._pipeline(self._models()[selected_name]),
            spaces[selected_name],
            n_iter=int(self.config["tuning_iterations"]),
            scoring=make_scorer(fbeta_score, beta=2),
            cv=cv, random_state=self.random_state, n_jobs=1, refit=True,
        )
        LOGGER.info("Tuning selected model: %s", selected_name)
        search.fit(X_train, y_train)
        return search

    def select_threshold(self, model: Pipeline, X_train, y_train) -> tuple[float, float]:
        """Choose the probability threshold using out-of-fold training predictions."""
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        probabilities = cross_val_predict(
            clone(model), X_train, y_train, cv=cv, method="predict_proba", n_jobs=1
        )[:, 1]
        beta = float(self.config["threshold_metric_beta"])
        thresholds = np.arange(0.20, 0.81, 0.01)
        precision_values = np.array([
            precision_score(y_train, probabilities >= threshold, zero_division=0)
            for threshold in thresholds
        ])
        scores = np.array([
            fbeta_score(y_train, probabilities >= threshold, beta=beta, zero_division=0)
            for threshold in thresholds
        ])
        eligible = precision_values >= float(self.config["threshold_min_precision"])
        if not eligible.any():
            raise ValueError("No threshold satisfies the configured minimum precision.")
        eligible_indices = np.flatnonzero(eligible)
        index = int(eligible_indices[np.argmax(scores[eligible])])
        return float(thresholds[index]), float(scores[index])

    @staticmethod
    def evaluate_model(model: Pipeline, threshold: float, X_test, y_test) -> dict[str, Any]:
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        return {
            "threshold": threshold,
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "specificity": float(tn / (tn + fp)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "f2": float(fbeta_score(y_test, predictions, beta=2, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "average_precision": float(average_precision_score(y_test, probabilities)),
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
        }

    def run(self, X_train, y_train, X_test, y_test):
        comparison = self.compare_models(X_train, y_train)
        predictive = comparison.loc[comparison["model"] != "Dummy Prior"]
        selected_name = predictive.iloc[0]["model"]
        search = self.tune_selected_model(selected_name, X_train, y_train)
        best_model = search.best_estimator_
        threshold, oof_f2 = self.select_threshold(best_model, X_train, y_train)
        results = self.evaluate_model(best_model, threshold, X_test, y_test)
        results.update({
            "selected_model": selected_name,
            "best_cv_f2": float(search.best_score_),
            "training_oof_f2_at_threshold": oof_f2,
            "best_parameters": search.best_params_,
        })
        return comparison, best_model, results
