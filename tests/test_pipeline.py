"""Focused regression tests for preparation, imbalance handling, and thresholding."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier

from main import load_config
from src.data_preparation import DataPreparation
from src.model_training import ModelTraining, RandomOversamplingClassifier

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def config() -> dict:
    """Load the production configuration used by the pipeline."""
    return yaml.safe_load((ROOT / "src" / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture()
def raw_data() -> pd.DataFrame:
    """Create valid records containing the important cleaning edge cases."""
    rows = []
    for index in range(40):
        rows.append({
            "booking_id": index + 1,
            "no_show": index % 2,
            "branch": " changi " if index % 2 else "Orchard",
            "booking_month": "january",
            "arrival_month": "March",
            "arrival_day": 10,
            "checkout_month": "March",
            "checkout_day": -12 if index == 0 else 12,
            "country": "Singapore" if index % 3 else "China",
            "first_time": "Yes" if index % 4 else "No",
            "room": None if index == 0 else "King",
            "price": None if index == 1 else "USD$ 100.00",
            "platform": "Website",
            "num_adults": " one " if index == 0 else "2",
            "num_children": index % 2,
        })
    return pd.DataFrame(rows)


def test_config_validation_rejects_invalid_split(tmp_path: Path, config: dict) -> None:
    """Invalid configuration values should fail before training starts."""
    config["test_size"] = 1.5
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="test_size"):
        load_config(path)


def test_schema_validation_reports_missing_column(config: dict, raw_data: pd.DataFrame) -> None:
    """A missing required feature should produce a targeted error."""
    preparation = DataPreparation(config, ROOT)
    with pytest.raises(ValueError, match="branch"):
        preparation.validate_schema(raw_data.drop(columns="branch"))


def test_cleaning_and_feature_engineering(config: dict, raw_data: pd.DataFrame) -> None:
    """Known data defects and EDA-derived features should be handled consistently."""
    preparation = DataPreparation(config, ROOT)
    prepared = preparation.prepare_features(raw_data)
    assert prepared.loc[0, "checkout_day"] == 12
    assert prepared.loc[0, "num_adults"] == 1
    assert prepared.loc[0, "arrival_month"] == "March"
    assert prepared.loc[2, "price_sgd"] == pytest.approx(135.0)
    assert prepared.loc[2, "booking_lead_months"] == 2
    assert prepared.loc[2, "length_of_stay_days"] == 2
    assert prepared.loc[2, "branch_country"] == "Orchard__Singapore"
    assert "booking_id" not in prepared.columns
    preprocessor = preparation.create_preprocessor()
    transformed = preprocessor.fit_transform(prepared.drop(columns="no_show"), prepared["no_show"])
    assert transformed.shape[0] == len(prepared)


def test_split_is_stratified_reproducible_and_leakage_free(config: dict, raw_data: pd.DataFrame) -> None:
    """Repeated splits must match and preserve both class prevalence and boundaries."""
    preparation = DataPreparation(config, ROOT)
    prepared = preparation.prepare_features(raw_data)
    first = preparation.split_data(prepared)
    second = preparation.split_data(prepared)
    assert first.X_train.index.tolist() == second.X_train.index.tolist()
    assert first.y_train.mean() == pytest.approx(first.y_test.mean())
    assert "no_show" not in first.X_train.columns
    assert "booking_id" not in first.X_train.columns


def test_random_oversampler_classifier_predicts_probabilities() -> None:
    """The in-fold oversampling wrapper should expose a normal classifier interface."""
    X = np.arange(60, dtype=float).reshape(30, 2)
    y = np.array([0] * 24 + [1] * 6)
    classifier = RandomOversamplingClassifier(
        HistGradientBoostingClassifier(max_iter=20, random_state=42), random_state=42
    ).fit(X, y)
    probabilities = classifier.predict_proba(X)
    assert probabilities.shape == (30, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_cost_aware_threshold_obeys_precision_guardrail(config: dict) -> None:
    """Threshold selection should use configured costs without violating precision."""
    trainer = ModelTraining(config, preprocessor=None)
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    probabilities = np.array([0.05, 0.15, 0.35, 0.60, 0.30, 0.55, 0.75, 0.95])
    table = trainer.threshold_table(y, probabilities)
    threshold, _ = trainer.select_threshold(table)
    selected = table.loc[np.isclose(table["threshold"], threshold)].iloc[0]
    assert selected["precision"] >= config["threshold"]["minimum_precision"]
    assert threshold != pytest.approx(0.50)
