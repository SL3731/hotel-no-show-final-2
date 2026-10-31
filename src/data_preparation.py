"""Validated cleaning, feature engineering, splitting, and preprocessing."""
from __future__ import annotations

import calendar
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)
MONTH_NUMBER = {month: index for index, month in enumerate(calendar.month_name) if month}
MONTH_START = dict(zip(calendar.month_name[1:], np.cumsum([0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30])))


@dataclass(frozen=True)
class DataSplit:
    """Immutable train/test data container."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


class DataPreparation:
    """Validate raw booking records and build leakage-safe model inputs."""

    def __init__(self, config: dict[str, Any], project_root: Path | None = None):
        self.config = config
        self.project_root = project_root or Path(__file__).resolve().parents[1]
        self.target = config["target_column"]

    def load_data(self, path: Path | None = None) -> pd.DataFrame:
        """Load a CSV and report its dimensions; raise a clear path/read error."""
        data_path = path or self.project_root / self.config["data_path"]
        if not data_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        try:
            dataframe = pd.read_csv(data_path)
        except (pd.errors.ParserError, UnicodeDecodeError) as error:
            raise ValueError(f"Dataset could not be parsed as CSV: {data_path}") from error
        LOGGER.info("Loaded %s: %d rows x %d columns", data_path, *dataframe.shape)
        return dataframe

    def validate_schema(self, dataframe: pd.DataFrame, require_target: bool = True) -> None:
        """Check required columns and the binary target before transformation."""
        if dataframe.empty:
            raise ValueError("Dataset is empty.")
        required = set(self.config["required_columns"])
        if not require_target:
            required.discard(self.target)
        missing = sorted(required - set(dataframe.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if require_target:
            target = pd.to_numeric(dataframe[self.target], errors="coerce")
            invalid = target.notna() & ~target.isin([0, 1])
            if invalid.any():
                values = sorted(target.loc[invalid].unique().tolist())
                raise ValueError(f"Target must contain only 0, 1, or missing; found {values}")

    def clean_data(self, dataframe: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
        """Repair documented formatting issues without learning from the data."""
        self.validate_schema(dataframe, require_target=require_target)
        cleaned = dataframe.copy()
        if require_target:
            cleaned = cleaned.drop_duplicates()
            cleaned[self.target] = pd.to_numeric(cleaned[self.target], errors="coerce")
            missing_targets = int(cleaned[self.target].isna().sum())
            if missing_targets:
                LOGGER.warning("Dropping %d rows with missing target", missing_targets)
            cleaned = cleaned.loc[cleaned[self.target].isin([0, 1])].copy()
            cleaned[self.target] = cleaned[self.target].astype(int)

        text_columns = ["branch", "booking_month", "arrival_month", "checkout_month", "country", "first_time", "room", "platform"]
        for column in text_columns:
            values = cleaned[column].astype("string").str.strip().str.title().replace("", pd.NA)
            cleaned[column] = values.astype(object).where(values.notna(), np.nan)

        adult_text = cleaned["num_adults"].astype("string").str.strip().str.lower()
        cleaned["num_adults"] = pd.to_numeric(adult_text.replace({"one": "1", "two": "2"}), errors="coerce")
        for column in ["num_children", "arrival_day", "checkout_day"]:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        negative_days = int((cleaned["checkout_day"] < 0).sum())
        if negative_days:
            LOGGER.warning("Correcting the sign of %d negative checkout days", negative_days)
        cleaned["checkout_day"] = cleaned["checkout_day"].abs()

        price_text = cleaned["price"].astype("string").str.strip()
        currency = price_text.str.extract(r"^([A-Za-z]{3})\$")[0].str.upper()
        cleaned["price_currency"] = currency.astype(object).where(currency.notna(), np.nan)
        cleaned["price_amount"] = pd.to_numeric(price_text.str.replace(r"^[A-Za-z]{3}\$\s*", "", regex=True), errors="coerce")
        cleaned["price_sgd"] = cleaned["price_amount"] * cleaned["price_currency"].map(self.config["currency_to_sgd"])
        cleaned = cleaned.drop(columns=["price", "price_amount"])
        LOGGER.info("Cleaned dataset: %d rows x %d columns", *cleaned.shape)
        return cleaned

    def engineer_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Create temporal, party-size, price, and subgroup interaction features."""
        engineered = dataframe.copy()
        booking_number = engineered["booking_month"].map(MONTH_NUMBER)
        arrival_number = engineered["arrival_month"].map(MONTH_NUMBER)
        engineered["booking_lead_months"] = (arrival_number - booking_number) % 12
        arrival_ordinal = engineered["arrival_month"].map(MONTH_START) + engineered["arrival_day"]
        checkout_ordinal = engineered["checkout_month"].map(MONTH_START) + engineered["checkout_day"]
        engineered["length_of_stay_days"] = (checkout_ordinal - arrival_ordinal) % 366
        engineered.loc[engineered["length_of_stay_days"] > 31, "length_of_stay_days"] = np.nan
        engineered["total_guests"] = engineered["num_adults"] + engineered["num_children"]
        engineered["price_per_guest_sgd"] = engineered["price_sgd"].div(engineered["total_guests"].replace(0, np.nan))

        for prefix, month_number in [("arrival", arrival_number), ("booking", booking_number)]:
            angle = 2 * np.pi * (month_number - 1) / 12
            engineered[f"{prefix}_month_sin"] = np.sin(angle)
            engineered[f"{prefix}_month_cos"] = np.cos(angle)

        def combine(left: str, right: str) -> pd.Series:
            return engineered[left].fillna("Missing") + "__" + engineered[right].fillna("Missing")

        engineered["branch_country"] = combine("branch", "country")
        engineered["branch_first_time"] = combine("branch", "first_time")
        engineered["country_first_time"] = combine("country", "first_time")
        return engineered.drop(columns=self.config["identifier_columns"], errors="ignore")

    def prepare_features(self, dataframe: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
        """Apply deterministic cleaning and engineering for training or inference."""
        prepared = self.engineer_features(self.clean_data(dataframe, require_target=require_target))
        expected = set(self.config["numeric_features"] + self.config["categorical_features"])
        missing = sorted(expected - set(prepared.columns))
        if missing:
            raise ValueError(f"Engineered feature columns are missing: {missing}")
        return prepared

    def split_data(self, dataframe: pd.DataFrame) -> DataSplit:
        """Create a reproducible stratified holdout that is never used for selection."""
        X = dataframe.drop(columns=[self.target])
        y = dataframe[self.target]
        forbidden = set(self.config["identifier_columns"] + [self.target])
        overlap = forbidden.intersection(X.columns)
        if overlap:
            raise AssertionError(f"Leakage columns remain in model inputs: {sorted(overlap)}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=float(self.config["test_size"]), random_state=int(self.config["random_state"]), stratify=y
        )
        LOGGER.info("Split: %d train, %d test; no-show rates %.3f / %.3f", len(X_train), len(X_test), y_train.mean(), y_test.mean())
        return DataSplit(X_train, X_test, y_train, y_test)

    def create_preprocessor(self) -> ColumnTransformer:
        """Return an unfitted transformer; fitting occurs inside each model fold."""
        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ])
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        return ColumnTransformer([
            ("numerical", numerical_pipeline, self.config["numeric_features"]),
            ("categorical", categorical_pipeline, self.config["categorical_features"]),
        ], verbose_feature_names_out=False)

    def run(self) -> tuple[DataSplit, ColumnTransformer]:
        """Load, validate, prepare, split, and return an unfitted preprocessor."""
        dataframe = self.prepare_features(self.load_data(), require_target=True)
        return self.split_data(dataframe), self.create_preprocessor()
