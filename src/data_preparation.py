"""Data validation, cleaning, feature engineering, splitting, and preprocessing."""
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
MONTH_START = dict(zip(calendar.month_name[1:], np.cumsum([0] + [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30])[:-1]))


@dataclass(frozen=True)
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


class DataPreparation:
    """Prepare hotel booking records using the central configuration."""

    def __init__(self, config: dict[str, Any], project_root: Path | None = None):
        self.config = config
        self.project_root = project_root or Path(__file__).resolve().parents[1]
        self.target = config["target_column"]

    def load_data(self) -> pd.DataFrame:
        path = self.project_root / self.config["data_path"]
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        dataframe = pd.read_csv(path)
        LOGGER.info("Loaded dataset: %d rows × %d columns", *dataframe.shape)
        return dataframe

    def validate_schema(self, dataframe: pd.DataFrame) -> None:
        if dataframe.empty:
            raise ValueError("Dataset is empty.")
        missing = sorted(set(self.config["required_columns"]) - set(dataframe.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def clean_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Repair documented formatting issues without fitting learned values."""
        self.validate_schema(dataframe)
        cleaned = dataframe.copy().drop_duplicates()
        cleaned[self.target] = pd.to_numeric(cleaned[self.target], errors="coerce")
        cleaned = cleaned.loc[cleaned[self.target].isin([0, 1])].copy()
        cleaned[self.target] = cleaned[self.target].astype(int)

        text_columns = [
            "branch", "booking_month", "arrival_month", "checkout_month",
            "country", "first_time", "room", "platform",
        ]
        for column in text_columns:
            cleaned[column] = cleaned[column].astype(object).str.strip().str.title()
            cleaned[column] = cleaned[column].where(cleaned[column].notna(), np.nan)

        cleaned["num_adults"] = cleaned["num_adults"].replace({"one": 1, "two": 2})
        cleaned["num_adults"] = pd.to_numeric(cleaned["num_adults"], errors="coerce")
        cleaned["num_children"] = pd.to_numeric(cleaned["num_children"], errors="coerce")
        cleaned["arrival_day"] = pd.to_numeric(cleaned["arrival_day"], errors="coerce")
        cleaned["checkout_day"] = pd.to_numeric(cleaned["checkout_day"], errors="coerce").abs()

        cleaned["price_currency"] = cleaned["price"].str.extract(r"^([A-Z]{3})\$")[0]
        cleaned["price_amount"] = pd.to_numeric(
            cleaned["price"].str.replace(r"^[A-Z]{3}\$\s*", "", regex=True),
            errors="coerce",
        )
        cleaned = cleaned.drop(columns=["price"])
        LOGGER.info("Cleaned dataset: %d rows × %d columns", *cleaned.shape)
        return cleaned

    def engineer_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        engineered = dataframe.copy()
        booking_number = engineered["booking_month"].map(MONTH_NUMBER)
        arrival_number = engineered["arrival_month"].map(MONTH_NUMBER)
        engineered["booking_lead_months"] = (arrival_number - booking_number) % 12

        arrival_ordinal = engineered["arrival_month"].map(MONTH_START) + engineered["arrival_day"]
        checkout_ordinal = engineered["checkout_month"].map(MONTH_START) + engineered["checkout_day"]
        engineered["length_of_stay_days"] = (checkout_ordinal - arrival_ordinal) % 366
        engineered.loc[engineered["length_of_stay_days"] > 31, "length_of_stay_days"] = np.nan
        engineered["total_guests"] = engineered["num_adults"] + engineered["num_children"]
        return engineered.drop(columns=self.config["identifier_columns"], errors="ignore")

    def split_data(self, dataframe: pd.DataFrame) -> DataSplit:
        X = dataframe.drop(columns=[self.target])
        y = dataframe[self.target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=float(self.config["test_size"]),
            random_state=int(self.config["random_state"]),
            stratify=y,
        )
        LOGGER.info("Split data: %d train rows, %d test rows", len(X_train), len(X_test))
        return DataSplit(X_train, X_test, y_train, y_test)

    def create_preprocessor(self) -> ColumnTransformer:
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
        dataframe = self.engineer_features(self.clean_data(self.load_data()))
        return self.split_data(dataframe), self.create_preprocessor()

