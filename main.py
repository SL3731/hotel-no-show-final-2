"""Command-line entry point for training and batch prediction."""
from __future__ import annotations

import argparse
import json
import logging
import platform
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml

from src.data_preparation import DataPreparation
from src.model_training import ModelTraining

ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML and fail early on missing, mistyped, or invalid settings."""
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping.")
    required = {
        "project_name", "data_path", "target_column", "random_state", "test_size",
        "validation_size", "required_columns", "identifier_columns", "categorical_features",
        "numeric_features", "currency_to_sgd", "cv_folds", "tuning_cv_folds",
        "tuning_iterations", "primary_metric", "calibration_method", "calibration_cv_folds",
        "threshold", "hyperparameters", "artifact_directory",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Configuration is missing required keys: {missing}")
    for key in ["test_size", "validation_size"]:
        if not 0 < float(config[key]) < 1:
            raise ValueError(f"{key} must be between 0 and 1.")
    for key in ["cv_folds", "tuning_cv_folds", "calibration_cv_folds"]:
        if int(config[key]) < 2:
            raise ValueError(f"{key} must be at least 2.")
    threshold = config["threshold"]
    required_threshold = {
        "minimum", "maximum", "step", "minimum_precision",
        "false_negative_cost", "false_positive_cost", "beta",
    }
    if required_threshold - set(threshold):
        raise ValueError(f"Threshold configuration is incomplete: {sorted(required_threshold - set(threshold))}")
    if not 0 <= float(threshold["minimum"]) < float(threshold["maximum"]) <= 1:
        raise ValueError("Threshold minimum and maximum must satisfy 0 <= minimum < maximum <= 1.")
    if float(threshold["step"]) <= 0 or float(threshold["false_negative_cost"]) <= 0 or float(threshold["false_positive_cost"]) <= 0:
        raise ValueError("Threshold step and business costs must be positive.")
    return config


def configure_logging(log_path: Path) -> None:
    """Log to both the console and a persistent run file."""
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)


def make_run_directory(config: dict[str, Any]) -> Path:
    """Create a timestamped directory so experiments never overwrite each other."""
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_directory = ROOT / config["artifact_directory"] / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def train(config_path: Path) -> Path:
    """Train the full workflow and save an auditable, versioned run."""
    config = load_config(config_path)
    run_directory = make_run_directory(config)
    configure_logging(run_directory / "pipeline.log")
    seed = int(config["random_state"])
    random.seed(seed)
    np.random.seed(seed)
    LOGGER.info("Starting %s", config["project_name"])
    LOGGER.info("Seed=%d; Python=%s; pandas=%s; scikit-learn=%s", seed, platform.python_version(), pd.__version__, sklearn.__version__)

    preparation = DataPreparation(config, ROOT)
    split, preprocessor = preparation.run()
    comparison, model, results, thresholds, importance, calibration = ModelTraining(config, preprocessor).run(
        split.X_train, split.y_train, split.X_test, split.y_test
    )

    comparison.to_csv(run_directory / "model_comparison.csv", index=False)
    thresholds.to_csv(run_directory / "threshold_analysis.csv", index=False)
    importance.to_csv(run_directory / "feature_importance.csv", index=False)
    calibration.to_csv(run_directory / "calibration_curve.csv", index=False)
    with (run_directory / "final_test_results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    with (run_directory / "config_snapshot.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "random_state": seed,
        "train_rows": len(split.X_train),
        "test_rows": len(split.X_test),
        "train_no_show_rate": float(split.y_train.mean()),
        "test_no_show_rate": float(split.y_test.mean()),
    }
    with (run_directory / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    joblib.dump(
        {"model": model, "threshold": results["threshold"], "config": config},
        run_directory / "model_bundle.joblib",
    )
    latest_path = ROOT / config["artifact_directory"] / "latest_run.txt"
    latest_path.write_text(run_directory.name, encoding="utf-8")

    print("\nCross-validation comparison")
    print(comparison.to_string(index=False))
    print("\nFinal untouched-test results")
    print(json.dumps(results, indent=2))
    print(f"\nArtifacts saved to: {run_directory}")
    return run_directory


def predict(config_path: Path, model_path: Path, input_path: Path, output_path: Path) -> None:
    """Score a batch of raw bookings with a saved model and selected threshold."""
    config = load_config(config_path)
    bundle = joblib.load(model_path)
    model_config = bundle.get("config", config)
    preparation = DataPreparation(model_config, ROOT)
    raw = preparation.load_data(input_path)
    booking_ids = raw["booking_id"].copy()
    prepared = preparation.prepare_features(raw, require_target=False)
    probabilities = bundle["model"].predict_proba(prepared)[:, 1]
    output = pd.DataFrame({
        "booking_id": booking_ids.to_numpy(),
        "no_show_probability": probabilities,
        "predicted_no_show": (probabilities >= float(bundle["threshold"])).astype(int),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse train or predict commands while keeping ``python main.py`` valid."""
    parser = argparse.ArgumentParser(description="Hotel no-show modelling pipeline")
    parser.add_argument("command", nargs="?", choices=["train", "predict"], default="train")
    parser.add_argument("--config", type=Path, default=ROOT / "src" / "config.yaml")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "predictions.csv")
    return parser.parse_args()


def main() -> None:
    """Dispatch the requested workflow and log actionable failures."""
    args = parse_args()
    try:
        if args.command == "train":
            train(args.config)
        else:
            if args.model is None or args.input is None:
                raise ValueError("predict requires both --model and --input paths.")
            predict(args.config, args.model, args.input, args.output)
    except (FileNotFoundError, ValueError, KeyError) as error:
        logging.error("Pipeline input/configuration error: %s", error)
        raise SystemExit(2) from error
    except Exception:
        logging.exception("Unexpected pipeline failure")
        raise


if __name__ == "__main__":
    main()
