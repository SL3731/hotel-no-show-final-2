"""Run the complete hotel no-show classification pipeline."""
import json
import logging
from pathlib import Path

import joblib
import yaml

from src.data_preparation import DataPreparation
from src.model_training import ModelTraining

ROOT = Path(__file__).resolve().parent


def load_config() -> dict:
    with (ROOT / "src" / "config.yaml").open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    config = load_config()
    logging.info("Starting %s", config["project_name"])
    split, preprocessor = DataPreparation(config, ROOT).run()
    comparison, model, results = ModelTraining(config, preprocessor).run(
        split.X_train, split.y_train, split.X_test, split.y_test
    )

    comparison.to_csv(ROOT / config["model_comparison_path"], index=False)
    with (ROOT / config["final_test_results_path"]).open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    joblib.dump(model, ROOT / config["model_artifact_path"])

    print("\nCross-validation comparison")
    print(comparison.to_string(index=False))
    print("\nFinal untouched-test results")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

