import logging
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

logger = logging.getLogger("src.model_evaluation.evaluate_model")


def load_model() -> XGBClassifier:
    """Load the trained XGBClassifier model from disk.

    Returns:
        XGBClassifier: Loaded XGBClassifier model.
    """
    model_path = "models/xgb_model.joblib"
    model = joblib.load(model_path)
    return model


def load_imputer() -> ColumnTransformer:
    """Load the Column Transformer from disk.

    Returns:
        ColumnTransformer: Loaded Column Transformer.
    """
    preprocessor_path = "artifacts/preprocessor.joblib"
    preprocessor = joblib.load(preprocessor_path)
    return preprocessor


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the test dataset from disk.

    Returns:
        tuple containing:
            pd.DataFrame: Test features
            pd.Series: Test labels
    """
    data_path = "data/processed/test_processed.csv"
    logger.info(f"Loading test data from {data_path}")
    data = pd.read_csv(data_path)
    X = data.drop("target", axis=1)
    y = data["target"]
    return X, y


def evaluate_model(
    model: XGBClassifier, encoder: ColumnTransformer, X: pd.DataFrame, y_true: pd.Series
) -> None:
    """Evaluate the model and generate performance metrics.

    Args:
        model (XGBClassifier): Trained XGBClassifier model.
        Imputer (ColumnTransformer): Fitted Column Transformer.
        X (pd.DataFrame): Test features.
        y_true (pd.Series): True labels.
    """
    # Generate model predictions
    y_pred = model.predict(X)    

    # Calculate evaluation metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    evaluation = {"classification_report": report, "confusion_matrix": cm.tolist()}

    # Log metrics
    os.makedirs("metrics", exist_ok=True)
    logger.info(f"Classification Report:\n{classification_report(y_true, y_pred)}\nConfusion_matrix:\n{cm}")    
    evaluation_path = "metrics/evaluation.json"
    with open(evaluation_path, "w") as f:
        json.dump(evaluation, f, indent=2)


def main() -> None:
    """Main function to orchestrate the model evaluation process."""
    model = load_model()
    encoder = load_imputer()
    X, y = load_test_data()
    evaluate_model(model, encoder, X, y)
    logger.info("Model evaluation completed")


if __name__ == "__main__":
    main()