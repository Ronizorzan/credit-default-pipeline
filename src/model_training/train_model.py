import logging
import os
import yaml
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

logger = logging.getLogger("src.model_training.train_model")


def load_data() -> pd.DataFrame:
    """Load the feature-engineered training data.

    Returns:
        pd.DataFrame: A dataframe containing the training data.
    """
    train_path = "data/processed/train_processed.csv"
    logger.info(f"Loading feature data from {train_path}")
    train_data = pd.read_csv(train_path)
    return train_data


def load_params() -> dict[str, float | int]:
    """Load model hyperparameters for the train stage from params.yaml.

    Returns:
        dict[str, int | float]: dictionary containing model hyperparameters.
    """
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["train"]


def prepare_data(train_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare data for XGBClassifier by separating features and target data.

    Args:
        train_data (pd.DataFrame): Full training dataset.

    Returns:
        tuple containing:
            pd.DataFrame: Training features
            pd.Series: Training labels           
    """
    X = train_data.drop("target", axis=1)
    y = train_data["target"]
    
    return X, y


def create_model(
    X: pd.DataFrame, y: pd.Series,    
    params: dict[str, int | float]
) -> XGBClassifier:
    """Create and train an XGBClassifier model.

    Args:
        X (pd.DataFrame): Training features
        y (pd.Series): Training labels        
        params (dict): Model hyperparameters

    Returns:
        XGBClassifier: Trained XGB model
    """
    model = XGBClassifier(**params)
    model.fit(X, y, verbose=False)

    return model


def save_training_artifacts(model: XGBClassifier) -> None:
    """Save model artifacts to disk.

    Args:
        model (XGBClassifier): Trained XGB model
    """
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "xgb_model.joblib")
    logger.info(f"Saving model to {model_path}")
    joblib.dump(model, model_path)


def train_model(train_data: pd.DataFrame, params: dict[str, int | float]) -> None:
    """Train an XGBClassifier model, logging metrics and artifacts.

    Args:
        train_data (pd.DataFrame): Training dataset
        params (dict[str, int | float]): Model hyperparameters
    """
    X_train, y_train= prepare_data(train_data)
    model = create_model(X_train, y_train, params=params)
    save_training_artifacts(model)


def main() -> None:
    """Main function to orchestrate the model training process."""
    train_data = load_data()
    params = load_params()
    train_model(train_data, params)
    logger.info("Model training completed")


if __name__ == "__main__":
    main()