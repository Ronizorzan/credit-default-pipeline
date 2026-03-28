import logging
import os
import yaml
import joblib
import json

import mlflow
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

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
    return params


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
    """Split data into train and evaluation sets.
    Create and train an XGBClassifier model.

    Args:
        X (pd.DataFrame): Training features
        y (pd.Series): Training labels        
        params (dict): Model hyperparameters

    Returns:
        XGBClassifier: Trained XGB model
    """
    logger.info("Splitting data and training model")
    X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                      test_size=params["val_split"]["test_size"], random_state=params["val_split"]["random_state"])
        

    model = XGBClassifier(**params["train"])
    model.fit(X_train, y_train, verbose=False, eval_set=[(X_val, y_val)])

    y_pred = model.predict(X_val)
    report = classification_report(y_val, y_pred, output_dict=True)

    os.makedirs("metrics", exist_ok=True)
    metrics_path = "metrics/training.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2 )

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


def train_model(train_data: pd.DataFrame, params:dict[str, dict[str, int | float]]) -> None:
    """Train a XGBClassifier model, logging metrics and artifacts.

    Args:
        train_data (pd.DataFrame): Training dataset
        params (dict[str, int | float]): Model hyperparameters
    """
    
    # Set up MLflow experiment
    mlflow.set_experiment("credit_card_experiment")

    # Set up XGBoost Autolog
    mlflow.xgboost.autolog()

    # Setting MLflow if we are running a DVC experiment
    is_experiment = os.getenv("DVC_EXP_NAME") is not None
    extra_args = {}

    if is_experiment:
        runs = mlflow.search_runs(
            experiment_ids=[os.getenv("MLFLOW_EXPERIMENT_ID")],
            filter_string="tags.dvc_exp = 'True'",
            order_by=["start_time DESC"]
        )
        
        if runs.empty:
            with mlflow.start_run() as parent_run:
                mlflow.set_tag("dvc_exp", True)
                parent_run_id = parent_run.info.run_id
            
        else:
            parent_run_id = runs.iloc[0].run_id        
        run_name = os.getenv("DVC_EXP_NAME")
        extra_args = {
            "parent_run_id": parent_run_id,
            "run_name": run_name,
            "nested": True}


    with mlflow.start_run(**extra_args):
        
        # Log parameters to MlFlow
        mlflow.log_params(params)

        X_train, y_train= prepare_data(train_data)
        model = create_model(X_train, y_train, params=params)
        save_training_artifacts(model)

        # Log preprocessing artifacts
        mlflow.log_artifact("artifacts/balance_discretizer.joblib")
        mlflow.log_artifact("artifacts/feature_selector.joblib")
        mlflow.log_artifact("artifacts/preprocessor.joblib")
        mlflow.log_artifact("artifacts/target_encoder.joblib")

        # Log Mlflow Model with consistent name
        mlflow.xgboost.log_model(model, artifact_path="xgb_model")


def main() -> None:
    """Main function to orchestrate the model training process."""
    train_data = load_data()
    params = load_params()
    train_model(train_data, params)
    logger.info("Model training completed")


if __name__ == "__main__":
    main()