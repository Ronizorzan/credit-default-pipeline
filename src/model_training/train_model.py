import json
import logging
import os
import yaml

import joblib
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
    return params["train"]


def prepare_data(train_data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Prepare data for XGBClassifier by separating features and target into train and validation data.

    Args:
        train_data (pd.DataFrame): Full training dataset.

    Returns:
        tuple containing:
            pd.DataFrame: Training features
            np.ndarray: Encoded training labels            
    """
    # Separate features and target for train data
    X_train = train_data.drop("target", axis=1)
    y_train = train_data["target"]

    # Splits the data into train and validation data
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=3214)
    
    
    return X_train, y_train, X_val, y_val


def create_model(
        X_train: pd.DataFrame, y_train: np.array,
        X_val: pd.DataFrame, y_val: np.array,
    params: dict[str, int | float] ) -> XGBClassifier:
    """Create a XGBClassifier model.

    Args:    
        X_train: DataFrame with features columns.
        y_train: Target Column to be predicted.        
        params (dict[str, int | float]): Model hyperparameters.

    Returns:
        XFBClassifier: Trainned XGB Model.
    """
    model = XGBClassifier(**params).fit(X_train, y_train, eval_set=[(X_val, y_val)])
    y_pred = model.predict(X_val)
    report = classification_report(y_val, y_pred)
    logger.info(f"Classification report: \n{report}")

    return model


def save_training_artifacts(model: XGBClassifier) -> None:
    """Save model artifacts to disk.

    Args:
        model (XGBClassifier): Trained XGB model.                
    """
    artifacts_dir = "artifacts"
    models_dir = "models"
    model_path = os.path.join(models_dir, "xgb_model.joblib")    

    # Save the model
    logger.info(f"Saving model to {model_path}")
    joblib.dump(model, model_path)
    

def train_model(train_data: pd.DataFrame, params: dict[str, int | float]) -> None:
    """Train a Keras model, logging metrics and artifacts with MLflow.

    Args:
        train_data (pd.DataFrame): Training dataset.
        params (dict[str, int | float]): Model hyperparameters.
    """   
    # Prepare the data
    X_train, y_train, X_val, y_val = prepare_data(train_data)
    
    # Create the model
    model = create_model(
        X_train, y_train, 
        X_val, y_val, params=params
    )
    

    save_training_artifacts(model)
     

def main() -> None:
    """Main function to orchestrate the model training process."""
    train_data = load_data()
    params = load_params()
    train_model(train_data, params)
    logger.info("Model training completed")


if __name__ == "__main__":
    main()