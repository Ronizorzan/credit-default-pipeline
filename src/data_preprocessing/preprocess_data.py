import logging
import os
import yaml
import joblib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

logger = logging.getLogger("src.data_preprocessing.preprocess_data")

def load_data() -> pd.DataFrame:
    """Load the raw data from disk.

    Returns:
        pd.DataFrame: Raw input data
    """
    input_path = "data/raw/raw.csv"
    logger.info(f"Loading raw data from {input_path}")
    data = pd.read_csv(input_path)
    return data


def load_params() -> dict[str, float | int]:
    """Load preprocessing parameters from params.yaml.

    Returns:
        dict[str, Any]: dictionary containing preprocessing parameters.
    """
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["preprocess_data"]

def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets using parameters from params.yaml.

    Args:
        data (pd.DataFrame): Input dataset

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Train and test datasets
    """
    params = load_params()
    logger.info("Splitting data into train and test sets...")
    train_data, test_data = train_test_split(
        data, test_size=params["test_size"], random_state=params["random_seed"]
    )
    return train_data, test_data

def preprocess_data( train_data: pd.DataFrame, test_data: pd.DataFrame ) -> tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """Perform preprocessing steps on train and test sets.
    Args:
        train_data (pd.DataFrame): Training dataset
        test_data (pd.DataFrame): Test dataset

    Returns:
        Tuple containing:
            pd.DataFrame: Processed training data
            pd.DataFrame: Processed test data
            Preprocessor: Fitted preprocessor for numerical and categorical columns
    """
    logger.info("Preprocessing data...")

    # Mapper for categorical columns
    mapper = {"Yes": 1, "No": 0}

    # Encode target column
    train_data["target"] = train_data["target"].map(mapper)
    test_data["target"] = test_data["target"].map(mapper)

    # Encode student column
    train_data["student"] = train_data["student"].map(mapper)
    test_data["student"] = test_data["student"].map(mapper)

    # Separates the features and target columns
    train_target = train_data['target']
    test_target = test_data['target']
    train_features = train_data.drop('target', axis=1)
    test_features = test_data.drop('target', axis=1)

    # Identifies numerical and categorical column types
    numerical_columns = train_features.select_dtypes(include="number").columns
    categorical_columns = train_features.select_dtypes(include="object").columns

    # Numerical Imputer
    numeric_transformer = Pipeline(
        steps=[(
            "imputer", SimpleImputer(strategy="median")
        )]
    )

    # Categorical Imputer
    categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
    )
        
    # Merge the two Pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_columns),
            ("cat", categorical_transformer, categorical_columns)
        ]
    )

    # Adjust on train and transform both types
    train_features_processed = preprocessor.fit_transform(train_features)
    test_features_processed = preprocessor.transform(test_features)

    train_processed = pd.DataFrame(train_features_processed.toarray() if hasattr(train_features_processed, "toarray") else train_features_processed, 
                                   columns=train_data.columns.drop("target"))
    test_processed = pd.DataFrame(test_features_processed.toarray() if hasattr(test_features_processed, "toarray") else test_features_processed, 
                                  columns=test_data.columns.drop("target"))
    
    # Merge target back with processed features
    train_processed = train_processed.assign(target=train_target.tolist())
    test_processed = test_processed.assign(target=test_target.tolist())

    return train_processed, test_processed, preprocessor


def save_artifacts( train_data: pd.DataFrame, test_data: pd.DataFrame, preprocessor: ColumnTransformer ) -> None:
    """Save processed data and preprocessing artifacts.

    Args:
        train_data (pd.DataFrame): Processed training data
        test_data (pd.DataFrame): Processed test data
        imputer (ColumnTransformer): Fitted imputer
    """
    # Save processed data
    data_dir = "data/preprocessed"
    logger.info(f"Saving processed data to {data_dir}")

    train_path = os.path.join(data_dir, "train_preprocessed.csv")
    test_path = os.path.join(data_dir, "test_preprocessed.csv")

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    # Save the full preprocessor (with imputer and encoder)
    preprocessor_path = os.path.join("artifacts", "preprocessor.joblib")
    logger.info(f"Saving imputer to {preprocessor_path}")
    joblib.dump(preprocessor, preprocessor_path)


def main() -> None:
     """Main function to orchestrate the preprocessing pipeline."""      
     raw_data = load_data()
     train_data, test_data = split_data(raw_data)
     train_processed, test_processed, imputer = preprocess_data(train_data, test_data)
     save_artifacts(train_processed, test_processed, imputer)
     logger.info("Data preprocessing completed")

if __name__ == "__main__":
    main()
