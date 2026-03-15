import logging
import os

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer


logger = logging.getLogger("src.feature_engineering.engineer_features")


def load_preprocessed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load preprocessed train and test datasets.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Train and test datasets
    """
    train_path = "data/preprocessed/train_preprocessed.csv"
    test_path = "data/preprocessed/test_preprocessed.csv"
    logger.info(f"Loading preprocessed data from {train_path} and {test_path}")
    train_preprocessed = pd.read_csv(train_path)
    test_preprocessed = pd.read_csv(test_path)
    return train_preprocessed, test_preprocessed


def engineer_features(
    train_preprocessed: pd.DataFrame, test_preprocessed: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, KBinsDiscretizer]:
    """Apply feature engineering transformations.

    Args:
        train_preprocessed (pd.DataFrame): Training dataset
        test_preprocessed (pd.DataFrame): Test dataset

    Returns:
        tuple containing:
            pd.DataFrame: Engineered training features
            pd.DataFrame: Engineered test features
            StandardScaler: Fitted scaler
    """
    logger.info("Engineering features...")
    feature_columns = [col for col in train_preprocessed.columns if col != "target"]

    scaler = StandardScaler()
    binning = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
    
    train_processed = train_preprocessed.copy()
    test_processed = test_preprocessed.copy()

    # Fits and apply transformations in train data
    train_processed[feature_columns] = scaler.fit_transform(train_processed[feature_columns])
    train_processed["balance"] = binning.fit_transform(train_preprocessed[["balance"]])

    # Apply transformations in the test data with the fitted objects
    test_processed[feature_columns] = scaler.transform(test_processed[feature_columns])
    test_processed["balance"] = binning.transform(test_preprocessed[["balance"]])


    return train_processed, test_processed, scaler, binning


def save_artifacts(
    train_processed: pd.DataFrame, test_processed: pd.DataFrame, scaler: StandardScaler, binning: KBinsDiscretizer
) -> None:
    """Save engineered features and scaler.

    Args:
        train_processed (pd.DataFrame): Engineered training data
        test_processed (pd.DataFrame): Engineered test data
        scaler (StandardScaler): Fitted scaler
        binning (KBinsDiscretizer): Fitted Discretizer
    """
    # Save processed data
    output_dir = "data/processed"
    logger.info(f"Saving engineered features to folder {output_dir}")

    train_path = os.path.join(output_dir, "train_processed.csv")
    test_path = os.path.join(output_dir, "test_processed.csv")

    train_processed.to_csv(train_path, index=False)
    test_processed.to_csv(test_path, index=False)

    # Save scaler and discretizer
    scaler_path = os.path.join("artifacts", "[features]_scaler.joblib")
    binning_path = os.path.join("artifacts", "[balance]_discretizer.joblib")

    logger.info(f"Saving scaler to {scaler_path} and discretizer to {binning_path}")
    joblib.dump(scaler, scaler_path)
    joblib.dump(binning, binning_path)


def main() -> None:
    """Main function to orchestrate feature engineering pipeline."""
    train_preprocessed, test_preprocessed = load_preprocessed_data()
    train_processed, test_processed, scaler, binning = engineer_features(train_preprocessed, test_preprocessed)
    save_artifacts(train_processed, test_processed, scaler, binning)
    logger.info("Feature engineering completed")


if __name__ == "__main__":
    main()
