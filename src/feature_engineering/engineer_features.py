import logging
import os
import yaml
import joblib

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from category_encoders.target_encoder import TargetEncoder
from scipy import sparse

logger = logging.getLogger("src.feature_engineering.engineer_features")


def load_params() -> dict[str, float | int]:
    """Load model hyperparameters for the train stage from params.yaml.

    Returns:
        dict[str, int | float]: dictionary containing model hyperparameters.
    """
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["engineer_features"]


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
    train_preprocessed: pd.DataFrame,
    test_preprocessed: pd.DataFrame,
    params: dict[str, int | float]
) -> tuple[pd.DataFrame, pd.DataFrame, KBinsDiscretizer, TargetEncoder, RFE]:
    """Apply feature engineering transformations.

    Args:
        train_preprocessed (pd.DataFrame): Training dataset
        test_preprocessed (pd.DataFrame): Test dataset
        params (dict): Hyperparameters

    Returns:
        tuple containing:
            pd.DataFrame: Engineered training features
            pd.DataFrame: Engineered test features            
            KBinsDiscretizer: Fitted discretizer
            TargetEncoder: Fitted target encoder
            RFE: Fitted feature selector
    """
    logger.info("Engineering features...")
    feature_columns = [col for col in train_preprocessed.columns if col not in ["target", "student"]]
    
    binning = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
    target_encoder = TargetEncoder(cols=["student"])
    feature_selector = RFE(
        XGBClassifier(n_estimators=params["n_estimators"]),
        n_features_to_select=params["n_features_to_select"],
        step=1
    )

    
    # Binning
    train_preprocessed["balance_bin"] = binning.fit_transform(train_preprocessed[["balance"]])
    test_preprocessed["balance_bin"] = binning.transform(test_preprocessed[["balance"]])

    # Target encoding
    train_preprocessed["student_target_enc"] = target_encoder.fit_transform(
        train_preprocessed[["student"]], train_preprocessed["target"]
    )
    test_preprocessed["student_target_enc"] = target_encoder.transform(test_preprocessed[["student"]])

    # Feature engineering (padronizado)
    for df in [train_preprocessed, test_preprocessed]:
        df["balance_warning_zone"] = df["balance"].between(1000, 2000).astype(int)
        df["balance_income_ratio"] = df["balance"] / df["income"]
        df["balance_over_mean_income"] = df["balance"] / df["income"].mean()
        df["balance_over_mean"] = (df["balance"] > df["balance"].mean()).astype(int)
        df["balance_quantile"] = (df["balance"] > df["balance"].quantile(0.25)).astype(int)
        df["income_over_mean_balance"] = df["income"] / df["balance"].mean()
        df["balance_flag_high"] = (df["balance"] > df["income"]).astype(int)
        df["income_flag_high"] = (df["income"] > df["balance"]).astype(int)

    # Feature selection
    X_train = feature_selector.fit_transform(train_preprocessed.drop(["target", "balance", "income", "student"], axis=1), train_preprocessed["target"])
    X_test = feature_selector.transform(test_preprocessed.drop(["target", "balance", "income", "student"], axis=1))

    logger.info("Train - Shape after feature selection: {}".format(X_train.shape))

    # Convert to DataFrame
    def to_dataframe(X, columns):
        if sparse.issparse(X):
            X = X.toarray()
        return pd.DataFrame(X, columns=columns[:X.shape[1]])

    train_processed = to_dataframe(X_train, train_preprocessed.drop(["target", "student"], axis=1).columns)
    test_processed = to_dataframe(X_test, test_preprocessed.drop(["target", "student"], axis=1).columns)

    # Merge target back
    train_processed["target"] = train_preprocessed["target"].values
    test_processed["target"] = test_preprocessed["target"].values

    return train_processed, test_processed, binning, target_encoder, feature_selector


def save_artifacts(
    train_processed: pd.DataFrame,
    test_processed: pd.DataFrame,    
    binning: KBinsDiscretizer,
    target_encoder: TargetEncoder,
    feature_selector: RFE
) -> None:
    """Save engineered features and fitted objects."""
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    logger.info(f"Saving engineered features to folder {output_dir}")

    train_path = os.path.join(output_dir, "train_processed.csv")
    test_path = os.path.join(output_dir, "test_processed.csv")

    train_processed.to_csv(train_path, index=False)
    test_processed.to_csv(test_path, index=False)

    # Save fitted objects    
    joblib.dump(binning, os.path.join("artifacts", "balance_discretizer.joblib"))
    joblib.dump(target_encoder, os.path.join("artifacts", "target_encoder.joblib"))
    joblib.dump(feature_selector, os.path.join("artifacts", "feature_selector.joblib"))

    logger.info("Artifacts saved successfully.")


def main() -> None:
    """Main function to orchestrate feature engineering pipeline."""
    params = load_params()
    train_preprocessed, test_preprocessed = load_preprocessed_data()
    train_processed, test_processed, binning, target_encoder, feature_selector = engineer_features(
        train_preprocessed, test_preprocessed, params
    )
    save_artifacts(train_processed, test_processed, binning, target_encoder, feature_selector)
    logger.info("Feature engineering completed")


if __name__ == "__main__":
    main()