import io
import logging
import os

import joblib
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from flask import Flask, render_template, request


logger = logging.getLogger("app.main")


class ModelService:
    def __init__(self) -> None:
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load all artifacts from the local project folder."""
        logger.info("Loading artifacts from local project folder")

        # Define base paths
        artifacts_dir = "artifacts"
        models_dir = "models"

        # Define paths to the preprocessing artifacts
        balance_discretizer_path = os.path.join(
            artifacts_dir, "balance_discretizer.joblib"
        )
        feature_selector_path = os.path.join(artifacts_dir, "feature_selector.joblib")
        preprocessor_path = os.path.join(artifacts_dir, "preprocessor.joblib")
        target_encoder_path = os.path.join(
            artifacts_dir, "target_encoder.joblib"
        )
        
        # Load model from registry
        logger.info("Loading registered model from MLflow Model Registry")
        self.model = mlflow.xgboost.load_model("models:/xgb_model/latest")

        # Get run_id from model version metadata
        client = MlflowClient()
        run_id = client.get_registered_model("xgb_model").latest_versions[0].run_id

        # Load related artifacts
        logger.info(f"Loading artifacts from run {run_id}")
        artifacts_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="")

        preprocessor_path = os.path.join(artifacts_dir, "preprocessor.joblib")
        self.preprocessor = joblib.load(preprocessor_path)
        balance_discretizer_path = os.path.join(artifacts_dir, "balance_discretizer.joblib")
        self.balance_discretizer = joblib.load(balance_discretizer_path)
        target_encoder_path = os.path.join(artifacts_dir, "target_encoder.joblib")
        self.target_encoder = joblib.load(target_encoder_path)
        feature_selector_path = os.path.join(artifacts_dir, "feature_selector.joblib")
        self.feature_selector = joblib.load(feature_selector_path)
        
        logger.info("Successfully loaded model and related artifacts")


        
        # Load all required artifacts
        #self.preprocessor = joblib.load(preprocessor_path)
        #self.balance_discretizer = joblib.load(balance_discretizer_path)        
        #self.target_encoder = joblib.load(target_encoder_path)
        #self.feature_selector = joblib.load(feature_selector_path)        
        #self.model = joblib.load(model_path)

        # logger.info("Successfully loaded all artifacts")

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Make predictions using the full pipeline.

        Args:
            features: DataFrame containing the input features

        Returns:
            Series containing the predictions
        """
        # Apply transformations in sequence
        X_imputed = self.preprocessor.transform(features)    
        X_discretized = self.balance_discretizer.transform(X_imputed)
        X_encoded = self.target_encoder.transform(X_discretized)
        X_selected = self.feature_selector.transform(X_encoded)

        # Get model predictions
        y_pred = self.model.predict(X_selected)

        # Decode predictions
        y_decoded = self.target_encoder.inverse_transform(y_pred)

        return pd.DataFrame({"Prediction": y_decoded.ravel()}, index=features.index)


def create_routes(app: Flask) -> None:
    """Create all routes for the application."""

    @app.route("/")
    def index() -> str:
        """Serve the HTML upload interface."""
        return render_template("index.html")

    @app.route("/upload", methods=["POST"])
    def upload() -> str:
        """Handle CSV file upload, validate features, and return predictions."""
        file = request.files["file"]
        if not file.filename.endswith(".csv"):
            return render_template("index.html", error="Please upload a CSV file")

        try:
            # Read CSV content
            content = file.read().decode("utf-8")
            features = pd.read_csv(io.StringIO(content))

            # Validate column names against breast cancer dataset
            expected_features = pd.read_csv("data/raw/raw.csv").columns
            missing_cols = [
                col for col in expected_features if col not in features.columns
            ]
            if missing_cols:
                return render_template(
                    "index.html",
                    error=f"Missing required columns: {', '.join(missing_cols).replace("default", "target")}",
                )
            features = features[expected_features]

            # Make predictions
            predictions = app.model_service.predict(features)

            # Format predictions for display
            result = predictions.to_string()

            return render_template("index.html", predictions=result)

        except Exception as e:
            logger.error(
                f"Error processing file: {e}", exc_info=True
            )  # Added exc_info for better logging
            return render_template(
                "index.html",
                error=f"Error processing file: {str(e)}",  # Ensure e is string
            )


# Create and configure Flask app at module level
app = Flask(__name__)
app.model_service = ModelService()
create_routes(app)
logger.info("Application initialized with model service and routes")


def main() -> None:
    """Run the Flask development server."""
    app.run(host="0.0.0.0", port=5001)


if __name__ == "__main__":
    main()
