import pandas as pd
import numpy as np
from .features import featurize_data
from .cluster_features import add_cluster_label

class PredictionPipeline:
    """Encapsulates the full prediction pipeline."""

    def __init__(self, model, scaler, feature_list):
        if model is None or scaler is None or feature_list is None:
            raise ValueError("Model, scaler, and feature_list must be provided.")
        self.model = model
        self.scaler = scaler
        self.feature_list = feature_list
        self.framework = self._get_framework_name()

    def _get_framework_name(self):
        """Identifies the ML framework from the model object's type."""
        model_class = str(type(self.model)).lower()
        if "xgboost" in model_class:
            return "xgboost"
        elif "torch" in model_class:
            return "torch"
        elif "tensorflow" in model_class or "keras" in model_class:
            return "tensorflow"
        return "unknown"

    def _prepare_features(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Runs the full feature engineering and selection pipeline."""
        print("Step 1: Generating features for input data...")
        X_featurized = featurize_data(input_df)
        
        # Add cluster label only if the model was trained with it
        if 'cluster_label' in self.feature_list:
            print("Adding cluster label feature...")
            X_featurized = add_cluster_label(X_featurized, models_dir="models")
        
        print(f"Generated {X_featurized.shape[1]} features.")

        print(f"Step 2: Preparing the {len(self.feature_list)} features the model was trained on...")
        X_selected = X_featurized.reindex(columns=self.feature_list).fillna(0)
        
        return X_selected

    def predict(self, input_df: pd.DataFrame) -> np.ndarray:
        """Makes predictions using the appropriate framework logic."""
        X_selected = self._prepare_features(input_df)

        print("Step 3: Scaling features...")
        X_scaled = self.scaler.transform(X_selected)

        print("Step 4: Making predictions...")
        if self.framework == "xgboost":
            log_predictions = self.model.predict(X_scaled)
        elif self.framework == "tensorflow":
            log_predictions = self.model.predict(X_scaled).flatten()
        elif self.framework == "torch":
            import torch
            device = next(self.model.parameters()).device
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            with torch.no_grad():
                log_predictions = self.model(X_tensor).cpu().numpy().flatten()
        else:
            raise NotImplementedError(f"Prediction for framework '{self.framework}' is not implemented.")

        print("Step 5: Applying inverse transformation...")
        predictions = np.expm1(log_predictions)
        return predictions