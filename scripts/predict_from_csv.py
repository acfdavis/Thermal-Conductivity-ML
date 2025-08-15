"""
Prediction script for the Thermal Conductivity ML model.

This script takes a CSV file with material compositions and process parameters,
runs the full feature engineering and prediction pipeline, and outputs a new
CSV with the predicted thermal conductivity.

Example usage:
    python scripts/predict_from_csv.py ^
        --model models/tuned_xgboost_model.joblib ^
        --scaler models/scaler.joblib ^
        --features data/processed/selected_features_xgb.json ^
        --input data/example_input.csv ^
        --output data/predictions.csv
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
import json
from dotenv import load_dotenv

# Add project root to Python path to allow importing from 'src'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# Explicitly load the .env file from the project root
# This ensures the correct API key is used, overriding any system-level variables.
dotenv_path = os.path.join(PROJECT_ROOT, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path, override=True)
    print("Successfully loaded environment variables from .env file.")
else:
    print("Warning: .env file not found. Relying on system environment variables.")


from src.features import featurize_data
from src.cluster_features import add_cluster_label

# Define the columns required in the input CSV
REQUIRED_COLUMNS = ["formula", "temperature", "pressure", "phase"]

def load_artifacts(model_path, scaler_path, features_path):
    """Loads the trained model, scaler, and the list of selected features."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at: {scaler_path}. Please run notebook 5 to generate it.")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found at: {features_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(features_path, 'r') as f:
        selected_features = json.load(f)
    
    print("Successfully loaded model, scaler, and feature list.")
    return model, scaler, selected_features

def validate_input_columns(df):
    """Checks if the input DataFrame contains all required columns."""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input CSV is missing required columns: {missing_cols}")
    print("Input columns validated.")

def make_predictions(model, scaler, selected_features_from_file, input_df):
    """Runs the full prediction pipeline on new data."""
    # 1. Generate all available features, including the cluster label
    print("Step 1: Generating features for input data...")
    X_featurized = featurize_data(input_df)
    X_featurized = add_cluster_label(X_featurized, models_dir="models")
    print(f"Generated {X_featurized.shape[1]} features.")

    # 2. Get the list of features the model ACTUALLY expects from the model object itself
    # This is the most reliable source of truth and overrides the JSON file.
    model_features = model.get_booster().feature_names
    print(f"Step 2: Preparing the {len(model_features)} features the model was trained on...")

    # 3. Select and order the columns based on the model's internal list
    X_selected = X_featurized.reindex(columns=model_features)

    # Handle any potential missing values
    if X_selected.isnull().values.any():
        print("Warning: Missing values detected after feature selection. Filling with 0.")
        X_selected = X_selected.fillna(0)

    # 4. Scale the features using the previously fitted scaler
    # The scaler should have been trained on the same feature set as the model.
    print("Step 3: Scaling features...")
    X_scaled_data = scaler.transform(X_selected)
    
    # Create a DataFrame with the scaled data, preserving column names and order.
    # This DataFrame is now ready for prediction.
    X_final = pd.DataFrame(X_scaled_data, columns=model_features, index=input_df.index)

    # 5. Make predictions
    print("Step 4: Making predictions...")
    # The model predicts the log-transformed value
    log_predictions = model.predict(X_final)

    # 6. Apply the inverse transformation to get the final prediction
    # np.expm1 is the inverse of np.log1p (log(1+x))
    predictions = np.expm1(log_predictions)

    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict thermal conductivity from a CSV file.")
    parser.add_argument("--model", type=str, default="models/tuned_xgboost_model.joblib", help="Path to the trained model file.")
    parser.add_argument("--scaler", type=str, default="models/scaler.joblib", help="Path to the fitted scaler file.")
    parser.add_argument("--features", type=str, default="data/processed/selected_features_xgb.json", help="Path to the JSON file with selected feature names.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output CSV with predictions.")

    args = parser.parse_args()

    try:
        # Load all necessary artifacts
        model, scaler, selected_features = load_artifacts(args.model, args.scaler, args.features)

        # Load and validate input data
        input_df = pd.read_csv(args.input)
        validate_input_columns(input_df)

        # Get predictions
        predictions = make_predictions(model, scaler, selected_features, input_df)

        # Create and save the output file with polished formatting
        output_df = input_df.copy()

        # Round the prediction and add units to the column name
        output_df["predicted_thermal_conductivity_W_mK"] = np.round(predictions, 3)

        # Add a column for the model's expected error (Mean Absolute Error from test set).
        # This value is taken from the final model evaluation in notebook 5 and provides
        # context for the model's expected accuracy.
        MODEL_MAE = 0.351  # NOTE: This should be the final MAE from the tuned model
        output_df["prediction_error_est_W_mK"] = MODEL_MAE

        output_df.to_csv(args.output, index=False)

        print(f"\nSuccess! Predictions saved to {args.output}")
        print(output_df.head())

    except (FileNotFoundError, ValueError) as e:
        print(f"\nError: {e}")
        sys.exit(1)
