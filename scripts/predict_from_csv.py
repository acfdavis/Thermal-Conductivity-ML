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

def make_predictions(model, scaler, selected_features, input_df):
    """Runs the full prediction pipeline on new data."""
    # 1. Generate all features using the matminer library
    print("Step 1: Generating features for input data...")
    X_featurized = featurize_data(input_df)
    print(f"Generated {X_featurized.shape[1]} features.")

    # 2. Select the subset of features the model was trained on
    print(f"Step 2: Selecting the {len(selected_features)} features the model was trained on...")
    # Ensure the columns are in the same order as during training
    X_selected = X_featurized[selected_features]

    # 3. Scale the features using the previously fitted scaler
    print("Step 3: Scaling features...")
    X_scaled = scaler.transform(X_selected)

    # 4. Predict on the log-transformed scale
    print("Step 4: Making predictions...")
    log_predictions = model.predict(X_scaled)

    # 5. Inverse transform the predictions to the original scale
    print("Step 5: Applying inverse transformation to predictions...")
    final_predictions = np.expm1(log_predictions)
    
    return final_predictions

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
