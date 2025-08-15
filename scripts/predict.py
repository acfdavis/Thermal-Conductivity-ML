# For XGBoost:  
#python scripts/predict.py --framework xgboost  --input data/example_input.csv --output data/predictions_xgb.csv --model models/tuned_xgboost_model.joblib --scaler models/scaler.joblib

# python scripts/predict.py ^
#    --framework xgboost ^
#    --model models/tuned_xgboost_model.joblib ^
#    --scaler models/scaler.joblib ^
#    --input data/example_input.csv ^
#    --output data/predictions_xgb.csv

#For TensorFlow:
#python scripts/predict.py --framework tensorflow  --input data/example_input.csv --output data/predictions_tf.csv --model models_tf/tf_model.keras --scaler models_tf/scaler.joblib --features models_tf/feature_list.txt

#python scripts/predict.py ^
#    --framework tensorflow ^
#    --model models_tf/tf_model.keras ^
#    --scaler models_tf/scaler.joblib ^
#    --features models_tf/feature_list.txt ^
#    --input data/example_input.csv ^
#    --output data/predictions_tf.csv

#For Torch:

#python scripts/predict.py --framework torch  --input data/example_input.csv --output data/predictions_torch.csv --model models_torch/torch_model.pt --scaler models_torch/scaler.joblib --features models_torch/feature_list.json

#python scripts/predict.py ^
#    --framework torch ^
#    --model models_torch/torch_model.pt ^
#    --scaler models_torch/scaler.joblib ^
#    --features models_torch/feature_list.json ^
#    --input data/example_input.csv ^
#    --output data/predictions_torch.csv

import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
import json
from dotenv import load_dotenv

# --- Path Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
dotenv_path = os.path.join(PROJECT_ROOT, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path, override=True)

from src.predictor import PredictionPipeline

# --- Framework-Specific Loaders ---
def load_xgboost_artifacts(model, scaler, **kwargs):
    """Loads XGBoost artifacts. Argument names match argparse destinations."""
    model_obj = joblib.load(model)
    scaler_obj = joblib.load(scaler)
    feature_list = model_obj.get_booster().feature_names
    return model_obj, scaler_obj, feature_list

def load_torch_artifacts(model, scaler, features, **kwargs):
    """Loads PyTorch artifacts. Argument names match argparse destinations."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set weights_only=False because we are loading a full model object from a trusted source.
    model_obj = torch.load(model, map_location=device, weights_only=False)
    
    model_obj.to(device).eval()
    scaler_obj = joblib.load(scaler)
    with open(features, 'r') as f:
        # This was reading a .txt file, but the torch training script saves a .json.
        # Let's make sure it reads JSON.
        feature_list = json.load(f)
    return model_obj, scaler_obj, feature_list

def load_tf_artifacts(model, scaler, features, **kwargs):
    """Loads TensorFlow artifacts. Argument names match argparse destinations."""
    from tensorflow import keras
    model_obj = keras.models.load_model(model)
    scaler_obj = joblib.load(scaler)
    # Read a plain text file with one feature per line
    with open(features, 'r') as f:
        feature_list = [line.strip() for line in f if line.strip()]
    return model_obj, scaler_obj, feature_list

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict thermal conductivity from a CSV file using a specified framework.")
    parser.add_argument("--framework", type=str, required=True, choices=['xgboost', 'torch', 'tensorflow'], help="The ML framework of the model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--scaler", type=str, required=True, help="Path to the fitted scaler file.")
    parser.add_argument("--features", type=str, help="Path to the JSON file with feature names (required for torch/tensorflow).")
    parser.add_argument("--input", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output CSV with predictions.")
    args = parser.parse_args()

    # --- Validate Arguments ---
    if args.framework in ['torch', 'tensorflow'] and not args.features:
        parser.error("--features is required for the 'torch' and 'tensorflow' frameworks.")

    try:
        # --- Load Artifacts ---
        print(f"Loading artifacts for '{args.framework}' framework...")
        loaders = {
            'xgboost': load_xgboost_artifacts,
            'torch': load_torch_artifacts,
            'tensorflow': load_tf_artifacts
        }
        model, scaler, feature_list = loaders[args.framework](**vars(args))
        print("Artifacts loaded successfully.")

        # --- Initialize and Run Pipeline ---
        pipeline = PredictionPipeline(model, scaler, feature_list)
        input_df = pd.read_csv(args.input)
        predictions = pipeline.predict(input_df)

        # --- Save Output ---
        output_df = input_df.copy()
        output_df["predicted_thermal_conductivity_W_mK"] = np.round(predictions, 3)
        output_df.to_csv(args.output, index=False)
        print(f"\nSuccess! Predictions saved to {args.output}")
        print(output_df.head())

    except (FileNotFoundError, ValueError, NotImplementedError, ImportError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)