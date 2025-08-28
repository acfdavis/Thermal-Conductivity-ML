"""
Prediction script for the Thermal Conductivity ML model.

This script takes a CSV file with material compositions and process parameters,
runs the full feature engineering and prediction pipeline, and outputs a new
CSV with the predicted thermal conductivity.

Example usage:  python .\scripts\predict_from_csv.py --model .\models\tuned_xgboost_model.joblib --scaler .\models\scaler.joblib --features .\data\processed\selected_features_xgb.json --input .\data\example_input.csv --output .\data\predictions.csv
"""
from dotenv import load_dotenv
import re
import os
import sys
import json
import joblib
import argparse
import numpy as np
import pandas as pd

# Add project root to Python path to allow importing from 'src'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(str(PROJECT_ROOT))

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


def is_generic_feature_list(names):
    """Check if a list of names looks like XGBoost's generic f0, f1, ..."""
    if not names:
        return False
    return all(re.fullmatch(r"f\d+", n) for n in names)

# Define the columns required in the input CSV
REQUIRED_COLUMNS = ["formula", "temperature", "pressure", "phase"]
# Optional columns the pipeline can leverage if present
OPTIONAL_COLUMNS = ["space_group"]  # integer 1-230 supplied by user to disambiguate polymorphs

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
    """Checks required columns; validates optional columns when present."""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input CSV is missing required columns: {missing_cols}")

    if 'space_group' in df.columns:
        # Coerce to numeric and check range
        sg_numeric = pd.to_numeric(df['space_group'], errors='coerce')
        if sg_numeric.isna().any():
            bad_rows = sg_numeric[sg_numeric.isna()].index.tolist()
            raise ValueError(f"space_group column contains non-numeric values at rows: {bad_rows}")
        out_of_range = sg_numeric[(sg_numeric < 1) | (sg_numeric > 230)]
        if not out_of_range.empty:
            raise ValueError(f"space_group values out of valid range 1-230 at rows: {out_of_range.index.tolist()}")
        # replace original with cleaned values
        df['space_group'] = sg_numeric.astype(int)
        print("Validated user-provided space_group column.")
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

def load_feature_lists(model, scaler, features_path):
    """Resolve feature name sets from artifacts; fall back safely if booster names missing."""
    with open(features_path, "r") as f:
        file_features = json.load(f)

    # Booster feature names (may be None or generic f0,f1,... if trained on numpy arrays)
    try:
        booster = model.get_booster()
        booster_features = booster.feature_names or []
    except Exception:
        booster_features = []

    scaler_features = list(getattr(scaler, "feature_names_in_", []))

    return file_features, booster_features, scaler_features


def decide_target_features(file_feats, booster_feats, scaler_feats, allow_cluster_label=False):
    """
    Pick a consistent ordered feature list for inference.
    Priority order:
      1. If scaler feature names exist, they define ordering (they were used during fit).
      2. Else if booster (model) feature names look non-generic, use those.
      3. Else use file feature list.
    Handle cluster_label mismatches (drop if incomplete across artifacts).
    """
    # Detect generic xgboost names f0,f1,...
    def looks_generic(names):
        return bool(names) and all(n.startswith("f") and n[1:].isdigit() for n in names)

    use_scaler = len(scaler_feats) > 0
    use_booster = len(booster_feats) > 0 and not looks_generic(booster_feats)

    if use_scaler:
        base = scaler_feats[:]
        source = "scaler"
    elif use_booster:
        base = booster_feats[:]
        source = "model"
    else:
        base = file_feats[:]
        source = "features file"

    # Cluster label logic
    if "cluster_label" in base:
        if not allow_cluster_label:
            base = [f for f in base if f != "cluster_label"]
            print("Info: Dropped cluster_label (artifacts not aligned).")
    else:
        if allow_cluster_label and "cluster_label" in file_feats and "cluster_label" in booster_feats and "cluster_label" in scaler_feats:
            base.append("cluster_label")

    return base, source


def ensure_columns(full_features, target_features):
    """Add any missing target columns with 0.0 and return DataFrame in exact order."""
    missing = [c for c in target_features if c not in full_features.columns]
    if missing:
        for c in missing:
            full_features[c] = 0.0
        print(f"Added {len(missing)} missing feature columns with 0 fill (e.g. {missing[:5]}).")
    return full_features.reindex(columns=target_features)


if __name__ == "__main__":
    import argparse, os, sys, json
    import pandas as pd
    import numpy as np
    import joblib

    parser = argparse.ArgumentParser(description="Predict thermal conductivity from a CSV file.")
    parser.add_argument("--model", type=str, default="models/tuned_xgboost_model.joblib")
    parser.add_argument("--scaler", type=str, default="models/scaler.joblib")
    parser.add_argument("--features", type=str, default="data/processed/selected_features_xgb.json")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--skip-jarvis", action="store_true",
                        help="Skip JARVIS feature retrieval; fill those features with fallback values.")
    parser.add_argument("--no-scale", action="store_true",
                        help="Skip scaler (only if model was trained without scaling).")
    parser.add_argument("--allow-cluster-label", action="store_true",
                        help="Force inclusion of cluster_label if present in all artifacts.")
    parser.add_argument("--raw-target", action="store_true",
                        help="Model was trained on raw target (no log1p); skip expm1.")
    args = parser.parse_args()

    # Load artifacts
    model = joblib.load(args.model)
    scaler = None
    if not args.no_scale:
        scaler = joblib.load(args.scaler)
    with open(args.features) as f:
        file_feature_names = json.load(f)
    print("Successfully loaded model, scaler, and feature list." if scaler else
          "Successfully loaded model and feature list (no scaler).")

    df_in = pd.read_csv(args.input)
    validate_input_columns(df_in)

    if args.skip_jarvis:
        os.environ["TCML_SKIP_JARVIS"] = "1"

    print("Step 1: Generating features for input data...")
    full_features = featurize_data(df_in, composition_col="formula")

    # Only add cluster label if explicitly allowed and artifacts align (decided later)
    if args.allow_cluster_label:
        try:
            full_features = add_cluster_label(full_features, models_dir="models")
        except Exception as e:
            print(f"Warning: could not add cluster_label ({e}).")

    # Resolve feature lists
    booster_feats = []
    scaler_feats = []
    if not args.no_scale:
        file_feats, booster_feats, scaler_feats = load_feature_lists(model, scaler, args.features)
    else:
        file_feats = file_feature_names

    target_features, src = decide_target_features(
        file_feature_names,
        booster_feats,
        scaler_feats,
        allow_cluster_label=args.allow_cluster_label
    )
    print(f"Using {len(target_features)} features from {src}.")

    # Align columns
    X = ensure_columns(full_features, target_features)

    # Ensure cluster_label exists if scaler expects it (even if we chose not to use it for model features)
    if (not args.no_scale) and scaler is not None:
        scaler_feats = list(getattr(scaler, "feature_names_in_", []))
        if "cluster_label" in scaler_feats and "cluster_label" not in X.columns:
            # Try to compute it; if fails fill with 0
            try:
                X = add_cluster_label(X, models_dir="models")
                print("Added cluster_label to satisfy scaler expectation.")
            except Exception as e:
                X["cluster_label"] = 0
                print(f"Inserted cluster_label=0 (could not compute: {e})")

    # After potential insertion, re-add any remaining missing scaler features as 0 (do NOT abort)
    if (not args.no_scale) and scaler is not None:
        scaler_feats = list(getattr(scaler, "feature_names_in_", []))
        still_missing = [c for c in scaler_feats if c not in X.columns]
        if still_missing:
            for c in still_missing:
                X[c] = 0.0
            print(f"Filled {len(still_missing)} scaler-required missing columns with 0 (e.g. {still_missing[:5]}).")

    # Impute NaNs (median; fallback 0)
    if X.isna().any().any():
        na_cols = X.columns[X.isna().any()].tolist()
        for c in na_cols:
            vals = X[c].values
            med = np.nanmedian(vals)
            if not np.isfinite(med):
                med = 0.0
            X[c] = np.where(np.isnan(vals), med, vals)
        print(f"Imputed NaNs in: {na_cols}")

    # Scale or not
    if args.no_scale:
        X_used = X[target_features].values if target_features else X.values
    else:
        scaler_feats = list(getattr(scaler, "feature_names_in_", []))
        # Order X to scaler feature order for transform
        X_for_scale = X.reindex(columns=scaler_feats)
        X_scaled = scaler.transform(X_for_scale)
        # Reorder to model feature order if needed
        try:
            model_feats = model.get_booster().feature_names or []
        except Exception:
            model_feats = []

        if model_feats and not is_generic_feature_list(model_feats):
            if set(model_feats) == set(scaler_feats):
                reorder_idx = [scaler_feats.index(f) for f in model_feats]
                X_used = X_scaled[:, reorder_idx]
            else:
                # Fallback if model and scaler features differ but are not generic
                X_used = X_scaled
        else:
            # Use as-is if model features are generic (f0, f1...) or unavailable
            X_used = X_scaled
    
    # Predict
    preds_raw = model.predict(X_used)

    # Heuristic to avoid applying expm1 to raw-scale predictions
    if np.nanmax(preds_raw) > 25:
        print("Heuristic: large prediction values detected; skipping expm1 (assuming raw target).")
        preds_final = preds_raw
    else:
        preds_final = np.expm1(preds_raw)
    
    df_out = df_in.copy()
    df_out["predicted_thermal_conductivity_W_mK"] = np.round(preds_final, 3)
    df_out.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
    if args.no_scale:
        print("NOTE: --no-scale used. Ensure model was trained without scaling.")
    if args.skip_jarvis:
        print("WARNING: JARVIS features skipped; predictions may be less reliable.")
