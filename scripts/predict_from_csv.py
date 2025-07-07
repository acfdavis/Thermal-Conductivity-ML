# predict_from_csv.py

import argparse
import pandas as pd
import joblib
from src.features import featurize_data

REQUIRED_COLUMNS = ["formula", "temperature", "pressure", "phase"]

def load_model(model_path):
    return joblib.load(model_path)

def validate_input(df):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def predict(model, X):
    return model.predict(X)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to saved model file")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV with raw features")
    parser.add_argument("--output", type=str, required=True, help="Path to save predictions CSV")

    args = parser.parse_args()

    df_input = pd.read_csv(args.input)
    validate_input(df_input)

    X = featurize_data(df_input)
    model = load_model(args.model)
    preds = predict(model, X)

    output_df = df_input.copy()
    output_df["prediction"] = preds
    output_df.to_csv(args.output, index=False)

    print(f"Predictions saved to {args.output}")
