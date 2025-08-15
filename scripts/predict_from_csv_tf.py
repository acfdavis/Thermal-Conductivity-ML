import argparse
import joblib
import pandas as pd
import numpy as np
from tensorflow import keras

from src.features import featurize_data

def main():
    p = argparse.ArgumentParser(description="Predict with TensorFlow model from CSV.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True)        # models_tf/tf_model.keras
    p.add_argument("--scaler", required=True)       # models_tf/scaler.joblib
    p.add_argument("--imputer", required=True)      # models_tf/imputer.joblib
    p.add_argument("--feature-list", required=True) # models_tf/feature_list.txt
    args = p.parse_args()

    df = pd.read_csv(args.input)
    feats = featurize_data(df["formula"].tolist())

    feature_cols = [c.strip() for c in open(args.feature_list, "r", encoding="utf-8").read().splitlines()]
    for col in feature_cols:
        if col not in feats.columns:
            feats[col] = np.nan
    X = feats.reindex(columns=feature_cols).to_numpy()

    imputer = joblib.load(args.imputer)
    scaler = joblib.load(args.scaler)
    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)

    model = keras.models.load_model(args.model)
    preds = model.predict(X_scaled, verbose=0).reshape(-1)

    df["prediction"] = preds
    df.to_csv(args.output, index=False)
    print(f"[predict_tf] Saved predictions to {args.output}")

if __name__ == "__main__":
    main()
