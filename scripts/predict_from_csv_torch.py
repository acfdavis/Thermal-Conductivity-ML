import argparse
import joblib
import pandas as pd
import numpy as np
import torch

from src.features import featurize_data
from scripts.train_torch import MLP  # reuses the same architecture

def main():
    p = argparse.ArgumentParser(description="Predict with PyTorch model from CSV.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True)   # models_torch/torch_model.pt
    p.add_argument("--scaler", required=True)  # models_torch/scaler.joblib
    p.add_argument("--imputer", required=True) # models_torch/imputer.joblib
    p.add_argument("--feature-list", required=True) # models_torch/feature_list.txt
    args = p.parse_args()

    df = pd.read_csv(args.input)
    feats = featurize_data(df["formula"].tolist())

    # Align columns & order with training
    feature_cols = [c.strip() for c in open(args.feature_list, "r", encoding="utf-8").read().splitlines()]
    for col in feature_cols:
        if col not in feats.columns:
            feats[col] = np.nan
    X = feats.reindex(columns=feature_cols).to_numpy()

    imputer = joblib.load(args.imputer)
    scaler = joblib.load(args.scaler)
    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)

    model = MLP(X_scaled.shape[1])
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        preds = model(torch.tensor(X_scaled, dtype=torch.float32)).squeeze(1).numpy()

    df["prediction"] = preds
    df.to_csv(args.output, index=False)
    print(f"[predict_torch] Saved predictions to {args.output}")

if __name__ == "__main__":
    main()
