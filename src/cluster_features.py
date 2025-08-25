import joblib
import numpy as np
import pandas as pd
from pathlib import Path

def add_cluster_label(df_features: pd.DataFrame, models_dir: str | Path = "models") -> pd.DataFrame:
    """
    Add 'cluster_label' to df_features by applying the *trained* PCA and KMeans.
    Expects artifacts saved as: models/pca.pkl, models/kmeans.pkl, models/pca_input_columns.txt
    Returns a new DataFrame (does not modify input).
    """
    models_dir = Path(models_dir)
    pca_path = models_dir / "pca.pkl"
    km_path = models_dir / "kmeans.pkl"
    cols_path = models_dir / "pca_input_columns.txt"

    if not (pca_path.exists() and km_path.exists() and cols_path.exists()):
        # Artifacts missing; just return unchanged and print a warning.
        # The calling script will then fail on the KeyError, which is the desired behavior
        # if the full feature set cannot be generated.
        print("Warning: PCA/KMeans model artifacts not found. Cannot generate 'cluster_label'.")
        return df_features

    pca = joblib.load(pca_path)
    kmeans = joblib.load(km_path)
    with open(cols_path, "r", encoding="utf-8") as f:
        pca_cols = [line.strip() for line in f]

    X = df_features.copy()
    # Ensure all PCA columns exist; fill missing with 0.0
    # This is a safe assumption if the data was scaled/imputed before PCA during training.
    for col in pca_cols:
        if col not in X.columns:
            X[col] = 0.0
    
    # Reorder columns to match the order used for training PCA
    X_pca_input = X[pca_cols].to_numpy()

    # Apply the transformations
    Z = pca.transform(X_pca_input)
    labels = kmeans.predict(Z).astype(int)
    
    # Add the new feature to the output dataframe
    out = df_features.copy()
    out["cluster_label"] = labels
    return out