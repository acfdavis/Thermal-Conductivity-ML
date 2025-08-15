from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
from tensorflow import keras
import pandas as pd
import numpy as np
import json
from src.features import featurize_data

app = FastAPI(title="Thermal Conductivity - TensorFlow API")

# --- Load all artifacts at startup ---
try:
    model = keras.models.load_model("models_tf/tf_model.keras")
    scaler = joblib.load("models_tf/scaler.joblib")
    imputer = joblib.load("models_tf/imputer.joblib")
    with open("models_tf/feature_list.txt", "r") as f:
        feature_list = [line.strip() for line in f if line.strip()]
except FileNotFoundError as e:
    raise RuntimeError(f"Could not load model artifacts: {e}") from e

# --- API request and response models ---
class PredictionRequest(BaseModel):
    formulas: list[str] = Field(..., example=["SiC", "GaN"], description="A list of chemical formulas.")
    # --- FIX: Add temperature as a required input ---
    temperature: float = Field(..., example=300.0, description="Temperature in Kelvin for the prediction.")

class PredictionResponse(BaseModel):
    formula: str
    predicted_thermal_conductivity_W_mK: float

@app.post("/predict", response_model=list[PredictionResponse])
def predict(request: PredictionRequest):
    """Runs the full prediction pipeline for a list of chemical formulas at a given temperature."""
    print("Starting feature engineering...")
    
    # Convert the list of formulas into a DataFrame
    input_df = pd.DataFrame({'formula': request.formulas})

    # Featurize the formulas
    featured_df = featurize_data(input_df)

    # --- FIX: Add the temperature column to the DataFrame ---
    featured_df['temperature'] = request.temperature

    # Ensure all required columns are present before selection
    missing_cols = set(feature_list) - set(featured_df.columns)
    if missing_cols:
        raise HTTPException(
            status_code=500, 
            detail=f"Feature engineering failed. Missing columns: {list(missing_cols)}"
        )

    # Select features in the correct order
    X = featured_df[feature_list].to_numpy()

    # Preprocess the data
    X_imp = imputer.transform(X)
    X_s = scaler.transform(X_imp)

    # Make predictions
    predictions_log = model.predict(X_s).flatten()
    predictions = np.expm1(predictions_log)

    # Format the response
    response = [
        PredictionResponse(
            formula=formula,
            predicted_thermal_conductivity_W_mK=round(pred, 3)
        )
        for formula, pred in zip(request.formulas, predictions)
    ]
    return response
