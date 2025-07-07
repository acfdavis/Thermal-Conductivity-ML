# main.py
# uvicorn main:app --reload --port 8000
# ngrok http 8000



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from src.data import load_and_merge_data, impute_and_clean_data  # You must define this if not present
from src.features import featurize_data
from src.modeling import evaluate_model, save_model

def run_pipeline(data_path):
    print("Loading data...")
    df = pd.read_csv(data_path)

    print("Building features...")
    X = featurize_data(df)
    y = df["target"]  # Replace with your actual target column

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model...")
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

    print("Saving model...")
    save_model(model)

if __name__ == "__main__":
    run_pipeline("data/processed/featurized.csv")


from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import os

app = FastAPI()
BASE_DIR = r"C:/Users/angel/thermal_conductivity/ml_conductivity_project_user_ready"

@app.get("/project-files")
def list_files():
    files = []
    for root, _, filenames in os.walk(BASE_DIR):
        for f in filenames:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, BASE_DIR)
            if "__pycache__" not in rel_path:
                files.append(rel_path)
    return {"files": files}

@app.get("/read-file", response_class=PlainTextResponse)
def read_file(path: str):
    full_path = os.path.join(BASE_DIR, path)
    if not os.path.isfile(full_path):
        return PlainTextResponse("File not found", status_code=404)
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()

class WriteRequest(BaseModel):
    path: str
    content: str

@app.post("/write-file")
def write_file(req: WriteRequest):
    full_path = os.path.join(BASE_DIR, req.path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(req.content)
    return {"status": "written", "path": req.path}