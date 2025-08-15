import os
import random
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

from src.utils import load_or_process_dataframe  # your existing loader

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CACHE_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'featurized.parquet')

df = load_or_process_dataframe(cache_path=CACHE_PATH, project_root=PROJECT_ROOT)

TARGET_COL = "thermal_conductivity"
if TARGET_COL not in df.columns:
    raise ValueError(f"[train_tf] Expected target column '{TARGET_COL}' not found. Columns: {list(df.columns)}")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c != TARGET_COL]
if not feature_cols:
    non_num = [c for c in df.columns if c not in numeric_cols]
    raise ValueError(f"[train_tf] No numeric features found. Non-numeric columns present: {non_num}")

X = df[feature_cols].to_numpy()

# --- CRITICAL CHANGE: Log-transform the target variable ---
y = np.log1p(df[TARGET_COL].to_numpy())
print(f"Target variable '{TARGET_COL}' has been log-transformed.")

Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=SEED)

imputer = SimpleImputer(strategy="median").fit(Xtr)
Xtr_imp = imputer.transform(Xtr)
Xva_imp = imputer.transform(Xva)

scaler = StandardScaler().fit(Xtr_imp)
Xtr_s = scaler.transform(Xtr_imp)
Xva_s = scaler.transform(Xva_imp)

# --- Save Preprocessing Artifacts and Feature List ---
MODELS_DIR = Path("models_tf")
MODELS_DIR.mkdir(exist_ok=True)
joblib.dump(imputer, MODELS_DIR / "imputer.joblib")
joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
(MODELS_DIR / "feature_list.txt").write_text("\n".join(feature_cols), encoding="utf-8")
print(f"Saved imputer, scaler, and feature list to {MODELS_DIR}")

# --- Build a More Robust Keras Model ---
model = keras.Sequential([
    layers.Input(shape=(Xtr_s.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),  # ADDED: Stabilizes learning
    layers.Dropout(0.3),          # ADDED: Prevents overfitting
    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),  # ADDED: Stabilizes learning
    layers.Dropout(0.3),          # ADDED: Prevents overfitting
    layers.Dense(32, activation="relu"),
    layers.BatchNormalization(),
    layers.Dense(1) # Output layer (predicts the log-transformed value)
])

# --- CRITICAL CHANGE: Compile with Gradient Clipping ---
optimizer = keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0)
model.compile(optimizer=optimizer, loss="mean_squared_error")
model.summary()

# --- CRITICAL CHANGE: Use Early Stopping for Training ---
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15, # Stop if val_loss doesn't improve for 15 epochs
    restore_best_weights=True # Restore model weights from the epoch with the best val_loss
)

print("\n--- Starting Model Training ---")
history = model.fit(
    Xtr_s,
    ytr,
    validation_data=(Xva_s, yva),
    epochs=200, # Set a high number of epochs; EarlyStopping will find the best one
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)

# --- Save the Final Model ---
model.save(MODELS_DIR / "tf_model.keras")
print(f"\n--- Training Complete ---")
print(f"Model saved to {MODELS_DIR / 'tf_model.keras'}")
