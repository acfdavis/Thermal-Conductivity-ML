import os
import random
import numpy as np
import joblib
import torch
import torch.nn as nn
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from torch.utils.data import Dataset, DataLoader

# --- For demonstration, using a placeholder function if utils is not available ---
def load_or_process_dataframe(cache_path, project_root):
    import pandas as pd
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    else:
        raise FileNotFoundError(f"Featurized data not found at {cache_path}. Please run the featurization script first.")

# --- Seed for reproducibility ---
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- Path and Data Loading ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CACHE_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'featurized.parquet')
df = load_or_process_dataframe(cache_path=CACHE_PATH, project_root=PROJECT_ROOT)

# --- Feature and Target Preparation ---
TARGET_COL = "thermal_conductivity"
if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found.")

feature_cols = [c for c in df.select_dtypes(include=np.number).columns if c != TARGET_COL]
X = df[feature_cols].to_numpy()

# --- CRITICAL CHANGE: Log-transform the target variable ---
y = np.log1p(df[TARGET_COL].to_numpy())
print(f"Target variable '{TARGET_COL}' has been log-transformed.")

# --- Train-Validation Split ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

# --- Preprocessing Pipeline (Imputer and Scaler) ---
imputer = SimpleImputer(strategy="median").fit(X_train)
X_train_imp = imputer.transform(X_train)
X_val_imp = imputer.transform(X_val)

scaler = StandardScaler().fit(X_train_imp)
X_train_s = scaler.transform(X_train_imp)
X_val_s = scaler.transform(X_val_imp)

# --- Save Preprocessing Artifacts and Feature List ---
MODELS_DIR = Path("models_torch")
MODELS_DIR.mkdir(exist_ok=True)
joblib.dump(imputer, MODELS_DIR / "imputer.joblib")
joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
with open(MODELS_DIR / "feature_list.json", "w") as f:
    json.dump(feature_cols, f)
print(f"Saved imputer, scaler, and feature list to {MODELS_DIR}")

# --- Define Model and Dataset Classes (these are fine at the top level) ---
class ThermalDataset(Dataset):
    def __init__(self, features, target):
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.target[idx], dtype=torch.float32)
        return x, y

class Net(nn.Module):
    def __init__(self, in_dim, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# --- Main Training Logic ---
def main():
    """Main function to run the training and evaluation pipeline."""
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(X_train_s.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss() # Using Mean Squared Error for regression

    # Convert data to tensors
    X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train[:, None], dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val[:, None], dtype=torch.float32).to(device)

    # --- Data Loading ---
    train_dataset = ThermalDataset(X_train_s, y_train)
    val_dataset = ThermalDataset(X_val_s, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # --- Model Training ---
    print("\n--- Starting Model Training ---")
    n_epochs = 200
    patience = 15
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            
            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient Clipping
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_losses = []
            for X_val_batch, y_val_batch in val_loader:
                y_val_pred = model(X_val_batch)
                val_loss = loss_fn(y_val_pred, y_val_batch)
                val_losses.append(val_loss.item())
            val_loss_epoch = np.mean(val_losses)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d}/{n_epochs} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss_epoch:.6f}")

        # Early stopping logic
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            epochs_no_improve = 0
            # Save the best model found so far
            torch.save(model, MODELS_DIR / "torch_model.pt")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve == patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break

    print(f"\n--- Training Complete ---")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to {MODELS_DIR / 'torch_model.pt'}")

    # --- Evaluation ---
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val_t)
    # Inverse transform the predictions
    y_val_pred_inv = np.expm1(y_val_pred.cpu().numpy())
    y_val_t_inv = np.expm1(y_val_t.cpu().numpy())

    # Calculate and print evaluation metrics
    mse = np.mean((y_val_pred_inv - y_val_t_inv) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_val_pred_inv - y_val_t_inv))
    r2 = 1 - (mse / np.var(y_val_t_inv))

    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation R^2: {r2:.4f}")

    # --- Save Artifacts ---
    joblib.dump(imputer, MODELS_DIR / "imputer.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    torch.save(model.state_dict(), MODELS_DIR / "torch_model_weights.pth")
    print(f"Saved model weights, imputer, and scaler to {MODELS_DIR}")

# --- FIX: Wrap the execution in a main block ---
if __name__ == "__main__":
    main()
