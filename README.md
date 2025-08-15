# Predicting Thermal Conductivity with Machine Learning

This project presents a complete, end-to-end machine learning pipeline to predict the thermal conductivity of inorganic materials using only their chemical formula. It showcases a professional, modular workflow, starting with an initial XGBoost model and expanding to include TensorFlow and PyTorch baselines, all accessible through a unified prediction script and deployable as a containerized API.

The final XGBoost model achieves a **Test R² of 0.855** and a **Mean Absolute Error (MAE) of 0.35 W/mK** on a log-transformed scale, demonstrating high accuracy in predicting this complex material property.

## Key Features & Technical Highlights

- **End-to-End Workflow:** Covers the entire ML lifecycle, from raw data ingestion to model deployment and interpretation.
- **Physics-Informed Feature Engineering:** Generates over 145 features from chemical formulas using `matminer` and `jarvis-tools`.
- **Multi-Framework Modeling:** Trains and evaluates models using XGBoost, TensorFlow, and PyTorch, allowing for robust comparison.
- **Advanced Interpretability with SHAP:** Employs SHAP to provide transparent, physics-based explanations for the XGBoost model's predictions.
- **Unified Prediction Pipeline:** A single, powerful script (`scripts/predict.py`) runs inference for any trained model, streamlining the prediction process.
- **Containerized Deployment:** Includes FastAPI endpoints and Dockerfiles for easy deployment of the models as a web service.
- **Modular & Reproducible Code:** Core logic is organized into a reusable `src` directory, following professional software engineering best practices.

## Installation and Setup

### Prerequisites
- Python 3.9+
- Conda for environment management
- Docker and Docker Compose (for API deployment)

### Installation Steps
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/acfdavis/Thermal-Conductivity-ML.git
    cd Thermal-Conductivity-ML
    ```
2.  **Create and activate the Conda environment:**
    The repository includes a `requirements.txt` file for easy setup.
    ```bash
    conda create -n thermal_env python=3.9
    conda activate thermal_env
    pip install -r requirements.txt
    # You may need to install deep learning libraries separately
    pip install tensorflow torch
    ```
3. **Set up API Key (Optional but Recommended):**
    Feature generation can be enhanced by querying the Materials Project database.
    - Get a key from the [Materials Project Dashboard](https://materialsproject.org/dashboard).
    - Create a file named `.env` in the project root and add your API key:
      ```
      MAPI_KEY=your_api_key_here
      ```

## Usage

The project workflow is managed through scripts for featurization, training, and prediction.

### 1. Data Featurization (If needed)

If the processed data file (`data/processed/featurized.parquet`) does not exist, run the featurization script first.
```bash
# This step is based on the original project notebooks.
# Ensure you have a script to generate the featurized data.
```

### 2. Model Training

Train models using the framework-specific scripts. This will save the trained model and necessary artifacts (scaler, feature list) to `models/`, `models_tf/`, or `models_torch/`.

```bash
# Train the final XGBoost model
python scripts/train_xgb.py

# Train the TensorFlow baseline
python scripts/train_tf.py

# Train the PyTorch baseline
python scripts/train_torch.py
```

### 3. Predict from a CSV File

Use the unified `predict.py` script to make predictions on new materials.

1.  **Prepare an input CSV file** (e.g., `data/example_input.csv`) with a `formula` column.
2.  **Run the prediction script**, specifying the framework and model paths.

**XGBoost:**
```bash
python scripts/predict.py `
    --framework xgboost `
    --model models/tuned_xgboost_model.joblib `
    --scaler models/scaler.joblib `
    --input data/example_input.csv `
    --output data/predictions_xgb.csv
```

**TensorFlow:**
```bash
python scripts/predict.py `
    --framework tensorflow `
    --model models_tf/tf_model.keras `
    --scaler models_tf/scaler.joblib `
    --features models_tf/feature_list.txt `
    --input data/example_input.csv `
    --output data/predictions_tf.csv
```

**PyTorch:**
```bash
python scripts/predict.py `
    --framework torch `
    --model models_torch/torch_model.pt `
    --scaler models_torch/scaler.joblib `
    --features models_torch/feature_list.json `
    --input data/example_input.csv `
    --output data/predictions_torch.csv
```
*(Note: Examples use PowerShell backticks (`` ` ``) for line continuation. Use backslashes (`\`) on Linux/macOS.)*

## Model Performance & Analysis

We evaluated the three models on a set of test materials with known experimental values. The results highlight the different strengths and weaknesses of each approach.

| Formula | XGBoost (W/mK) | TensorFlow (W/mK) | PyTorch (W/mK) | Experimental (Approx.) |
| :--- | :--- | :--- | :--- | :--- |
| **GaN** | 80.3 | **273.3** | 106.0 | ~130-230 |
| **SiC** | **123.9** | 178.6 | 109.3 | >120 |
| **Si** | **99.9** | 29.1 | 82.3 | ~150 |
| **SrTiO3**| **8.2** | 3.1 | 4.5 | ~11 |
| **Bi2Te3**| 3.9 | 2.4 | **3.1** | ~1.5-3 |
| **SiO2** | 3.9 | 3.7 | **4.8** | ~1-10 |

*(Best performing model for each material is in **bold**)*

### Key Insights
- **No Single Best Model:** The best model choice depends on the material class of interest.
- **XGBoost** is the most consistent all-around performer.
- **TensorFlow** shows a remarkable ability to predict the very high conductivity of GaN.
- **PyTorch** performs similarly to XGBoost and is most accurate for the thermoelectric material Bi2Te3.

## API Deployment with Docker

The project includes containerized FastAPI services for each model. The `Makefile` provides convenient shortcuts for building and running these services.

```bash
# Build and run the containerized APIs in the background
make up

# Query the running APIs with an example formula (e.g., SiC)
make curl-tf
make curl-torch

# Stop and remove the containers
make down
```

## Project Structure

```
Thermal-Conductivity-ML/
├── api/                    # FastAPI application files
├── data/
│   ├── raw/
│   └── processed/
├── models/                 # Saved XGBoost model
├── models_tf/              # Saved TensorFlow model and artifacts
├── models_torch/           # Saved PyTorch model and artifacts
├── notebooks/              # Jupyter notebooks for exploration and analysis
├── scripts/                # Training and prediction scripts
│   ├── train_xgb.py
│   ├── train_tf.py
│   ├── train_torch.py
│   └── predict.py          # Unified prediction script
├── src/                    # Reusable source code
│   └── predictor.py        # Core prediction pipeline logic
├── tests/                  # Integration and unit tests
├── .env
├── Dockerfile.tf
├── Dockerfile.torch
├── docker-compose.yml
├── Makefile
```
