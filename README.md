# Predicting Thermal Conductivity with Machine Learning

This project presents a complete, end-to-end machine learning pipeline to predict the thermal conductivity of inorganic materials using only their chemical formula. It is designed to showcase a professional, modular, and reproducible workflow suitable for a materials informatics or data science portfolio.

The final XGBoost model achieves a **Test R² of 0.855** and a **Mean Absolute Error (MAE) of 0.35 W/mK** on a log-transformed scale, demonstrating high accuracy in predicting this complex material property.

## Key Features & Technical Highlights

- **End-to-End Workflow:** Covers the entire ML lifecycle, from raw data ingestion and cleaning to model deployment and interpretation.
- **Physics-Informed Feature Engineering:** Generates over 145 features from chemical formulas using `matminer` and `jarvis-tools`, capturing elemental properties, stoichiometry, and structural information.
- **Systematic Model Selection:** Rigorously compares multiple models (XGBoost, Random Forest, SVR) and preprocessing strategies to identify the optimal pipeline.
- **Advanced Interpretability with SHAP:** Employs SHAP (SHapley Additive exPlanations) to provide transparent, physics-based explanations for model predictions, identifying the most influential material features.
- **Modular & Reusable Code:** Core logic is organized into a reusable `src` directory, following professional software engineering best practices.
- **Automated Integration Testing:** Includes a `pytest` suite to ensure the prediction script runs reliably.
- **Reproducible Pipeline:** The entire workflow is captured in version-controlled notebooks, guaranteeing full reproducibility.

## Installation and Setup

### Prerequisites
- Python 3.9+
- Conda for environment management

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
    ```
3. **Set up API Key (Optional but Recommended):**
    Feature generation relies on the `matminer` library, which can query the Materials Project database for additional data. To enable this, you need a Materials Project API key.
    - Get a key from the [Materials Project Dashboard](https://materialsproject.org/dashboard).
    - Create a file named `.env` in the root of the project directory.
    - Add your API key to the file like this:
      ```
      MAPI_KEY=your_api_key_here
      ```

## Project Workflow

The project is structured as a sequence of notebooks, each handling a distinct stage of the machine learning pipeline.

1.  **`notebooks/1_eda.py`**: **Exploratory Data Analysis**
    -   Performs initial analysis of the dataset, including summary statistics, distribution plots, and correlation analysis to understand the data's structure and identify potential challenges.

2.  **`notebooks/2_clustering_and_pca.py`**: **Unsupervised Learning & Feature Engineering**
    -   Applies PCA for dimensionality reduction and K-Means clustering to uncover natural groupings within the materials data. These clusters are later used as engineered features.

3.  **`notebooks/3_modeling_and_feature_selection.py`**: **Model Comparison & Initial Feature Selection**
    -   Systematically trains and evaluates multiple regression models (XGBoost, Random Forest, etc.) on different feature sets.
    -   Performs an initial round of feature selection to identify the most promising model and feature combination.

4.  **`notebooks/4_model_comparison.py`**: **Advanced Feature Selection & SHAP Analysis**
    -   Takes the best-performing model (XGBoost) and applies more advanced feature selection techniques to create a minimal, high-performance feature set.
    -   Uses SHAP to analyze feature importance and model behavior.

5.  **`notebooks/5_hyperparameter_tuning.py`**: **Model Optimization & Finalization**
    -   Performs hyperparameter tuning on the final model and feature set using `RandomizedSearchCV` to maximize predictive performance. The final, tuned model and pre-processing scaler are saved for inference.

## How to Predict New Materials

You can predict the thermal conductivity of new materials using the `scripts/predict_from_csv.py` script.

1.  **Prepare an input CSV file.** Create a file (e.g., `my_materials.csv`) with a `formula` column containing the chemical formulas you want to predict.
    ```csv
    formula
    SiO2
    GaN
    Bi2Te3
    ```
2.  **Run the prediction script from your terminal:**
    ```bash
    python scripts/predict_from_csv.py --input-path my_materials.csv --output-path predictions.csv
    ```
    - The script will load the final model from `models/tuned_xgboost_model.joblib`.
    - It will generate features, apply the saved scaler, and make predictions.
    - The output will be saved to `predictions.csv` and will include the predicted thermal conductivity and an estimated error.

## How to Run Tests

The project includes an integration test to verify that the prediction script works correctly from end to end.

1.  **Make sure you have installed the dependencies** as described in the installation section.
2.  **Run the tests using `pytest`:**
    ```bash
    pytest
    ```
    The tests will run and confirm that the script can successfully load the model and generate predictions.

## Project Structure

```
Thermal-Conductivity-ML/
├── data/
│   ├── raw/                # Original raw datasets
│   └── processed/          # Cleaned, featurized, and intermediate data
├── models/                 # Saved and tuned final models
├── notebooks/              # Jupyter notebooks for each stage of the workflow
├── plots/                  # Saved plots and visualizations
├── scripts/                # Helper scripts (e.g., for prediction)
├── src/                    # Reusable source code for the pipeline
│   ├── data.py             # Data loading and cleaning functions
│   ├── features.py         # Feature engineering functions
│   ├── modeling.py         # Model training and evaluation functions
│   ├── viz.py              # Visualization functions
│   └── utils.py            # Shared utility functions
├── tests/                  # Integration and unit tests
├── .env                    # API keys (not version controlled)
├── requirements.txt        # Python dependencies
└── README.md               # Project overview
```
