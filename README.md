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

## Example Predictions vs. Literature Values

The table below compares model predictions (room temperature, ~300 K, 1 atm) for a small illustrative set of well-studied materials against typical literature thermal conductivity values. Literature values are approximate ranges; actual values depend on crystal quality, microstructure, stoichiometry, porosity, and anisotropy.

| Material | Literature κ (W/m·K, ~300 K) | Source (typical) | Model Prediction (W/m·K) | Abs Error | Rel Error (%) | Notes |
|----------|------------------------------|------------------|--------------------------|-----------|---------------|-------|
| SiO₂ (fused) | 1.3–1.5 (amorphous) / 6–7 (α-quartz along a) | CRC / materials handbooks | 3.908 | +2.4 vs fused | +160% (fused ref) | Overpredicts low-κ amorphous; lacks microstructural/phase detail |
| Bi₂Te₃ | 1.5–1.6 (polycryst.) | Thermoelectric literature | 3.984 | +2.4 | +150% | Overpredict; complexity & strong anharmonic scattering not fully captured |
| GaN | 130–230 (bulk single crystal) | Wide bandgap semiconductor data | 80.262 | −50 to −150 | −38% (vs 130) | Underpredict; missing defect / crystal quality indicators |
| SiC (poly / 4H) | 120–270 (4H, poly) | Semiconductor data | 123.917 | Near lower bound | ~0% (vs 124) | Reasonable; captures order of magnitude |
| Si (single crystal) | 148–150 | Standard reference | 99.886 | −48 | −32% | Underpredict (phonon–phonon & isotope scattering under-modeled) |
| SrTiO₃ | 10–12 | Perovskite oxide data | 8.178 | −2 | −18% | Slight underprediction; trend directionally correct |

### Discussion

Strengths:

- Provides correct relative ordering separating very low (Bi₂Te₃, SiO₂) vs high (SiC, Si) vs intermediate (SrTiO₃) κ classes, except for magnitude inflation in the very low regime.
- Captures that SiC > Si and both >> typical thermoelectrics/oxides.
- Achieves errors within the same order of magnitude for all samples (no catastrophic failures).

Limitations observed:

- Systematically overestimates ultra–low conductivity materials (SiO₂ amorphous, Bi₂Te₃) and underestimates some high conductivity semiconductors (GaN, Si). This suggests the model is not fully capturing dominant phonon scattering mechanisms (point defects, anisotropy, microstructure, grain boundaries) that suppress or enhance κ.
- Input features rely primarily on composition-derived statistical descriptors plus a subset of database-derived structural/elastic properties. Missing or sparse descriptors (e.g. Grüneisen parameter, phonon group velocity, defect concentrations, crystallinity, isotopic purity) limit absolute accuracy.

Model usefulness:

- Effective for rapid screening and prioritization: narrows candidate sets by conductivity class (low / moderate / high) before more expensive first-principles or experimental evaluation.
- Supports materials design workflows where relative ranking matters more than exact absolute κ.
- Acts as a feature importance interpreter (via SHAP) to highlight which elemental or structural factors drive predicted trends.

Recommended future enhancements:

1. Integrate phonon-informed proxies (e.g. predicted Debye temperature, average sound velocity, Grüneisen parameter) where available.
2. Add structural complexity metrics (atoms per primitive cell, mass variance) and defect / vacancy descriptors if datasets permit.
3. Calibrate a post-hoc correction model (e.g. isotonic regression) to debias systematic over/underestimation regions.
4. Stratify training by material class (oxides, thermoelectrics, wide-bandgap semiconductors) and apply class-conditional models or multi-task learning.
5. Incorporate uncertainty estimation (e.g. Monte Carlo dropout or ensemble variance) to communicate confidence alongside point predictions.

> Disclaimer: Literature values are broad averages; for rigorous benchmarking, compile a curated reference dataset with matched measurement conditions (crystal phase, orientation, purity, density). The simple comparison here is illustrative only.

## Handling Missing External Data

If individual formulas lack JARVIS or Materials Project matches:
- Missing feature columns are imputed (per-batch median, fallback 0) prior to scaling.
- A log message lists imputed columns.
- Rows are not dropped by default to preserve user input coverage.

## Polymorph Alignment Logic

When space_group is supplied:
1. Materials Project query attempts to select an entry whose symmetry.space_group_number matches the user value.
2. If no match exists, it falls back to the lowest energy entry (same behavior as before).
3. crystal_structure is derived from the user space group (not the fallback proxy).
4. A mismatch flag (internal) allows future auditing (not all flags are persisted in the final CSV).

Limitations:
- If the model was originally trained without explicit polymorph differentiation, changing space_group may yield modest or no prediction shifts.
- Extreme materials (e.g., BAs, very low κ layered anisotropic compounds) may be outside the original training distribution; treat results as qualitative.

## Recommended Best Practices

- Always supply space_group when targeting a specific polymorph (diamond vs graphite, rutile vs anatase, etc.).
- Use --skip-jarvis only when network/data access is constrained; otherwise keep full feature fidelity.
- For benchmarking, compare predicted vs literature κ on a curated, diverse set (metals, high κ covalent, low κ thermoelectrics, perovskites,

