# ML Conductivity Project

This project demonstrates a full machine learning pipeline for predicting thermal conductivity using scientific data. It is structured to highlight your skills in data processing, feature engineering, modeling, and reproducibility — ideal for portfolio review by technical recruiters or hiring managers.

## 📁 Project Structure

```
ml_conductivity_project/
├── data/
│   ├── raw/                # Original ThermoML or CSV data
│   └── processed/          # Cleaned and featurized data
├── notebooks/              # Lightweight notebooks for demo and visualization
├── src/                    # Modular and reusable pipeline code
│   ├── data.py             # Data loading and preprocessing
│   ├── features.py         # Feature engineering
│   ├── modeling.py         # Model training and evaluation
│   ├── viz.py              # Visualizations
│   └── utils.py            # Shared utility functions
├── requirements.txt        # Python dependencies
├── README.md               # Project overview
└── .gitignore              # Ignore large and generated files
```

## 🔍 Highlights

- Chunked and memory-safe loading for large datasets
- Feature engineering using domain-specific logic
- Modular code for reproducibility
- Lightweight demo notebooks (`.py` or `.ipynb`)
- Ready to deploy or extend for materials informatics applications

## 🚀 Getting Started

```bash
# Setup (Python >= 3.9 recommended)
pip install -r requirements.txt
```

## ✅ Running the Demo

Open `eda_and_clustering.py` in VS Code and run cells with `# %%` markers.
Or, run the cleaned notebooks using Jupyter Lab (for best performance).

---

© Your Name, 2025 — For demonstration and job application use only.
## 🔮 Predicting a New Material

To predict the thermal conductivity of a new material:

1. Prepare a CSV with the same structure as the training data. You can use this example:

```
formula,temperature,pressure,phase
SiO2,300,101325,solid
```

Save it as `data/example_input.csv`.

2. Run the prediction script:

```bash
python scripts/predict_from_csv.py --model outputs/final_model.pkl --input data/example_input.csv --output data/predicted_kappa.csv
```

The predictions will be saved in `data/predicted_kappa.csv`.

## 🛠️ Workflow Overview (2025 Refactor)

This project is a recruiter-ready, modular ML workflow for predicting thermal conductivity, with a focus on:
- **Modularity:** All data, feature, modeling, and visualization logic is in `src/` modules.
- **Reproducibility:** All notebooks/scripts use robust utility functions for environment setup, data loading/caching, plot saving, DataFrame styling, and logging.
- **Professional Outputs:** All plots and tables are recruiter/statistician-ready, with consistent, publication-quality style.
- **Pipeline Stages:**
  1. **EDA:** `notebooks/1_eda.py` — Exploratory data analysis, summary statistics, and target distribution.
  2. **Clustering & PCA:** `notebooks/2_clustering_and_pca.py` — Unsupervised clustering, PCA, and cluster analysis.
  3. **Feature Engineering:** `notebooks/3_feature_engineering.py` — Feature creation, selection, and curation (outputs `selected_features.json`).
  4. **Advanced Modeling:** `notebooks/4_advanced_modeling.py` — Model comparison, SHAP analysis, and decision summary.
  5. **Hyperparameter Tuning:** `notebooks/5_hyperparameter_tuning.py` — XGBoost tuning, final model/plot artifacts, and recruiter-friendly reporting.

## 🧰 Utility Functions

All notebooks/scripts use standardized utilities from `src/utils.py`:
- `setup_environment()`: Loads environment variables and sets plotting defaults.
- `load_cached_dataframe()`: Robustly loads cached data from Parquet, Pickle, or CSV.
- `save_plot()`: Saves Plotly or Matplotlib figures with consistent style.
- `style_df()`: Styles DataFrames for professional, readable tables.
- `log_and_print()`: Unified logging/printing for notebook/console.
- `prepare_data_for_modeling()`: Standard feature/target extraction and imputation.

## 📊 Plot & Table Artifacts

All key plots are saved as PDFs in organized subfolders under `plots/`, using `save_plot`. Tables in notebooks are styled with `style_df` for recruiter/statistician review.

## 🔄 How to Run the Workflow

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run each notebook in order:**
   - `notebooks/1_eda.py`
   - `notebooks/2_clustering_and_pca.py`
   - `notebooks/3_feature_engineering.py`
   - `notebooks/4_advanced_modeling.py`
   - `notebooks/5_hyperparameter_tuning.py`

   Or, open in Jupyter Lab/VS Code and run cell-by-cell.

3. **All outputs (plots, models, features) are saved for downstream use.**
