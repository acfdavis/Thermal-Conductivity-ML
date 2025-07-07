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
