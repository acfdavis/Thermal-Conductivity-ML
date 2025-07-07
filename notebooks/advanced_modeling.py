# %%
#!/usr/bin/env python
# coding: utf-8

# %%
# # Predicting Thermal Conductivity of Inorganic Materials
# 
# **Objective:** Develop a machine learning workflow to predict the thermal conductivity of inorganic materials based on their chemical formula and temperature.
# 
# **Workflow:**
# 1.  **Data Loading & Preprocessing:** Load data from multiple sources, clean it, and combine it into a single dataset.
# 2.  **Feature Engineering:** Generate a comprehensive set of features for each material using `matminer`.
# 3.  **Modeling:** Train and evaluate a Random Forest Regressor model.
# 4.  **Evaluation:** Assess model performance using R² and MSE metrics and visualize the results.
# 
# This notebook demonstrates a clean, modular, and reproducible machine learning pipeline.

# %%
# # Thermal Conductivity of Inorganic Materials – Data Integration and Modeling
# 
# This notebook analyzes the Citrine thermal conductivity dataset to develop interpretable models for predicting thermal conductivity of inorganic compounds. We integrate domain knowledge in materials science (feature engineering), ensure clean, reproducible data workflows (caching, merging), and demonstrate data exploration and modeling with publication-quality plots.
# 
# Efficient thermal management is critical in electronics and energy applications. For example, cooling can account for \~40% of a data center’s energy consumption. Materials with high thermal conductivity enable faster heat transfer and more efficient energy storage, such as in [solar thermal systems](https://thermtest.com/thermal-methods-in-thermal-energy-storage).
# 
# > Example: Improving cooling efficiency by just 10% in a 1 MW data center could save on the order of \$30–50k per year in electricity, yielding a rapid ROI on material development and deployment ([Energy Consumption in Data Centers](https://www.boydcorp.com/blog/energy-consumption-in-data-centers-air-versus-liquid-cooling.html)).
# 
# ### Key Benefits:
# 
# * Enhanced thermal conduction improves electronics cooling, device reliability, and allows higher power densities.
# * In renewable energy, faster heat transfer enables more efficient energy capture and reduced losses.
# * Reduced cooling load directly saves energy and costs.
# * Informatics-driven discovery shortens development cycles and amplifies material impact.
# 
# ---
# 
# ### Why Use AI for Predicting Thermal Conductivity?
# 
# AI and machine learning have emerged as powerful tools in materials science, especially for predicting properties that are expensive or time-consuming to measure experimentally. In this notebook, we demonstrate how machine learning models—when paired with domain-specific feature engineering and robust datasets—can accelerate the discovery of high-performance materials.
# 
# **Benefits of AI in Materials Property Prediction:**
# 
# * **Efficiency:** Reduces the need for costly experiments or simulations by learning from existing data.
# * **Scalability:** Once trained, models can screen thousands of compounds in seconds.
# * **Insight:** Helps identify which material features (e.g., atomic mass, bonding type) most influence thermal transport.
# * **Generalizability:** The workflow in this notebook is adaptable to predict other properties such as elastic moduli, Seebeck coefficient, or bandgap, simply by substituting the target variable and adjusting input features.
# 
# **Scientific References:**
# 
# * [Ward et al., 2016](https://doi.org/10.1038/npjcompumats.2016.28): A general-purpose machine learning framework for predicting materials properties.
# * [Butler et al., 2018](https://doi.org/10.1038/s41586-018-0337-2): "Machine learning for molecular and materials science," *Nature*.
# 
# 
# 
# 

# %%
# # Data Loading, Cleaning, and Feature Gathering

# %%
# In[ ]:

# %%

try:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except NameError:
    pass  # Not running in IPython/Jupyter, so skip these lines

# %%
import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# %%
# Add src directory to path
import os
import sys

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# %%
# Correct imports to point to the scripts, not the directories
from data import load_and_merge_data, impute_and_clean_data
from features import featurize_data
from utils import prepare_data_for_modeling
from modeling import( 
    split_data,
    scale_features,
    train_xgboost, 
    train_random_forest,
    train_gradient_boosting,
    train_svr,
    evaluate_model,
)
from viz import (
    plot_model_comparison,
    plot_feature_importance,
    plot_parity_logscale,
    plot_residuals,
    plot_parity_grid,
    add_subplot_border
)
# %%
# Load the API key from environment variables
load_dotenv()
API_KEY = os.getenv("MP_API_KEY")
# %%
# Load environment variables
load_dotenv(dotenv_path='../.env')

# %%
# Define cache path
cache_path = '../data/processed/featurized.pkl'  # change to .pkl to match joblib format

# Load and process raw data only if cache is not present
df_raw = load_and_merge_data()# Ensure NIST data is loaded

#%%
df_clean = impute_and_clean_data(df_raw)

# Run featurization (with automatic cache loading/saving)
df_featurized = featurize_data(df_clean, composition_col='formula', cache_path=cache_path)

# Confirm cache status
print("Current working directory:", os.getcwd())
print("Cache path:", os.path.abspath(cache_path))
print("Cache exists:", os.path.exists(cache_path))


# %%
# Prepare data for modeling
X, y = prepare_data_for_modeling(df_featurized, target_col='thermal_conductivity')

# %%
print("Data loaded and featurized successfully.")
X.head()

# %%

# In[ ]:

# %%

# ---------------------------------------------------------------------------
# Assemble the combined dataset
# ---------------------------------------------------------------------------

# %%
# Split and scale the data
X_train, X_test, y_train_log, y_test_log, y_train, y_test = split_data(X, y)
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

# %%
print("Data split and scaled.")
print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)

# %%

# In[ ]:

# %%

# Dictionary to store models and their results
models = {}
results = {}
from modeling import (
    train_xgboost, 
    train_random_forest,
    train_gradient_boosting,
    train_svr,
    evaluate_model,
)

# %%
# Train and evaluate XGBoost
print("Training XGBoost...")
xgb_model = train_xgboost(X_train_scaled, y_train_log)
models['XGBoost'] = xgb_model
results['XGBoost'] = evaluate_model(xgb_model, X_test_scaled, y_test_log, y_test)
print("XGBoost training complete.")

# %%
# Train and evaluate Random Forest
print("\nTraining Random Forest...")
rf_model = train_random_forest(X_train_scaled, y_train_log)
models['Random Forest'] = rf_model
results['Random Forest'] = evaluate_model(rf_model, X_test_scaled, y_test_log, y_test)
print("Random Forest training complete.")

# %%
# Train and evaluate Gradient Boosting
print("\nTraining Gradient Boosting...")
gb_model = train_gradient_boosting(X_train_scaled, y_train_log)
models['Gradient Boosting'] = gb_model
results['Gradient Boosting'] = evaluate_model(gb_model, X_test_scaled, y_test_log, y_test)
print("Gradient Boosting training complete.")

# %%
# Train and evaluate SVR
print("\nTraining SVR...")
svr_model = train_svr(X_train_scaled, y_train_log)
models['SVR'] = svr_model
results['SVR'] = evaluate_model(svr_model, X_test_scaled, y_test_log, y_test)
print("SVR training complete.")

# %%
print("\n--- All models trained and evaluated ---")

# %%
# Display results for all models on the log-transformed scale
for model_name, result in results.items():
    print(f"\n--- {model_name} Evaluation (Log-Transformed Scale) ---")
    print(f"MAE: {result['log']['mae']:.4f}, RMSE: {result['log']['rmse']:.4f}, R²: {result['log']['r2']:.4f}")

# %%
# ### Model Performance Metrics
# %%
# Create a DataFrame to display the results
results_summary = {}
for model_name, result_data in results.items():
    results_summary[model_name] = result_data['log']
results_df = pd.DataFrame(results_summary).T

# Display the table
print("--- Model Performance Comparison (Log-Transformed Scale) ---")
display(results_df[['r2', 'mae', 'rmse']])

# %%
# Visualizations using updated, modular functions
print("\nGenerating visualizations...")

for model_name, model_obj in models.items():
    plot_parity_logscale(model_obj, X_test_scaled, y_test_log, model_name)
    plot_residuals(model_obj, X_test_scaled, y_test_log, model_name)
    plot_feature_importance(model_obj, X_train_scaled, X.columns, model_name)

plot_model_comparison(results)

fig = plot_parity_grid(models, X_test_scaled, y_test_log)
add_subplot_border(fig, row=1, col=1, rows=2, cols=2, color="#2A9D8F", width=2)
fig.show()
fig.write_image("../plots/parity_plot_grid.png", scale=2)
fig.write_image("../plots/parity_plot_grid.pdf", width=900, height=700)

# %%
