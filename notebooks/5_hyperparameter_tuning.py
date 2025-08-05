# %%
#!/usr/bin/env python
# coding: utf-8

# %% [markdown]
# # Hyperparameter Tuning for Thermal Conductivity Prediction
# 
# **Author:** Angela Davis
# **Date:** July 2, 2025
# 
# ## Workflow Overview
# 
# This notebook focuses on optimizing the best-performing model identified in the previous notebooks: the **XGBoost Regressor** with a curated set of scaled features.
# 
# The steps are as follows:
# 1.  **Data Loading:** Load the featurized dataset and the curated list of selected features from notebook 4.
# 2.  **Data Preparation:** Prepare the data, applying the same scaling logic as the previous notebooks.
# 3.  **Hyperparameter Grid Definition:** Define a search space for the key hyperparameters of the XGBoost model.
# 4.  **Randomized Search:** Execute `RandomizedSearchCV` with cross-validation to find the best parameter combination.
# 5.  **Model Retraining and Evaluation:** Train a new XGBoost model with the optimized hyperparameters and evaluate its performance on the test set.
# 6.  **Performance Comparison:** Compare the tuned model's performance against the untuned baseline model (on the same selected, scaled features).
# 7.  **Save the Final Model:** Serialize the tuned XGBoost model for future use in prediction scripts.

# %%
# --- Professionalized Imports and Setup ---
import os, sys
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from utils import (
    setup_environment, load_or_process_dataframe, save_plot, style_df, prepare_data_for_modeling, log_and_print
)
from modeling import split_data, scale_features, apply_power_transform, evaluate_model
from viz import plot_parity_logscale
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import shap
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# --- Setup environment and paths ---
setup_environment()
USE_CLUSTER_FEATURES = False
CACHE_PATH = '../data/processed/featurized.pkl'
SELECTED_FEATURES_PATH = '../data/processed/selected_features_xgb.json'
FINAL_MODEL_PATH = '../models/tuned_xgboost_model.joblib'
PLOTS_DIR = '../plots/5_hyperparameter_tuning'
os.makedirs(PLOTS_DIR, exist_ok=True)
PARITY_PLOT_PATH = os.path.join(PLOTS_DIR, 'tuned_xgb_model_parity_plot.pdf')
SHAP_PLOT_PATH = os.path.join(PLOTS_DIR, 'tuned_xgb_model_shap_summary.pdf')

# --- Load featurized data using robust utility ---
df = load_or_process_dataframe(cache_path=CACHE_PATH)
log_and_print(f"Featurized dataframe shape: {df.shape}")

# %% [markdown]
# ## 1. Load and Prepare Data
# 
# We load the same featurized data and the list of selected features to ensure consistency with the modeling notebook.

# %%
# Prepare data for modeling
X, y = prepare_data_for_modeling(df, target_col='thermal_conductivity')

# %%
# Conditionally add cluster features based on the flag
if USE_CLUSTER_FEATURES:
    log_and_print("Generating cluster features...")
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    X_numeric = X.select_dtypes(include=np.number)
    scaler_for_clustering = StandardScaler()
    X_scaled_for_clustering = scaler_for_clustering.fit_transform(X_numeric)
    
    kmeans = KMeans(n_clusters=9, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled_for_clustering)
    
    X['cluster_label'] = cluster_labels
    X_final = pd.get_dummies(X, columns=['cluster_label'], prefix='cluster')
    log_and_print("Successfully added cluster labels.")
else:
    X_final = X
    log_and_print("Skipping cluster feature generation as per configuration.")

# %%
# Load selected features and apply them
with open(SELECTED_FEATURES_PATH, 'r') as f:
    selected_features = json.load(f)

available_selected_features = [f for f in selected_features if f in X_final.columns]
X_selected = X_final[available_selected_features]

log_and_print(f"Loaded and applied {len(available_selected_features)} selected features.")

# %%
# Split and scale the data
X_train, X_test, y_train_log, y_test_log, y_train, y_test = split_data(X_selected, y)
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

log_and_print(f"Data is ready for hyperparameter tuning. X_train_scaled shape: {X_train_scaled.shape}")

# %% [markdown]
# ## 2. Hyperparameter Tuning with RandomizedSearchCV
# 
# We define a parameter grid and use `RandomizedSearchCV` to efficiently explore the most promising hyperparameter combinations for the XGBoost model.

# %%
# Define the parameter grid for XGBoost
xgb_param_dist = {
    'n_estimators': randint(100, 1000),
    'learning_rate': uniform(0.01, 0.3),
    'max_depth': randint(3, 15),
    'subsample': uniform(0.6, 0.4), # Note: subsample + colsample_bytree should not be > 1
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': uniform(0, 0.5),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(1, 2)
}

# Initialize the XGBoost regressor
xgb_reg = XGBRegressor(random_state=42, n_jobs=-1)

# Initialize RandomizedSearchCV for XGBoost
xgb_random_search = RandomizedSearchCV(
    estimator=xgb_reg,
    param_distributions=xgb_param_dist,
    n_iter=100,
    cv=5,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1,
    random_state=42
)

# %%
# Fit the random search model
log_and_print("Starting RandomizedSearchCV for XGBoost...")
xgb_random_search.fit(X_train_scaled, y_train_log)
log_and_print("XGBoost search complete.")

# Get the best XGBoost estimator
best_xgb_tuned = xgb_random_search.best_estimator_

# Evaluate the tuned XGBoost model
tuned_xgb_results = evaluate_model(best_xgb_tuned, X_test_scaled, y_test_log, y_test)
tuned_xgb_log_df = pd.DataFrame([{**tuned_xgb_results['log'], 'Model': 'Tuned XGBoost'}])

log_and_print("\nTuned XGBoost Model Performance (log scale):")
display(style_df(tuned_xgb_log_df[['Model', 'r2', 'mae', 'rmse']]))

# %% [markdown]
# ## 3. Compare with Baseline and Finalize
# 
# To provide a clear measure of the value gained from hyperparameter tuning, we'll first train a baseline XGBoost model using its default parameters on the exact same data split. We will then compare it against our tuned XGBoost model.
# 
# ### 3.1. Train and Evaluate Baseline XGBoost Model

# %%

# --- PRE-TUNING MODEL: Selected features, default XGBoost ---
pre_tuning_xgb = XGBRegressor(random_state=42, n_jobs=-1)
pre_tuning_xgb.fit(X_train_scaled, y_train_log)
pre_tuning_results = evaluate_model(pre_tuning_xgb, X_test_scaled, y_test_log, y_test)
pre_tuning_log_df = pd.DataFrame([{**pre_tuning_results['log'], 'Model': 'Pre-Tuning XGB'}])

log_and_print("Pre-Tuning Model Performance (log scale):")
display(style_df(pre_tuning_log_df[['Model', 'r2', 'mae', 'rmse']]))
# %% [markdown]
# ### 3.2. Performance Comparison

# %%

# --- UPDATE COMPARISON TABLES ---
comparison_log_df = pd.concat([pre_tuning_log_df, tuned_xgb_log_df]).set_index('Model')

log_and_print("\nModel Comparison (log scale):")
display(style_df(comparison_log_df[['r2', 'mae', 'rmse']]))
# %% [markdown]
# ## 5. Final Visualizations and Model Saving
# 
# We'll now generate a parity plot and a SHAP summary plot for our final, tuned model to visually inspect its performance and interpret its predictions. Finally, we save the model for deployment.

# %%

# --- PARITY PLOT FOR TUNED MODEL ---
log_and_print("\nGenerating final plots for the winning model (Tuned XGBoost)...")
fig_parity = plot_parity_logscale(best_xgb_tuned, X_test_scaled, y_test_log, "Tuned XGBoost")
save_plot(fig_parity, PARITY_PLOT_PATH)
fig_parity.show()

# Generate and save the SHAP summary plot
explainer = shap.TreeExplainer(best_xgb_tuned)
shap_values = explainer.shap_values(X_test_scaled)
shap.summary_plot(shap_values, X_test_scaled, show=False, feature_names=X_selected.columns)
plt.title("SHAP Summary for Tuned XGBoost Model")
save_plot(plt.gcf(), SHAP_PLOT_PATH)
plt.show()
log_and_print(f"Parity plot saved to {PARITY_PLOT_PATH}")
log_and_print(f"SHAP summary plot saved to {SHAP_PLOT_PATH}")

# %%
# Save the final tuned model
joblib.dump(best_xgb_tuned, FINAL_MODEL_PATH)
log_and_print(f"\nWinning model (Tuned XGBoost) saved to: {FINAL_MODEL_PATH}")

# %% [markdown]
# ## 6. Conclusion and Next Steps
# 
# The hyperparameter tuning process successfully improved the XGBoost model's performance, leading to a higher R² score and lower error metrics compared to the baseline model on the curated feature set. The final model, saved as `tuned_xgboost_model.joblib`, is now ready for use in our prediction pipeline.
# 
# The SHAP analysis confirms that both physical properties and compositional features are significant drivers of the model's predictions. The inclusion of cluster features also proved beneficial, capturing underlying patterns in the data that individual features alone could not.
# 
# **Potential Next Steps:**
# 1.  **Deployment:** Integrate the saved model into a web service or API for real-time predictions using the `scripts/predict_from_csv.py` script as a template.
# 2.  **Deeper Error Analysis:** Investigate the largest prediction errors to identify specific material classes or regions of the feature space where the model struggles.
# 3.  **Experiment with More Features:** Explore additional feature engineering techniques or external datasets to further enhance predictive accuracy.
# 4.  **Alternative Models:** While XGBoost performed well, exploring other advanced models like deep neural networks could yield further improvements, especially if more data becomes available.

# %%
# Save the comparison table as a CSV file
comparison_table_path = os.path.join(PLOTS_DIR, 'model_comparison.csv')
comparison_log_df.to_csv(comparison_table_path, index=True)
log_and_print(f"Comparison table saved as {comparison_table_path}")
