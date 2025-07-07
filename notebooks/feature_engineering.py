# %%
#!/usr/bin/env python
# coding: utf-8

# %%
# ## Workflow Overview
# 
# This notebook follows these steps:
# 1. **Data Loading & Caching:** Load and clean raw data, cache featurized results for efficiency.
# 2. **Feature Engineering:** Apply Matminer and domain-specific featurizers, handle missing values, and engineer robust features.
# 3. **Exploratory Data Analysis (EDA):** Visualize feature distributions, missingness, and correlations.
# 4. **Baseline Modeling:** Train a baseline model using all features.
# 5. **Feature Selection:** Drop features with high missingness, low variance, or high correlation.
# 6. **Model Retraining & Comparison:** Retrain the model and compare performance before and after feature selection.
# 7. **Interpretability:** Use SHAP to interpret model predictions and feature importance.
# 
# *Random seeds are set for reproducibility where possible. All code is modular and leverages the project’s `src/` package structure.*

# %%
# # Feature Engineering and Model Interpretation for Thermal Conductivity Prediction
# 
# **Author:** Angela Davis
# **Date:** June 30, 2025
# 
# This notebook demonstrates a robust, recruiter-ready workflow for predicting the thermal conductivity of inorganic materials. It covers data loading, feature engineering, model training, feature selection, and interpretability using SHAP. The workflow emphasizes best practices in reproducibility, modularity, and scientific rigor.

# %%
# In[ ]:

# %%

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# %%
import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# %%
# Add src directory to path
module_path = os.path.abspath(os.path.join('..', 'src'))
if module_path not in sys.path:
    sys.path.append(module_path)

# %%
# Correct imports to point to the scripts, not the directories
from data_wrangling import load_and_merge_data, impute_and_clean_data
from feature_builder import featurize_data
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

# %%
# Load environment variables
load_dotenv(dotenv_path='../.env')

# %%
# Path to cache file (ensure this is correct)
cache_path = '../data/processed/featurized.csv'

# %%
# Try to load cached featurized data
from utils import load_cached_dataframe, cache_dataframe

# %%
df_featurized = load_cached_dataframe(cache_path)
if df_featurized is None:
    print('No cache found. Cleaning and featurizing data...')
    df_raw = load_and_merge_data()
    df_clean = impute_and_clean_data(df_raw)
    df_featurized = featurize_data(df_clean, composition_col='formula')
    cache_dataframe(df_featurized, cache_path)
    print('Featurized data cached.')
else:
    print('Loaded featurized data from cache.')

# %%
# Prepare data for modeling
X, y = prepare_data_for_modeling(df_featurized, target_col='thermal_conductivity')

# %%
print("Data loaded and featurized successfully.")
X.head()

# %%

# ## Exploratory Data Analysis (EDA) and Feature Importance
# 
# We begin by visualizing feature correlations and missingness, and by examining feature importances from a baseline Random Forest model. This helps guide feature selection and engineering decisions.

# %%
# In[2]:

# %%

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from modeling import train_random_forest

# %%
# 1. Correlation matrix (numeric columns only)
numeric_cols = df_featurized.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
corr = numeric_cols.corr()
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()

# %%
# 2. Random Forest feature importances using modeling.py
X = df_featurized.select_dtypes(include='number').drop(columns=['thermal_conductivity'], errors='ignore')
y = df_featurized['thermal_conductivity']
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
rf = train_random_forest(X_train, y_train_log)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

# %%
plt.figure(figsize=(10, 6))
importances.head(20).plot(kind='bar')
plt.title('Top 20 Feature Importances (Random Forest)')
plt.ylabel('Importance')
plt.show()

# %%
# 3. Check for features with many missing values or low variance
missing = df_featurized.isnull().mean().sort_values(ascending=False)
print('Features with most missing values:')
print(missing.head(10))

# %%
low_var = X.std().sort_values()
print('Features with lowest variance:')
print(low_var.head(10))

# %%

# ## Feature Selection, Model Retraining, and Interpretability
# 
# Now that we've visualized feature importances and missingness, we:
# - Drop features with high missingness (>30%) or very low variance
# - Drop one of each pair of highly correlated features (|corr| > 0.85)
# - Retrain the model and compare performance
# - Use SHAP to interpret the model and visualize feature impact
# 
# This demonstrates a robust feature engineering workflow.

# %%
# In[3]:

# %%

# 1. Feature selection: drop high-missing, low-variance, and highly correlated features
missing_thresh = 0.3
low_var_thresh = 1e-5
corr_thresh = 0.85

# %%
# Drop features with >30% missing values
missing = df_featurized.isnull().mean()
features_to_drop = missing[missing > missing_thresh].index.tolist()

# %%
# Drop features with very low variance
X = df_featurized.select_dtypes(include='number').drop(columns=['thermal_conductivity'], errors='ignore')
low_var = X.std()
features_to_drop += low_var[low_var < low_var_thresh].index.tolist()

# %%
# Drop one of each pair of highly correlated features
corr_matrix = X.drop(columns=features_to_drop, errors='ignore').corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = [column for column in upper.columns if any(upper[column] > corr_thresh)]
features_to_drop += high_corr

# %%
# Remove duplicates
features_to_drop = list(set(features_to_drop))

# %%
print(f"Dropping {len(features_to_drop)} features: {features_to_drop}")

# %%
X_selected = X.drop(columns=features_to_drop, errors='ignore')

# %%
# Retrain model and compare performance
from modeling import evaluate_model
# Split and scale the data (now using X_selected)
X_train, X_test, y_train_log, y_test_log, y_train, y_test = split_data(X_selected, y)
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
rf_selected = train_random_forest(X_train, y_train_log)
results_selected = evaluate_model(rf_selected, X_test, y_test_log, y_test)

# %%
# Also get baseline results for comparison (assume baseline_results already computed in previous cell)
try:
    baseline_results
except NameError:
    # Compute baseline if not already available
    X_base = df_featurized.select_dtypes(include='number').drop(columns=['thermal_conductivity'], errors='ignore')
    y_base = df_featurized['thermal_conductivity']
    X_train_b, X_test_b, y_train_log_b, y_test_log_b, y_train_b, y_test_b = split_data(X_base, y_base)
    X_train_scaled_b, X_test_scaled_b, scaler_b = scale_features(X_train_b, X_test_b)
    rf_baseline = train_random_forest(X_train_b, y_train_log_b)
    baseline_results = evaluate_model(rf_baseline, X_test_b, y_test_log_b, y_test_b)

# %%


# %%

# ## Model Performance Comparison
# 
# The table below compares the model’s predictive performance before and after feature selection. Metrics are reported on the log-transformed target for consistency with the modeling workflow.

# %%
# In[4]:

# %%

import pandas as pd
from IPython.display import display, HTML

# %%
def get_metrics(results):
    if 'log' in results:
        return [results['log']['r2'], results['log']['mae'], results['log']['rmse']]
    else:
        return [results.get('r2', float('nan')), results.get('mae', float('nan')), results.get('rmse', float('nan'))]

# %%
metrics_df = pd.DataFrame({
    'R²': [get_metrics(baseline_results)[0], get_metrics(results_selected)[0]],
    'MAE': [get_metrics(baseline_results)[1], get_metrics(results_selected)[1]],
    'RMSE': [get_metrics(baseline_results)[2], get_metrics(results_selected)[2]],
}, index=['Before\nFeature Selection', 'After\nFeature Selection'])

# %%
# Use Styler to improve table appearance
styled = (
    metrics_df.round(3)
    .style.set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center'), ('white-space', 'pre-line'), ('min-width', '110px')]},
        {'selector': 'td', 'props': [('text-align', 'center'), ('min-width', '60px')]},
    ])
    .set_properties(**{'text-align': 'center'})
    .set_caption("<b>Model Performance Comparison (log-scaled)</b>")
)
display(HTML(styled.to_html()))
_ = None

# %%

# ## Model Interpretability with SHAP
# 
# We use SHAP (SHapley Additive exPlanations) to interpret the impact of each feature on the model’s predictions. The bar plot shows global feature importance, while the beeswarm plot visualizes the distribution of feature impacts across the test set. The dependence plot highlights the effect of the most important feature.

# %%
# In[5]:

# %%

# 2. SHAP interpretability for feature impact
import shap

# %%
# Use TreeExplainer for Random Forest
explainer = shap.TreeExplainer(rf_selected)
shap_values = explainer.shap_values(X_test)

# %%
# Summary plot (global feature importance)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('SHAP Feature Importance (Bar)')
plt.show()

# %%
# Beeswarm plot (distribution of impacts)
shap.summary_plot(shap_values, X_test, show=False)
plt.title('SHAP Feature Importance (Beeswarm)')
plt.show()

# %%
# Optional: Dependence plot for top feature
top_feature = X_selected.columns[np.argmax(np.abs(shap_values).mean(axis=0))]
shap.dependence_plot(top_feature, shap_values, X_test, show=False)
plt.title(f'SHAP Dependence Plot: {top_feature}')
plt.show()

# %%

# In[ ]:

# %%


# %%


# %%
# In[ ]:

# %%


# %%


# %%
# ## Summary & Interpretation
# 
# - **Feature selection** led to a meaningful improvement in model performance. The R² increased from 0.799 to 0.834, MAE decreased from 0.193 to 0.180, and RMSE improved from 0.429 to 0.390 (log-scaled). This demonstrates that removing high-missing, low-variance, and highly correlated features not only simplified the model but also enhanced its predictive accuracy and generalizability.
# 
# - **Key features** identified by SHAP analysis include:
#     - `MagpieData maximum GSvolume_pa`
#     - `temperature`
#     - `MagpieData minimum Number`
#     - `MagpieData mode GSvolume_pa`
#     - `MagpieData avg_dev NdValence`
#     - `MagpieData mean NpValence`
#     - `MagpieData mode MendeleevNumber`
#     - `MagpieData maximum CovalentRadius`
#     - `MagpieData mean NpUnfilled`
#     - `MagpieData mean NUnfilled`
#     - `frac s valence electrons`
#     - ...and others
#   
#   These features are physically meaningful and align with known structure-property relationships in inorganic solids. For example, atomic volume, temperature, and electronic structure descriptors are well-established drivers of thermal transport.
# 
# - **Interpretation:**
#     - The improved model performance after feature selection suggests that the workflow effectively mitigates noise and redundancy, which is critical for robust property prediction in materials science.
#     - The SHAP results provide actionable insights for both further feature engineering and scientific interpretation, highlighting the importance of both compositional and structural descriptors.
# 
# **Next steps:**
# - Explore advanced algorithms (e.g., XGBoost, SVR) and hyperparameter optimization to further boost performance.
# - Integrate additional structure-based or domain-specific features, especially if 3D structural data becomes available.
# - Validate the model using cross-validation and, if possible, additional datasets to ensure generalizability.
# 
# This notebook exemplifies a best-practice, interpretable, and modular ML workflow for materials property prediction, suitable for both research and industrial applications.
