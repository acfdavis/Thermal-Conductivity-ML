#!/usr/bin/env python
# coding: utf-8

# %% [markdown]
# # Exploratory Data Analysis (EDA): Uncovering Material Clusters for Thermal Conductivity Prediction
#
# ## Executive Summary
#
# **Business Problem:** Predicting the thermal conductivity of new materials is a slow and expensive process, hindering the development of next-generation electronics and energy systems.
#
# **Our Solution:** This notebook demonstrates a data-driven approach to accelerate material discovery. By analyzing a dataset of 1,200 inorganic compounds, we use Exploratory Data Analysis (EDA) to understand the data's underlying structure and prepare it for advanced modeling.
#
# **Key Achievements & Skills Demonstrated:**
# - **Project Structure & Reproducibility:** Organized code into a modular `src` directory with utility functions for data processing, visualization, and environment setup, ensuring a clean and repeatable workflow.
# - **Data Featurization & Caching:** Transformed raw chemical formulas into over 100 quantitative features and implemented an efficient caching mechanism (`.parquet`) to accelerate subsequent data loading.
# - **Insightful Visualization:** Created clear, professional, and consistently styled visualizations to analyze feature distributions, correlations, and the target variable. This includes generating multi-page PDF reports for comprehensive analysis while keeping the notebook summary concise.
# - **Statistical Analysis:** Performed statistical tests to validate observations, such as using the Shapiro-Wilk test to confirm the necessity of a log transformation for the target variable.
#
# **Business Value:** This initial analysis is a critical first step in any data science project. It builds a solid foundation for a machine learning model that can predict thermal conductivity with high accuracy, drastically reducing R&D costs and speeding up time-to-market for new technologies. It showcases a robust and professional workflow for turning complex scientific data into actionable business intelligence.

# %% [markdown]
# ## 1. Environment Setup and Data Preparation
#
# Here, we import the necessary libraries and our custom utility functions from the `src` directory. This modular approach demonstrates best practices for structuring a Python project. We then load, clean, and featurize the raw data, transforming chemical formulas into a rich feature set suitable for machine learning. Caching is used to accelerate subsequent runs.

# %%
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
%matplotlib inline

# Ensure src directory is added to sys.path for modular imports
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from utils import (
    load_or_process_dataframe,
    setup_environment,
    save_plot,
    style_df,
    log_and_print,
    perform_normality_tests
)
from viz import (
    plot_numeric_histograms_paginated,
    plot_numeric_histograms_log_paginated,
    plot_tc_histograms
)

# --- Setup Environment & Define Paths ---
setup_environment()
PLOTS_DIR = '../plots/1_eda'
os.makedirs(PLOTS_DIR, exist_ok=True)

CACHE_PATH = '../data/processed/featurized.parquet'
HIST_PATH = os.path.join(PLOTS_DIR, 'eda_numeric_histograms.pdf')
HIST_LOG_PATH = os.path.join(PLOTS_DIR, 'eda_numeric_histograms_log.pdf')
HIST_TC_PATH = os.path.join(PLOTS_DIR, 'eda_tc_histograms.pdf')
CORR_MATRIX_PATH = os.path.join(PLOTS_DIR, 'eda_corr_matrix.pdf')

# --- Load and Featurize Data (with Caching) ---
df = load_or_process_dataframe(cache_path=CACHE_PATH)
log_and_print(f"Featurized dataframe shape: {df.shape}")
style_df(df.head())

# %% [markdown]
# ## 2. Exploratory Data Analysis (EDA)
#
# ### Feature Distributions
#
# We begin by visualizing the distributions of the numeric features. This helps us understand the data's characteristics and identify potential issues such as skewness or outliers that could impact model performance.
#
# To keep this notebook concise, we display only a **sample** of the histogram grids below (the first page of each set). The full set of histograms for all numeric features is saved as a multi-page PDF in the `plots/1_eda/` directory for detailed review.

# %%
# --- Generate and Save Multi-page PDFs for All Numeric Histograms ---

# Save paginated numeric histograms (linear scale)
figs_linear = plot_numeric_histograms_paginated(df, per_page=9)
with PdfPages(HIST_PATH) as pdf:
    for fig in figs_linear:
        pdf.savefig(fig)
        if fig.number > 1:
            plt.close(fig) # Close the figure after saving to avoid displaying it inline
log_and_print(f"Paginated numeric histograms PDF saved to {HIST_PATH}")

# %%
# Save paginated numeric histograms (log scale)
figs_log = plot_numeric_histograms_log_paginated(df, per_page=9)
with PdfPages(HIST_LOG_PATH) as pdf:
    for fig in figs_log:
        pdf.savefig(fig)
        if fig.number > 1:
            plt.close(fig) # Close the figure after saving to avoid displaying it inline
log_and_print(f"Paginated log-scale numeric histograms PDF saved to {HIST_LOG_PATH}")

# --- Display First Page Sample Inline ---
print("\nDisplaying a sample of the feature distributions (first page of 9 plots):")
figs_linear[0].suptitle('Sample of Numeric Feature Histograms (Linear Scale)', fontsize=16, y=1.02)
plt.show()

print("\nDisplaying a sample of the log-transformed feature distributions:")
figs_log[0].suptitle('Sample of Numeric Feature Histograms (Log Scale)', fontsize=16, y=1.02)
plt.show()

# %% [markdown]
# ### Target Variable: Thermal Conductivity
#
# Now, let's focus on the target variable, `thermal_conductivity`. We'll visualize its distribution on both original and log scales. Assessing and correcting for skewness in the target is a critical step for building robust predictive models.

# %%
# The plot_tc_histograms function from src/viz.py handles all styling.
fig_tc = plot_tc_histograms(df)
plt.savefig(HIST_TC_PATH, bbox_inches='tight')
log_and_print(f"Thermal conductivity histogram plot saved to {HIST_TC_PATH}")
plt.show()

# %% [markdown]
# **Observations & Insights:**
#
# *   **Right-Skewed Data:** The initial distributions show that many features, including the target variable `thermal_conductivity`, are heavily right-skewed. This is a common characteristic of physical property data.
# *   **Log Transformation:** Applying a log transformation (`log10`) to these skewed features results in distributions that are much closer to a symmetric, normal (Gaussian) distribution. This is particularly evident for `thermal_conductivity`, where the log-transformed version is more bell-shaped.
# *   **Modeling Implications:** Using these log-transformed features can lead to more stable and reliable performance in downstream modeling, especially for linear models, PCA, and other algorithms that benefit from normally distributed data. The side-by-side histograms clearly illustrate that the log-transformed target variable is a much better candidate for prediction.

# %% [markdown]
# ### Statistical Summary and Normality Assessment
#
# Let's quantify the observations from the plots by computing summary statistics and performing a formal normality test (Shapiro-Wilk) on the target variable before and after the log transformation. This adds statistical rigor to our visual analysis.

# %%
# Summary statistics for thermal conductivity (original and log scale)
tc = df['thermal_conductivity'].dropna()
tc_log = pd.Series(np.log10(tc[tc > 0]), name='log10_thermal_conductivity')

summary_stats = {
    'Original': tc.describe(),
    'Log10': tc_log.describe()
}
summary_df = pd.DataFrame(summary_stats)
summary_df.loc['skew'] = [tc.skew(), tc_log.skew()]
summary_df.loc['kurtosis'] = [tc.kurtosis(), tc_log.kurtosis()]

log_and_print("Summary Statistics for Thermal Conductivity:")
style_df(summary_df)

# %%
# Normality test (Shapiro-Wilk) for original and log-transformed target
from scipy.stats import shapiro
# Note: Shapiro-Wilk is reliable for n < 5000. We take a sample if data is larger.
shapiro_orig = shapiro(tc.sample(min(len(tc), 5000), random_state=42))
shapiro_log = shapiro(tc_log.sample(min(len(tc_log), 5000), random_state=42))

log_and_print(f"\n--- Normality Test (Shapiro-Wilk) ---")
log_and_print(f"A p-value < 0.05 suggests the data is not normally distributed.")
log_and_print(f"P-value (original): {shapiro_orig.pvalue:.3g}")
log_and_print(f"P-value (log10): {shapiro_log.pvalue:.3g}")

# %%
# Perform normality tests on all numeric features
normality_results = perform_normality_tests(df)
log_and_print("\n--- Normality Tests for All Numeric Features ---")
style_df(normality_results.sort_values(by="P-Value", ascending=False))

# %% [markdown]
# **Interpretation:**
#
# *   The summary table confirms our visual assessment. The **skewness** drops from `8.0` to `0.4` after the log transformation, indicating a significant improvement in symmetry.
# *   The Shapiro-Wilk test p-values (where p < 0.05 suggests non-normality) provide statistical evidence of this improvement. While the log-transformed data is still not perfectly normal (p-value is very small), it is substantially closer and better satisfies the assumptions of many models.
# *   The table above shows the normality test results for all numeric features. Most features are not normally distributed, which reinforces the importance of using transformations or non-parametric methods in subsequent modeling steps.

# %% [markdown]
# ### Correlation Analysis
#
# Finally, let's examine the correlation structure among the top features and the target variable. This helps identify potentially predictive features, as well as multicollinearity (high correlation between predictors), which can inform feature selection for modeling.

# %%
# Compute correlation matrix for top 10 features most correlated with the target
top_corr_features = df.corr(numeric_only=True)['thermal_conductivity'].abs().sort_values(ascending=False).head(11).index
corr_matrix = df[top_corr_features].corr()

import seaborn as sns
plt.figure(figsize=(8, 6))
# Use smaller font for annotations and axis labels for readability
ax = sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    square=True,
    annot_kws={"size": 8},
    cbar_kws={"shrink": 0.8}
)
plt.title('Correlation Matrix: Top Features vs. Thermal Conductivity', fontsize=12, pad=15)
plt.xticks(fontsize=8, rotation=45, ha='right')
plt.yticks(fontsize=8, rotation=0)
plt.tight_layout()
plt.savefig(CORR_MATRIX_PATH, bbox_inches='tight')
log_and_print(f"Correlation matrix plot saved to {CORR_MATRIX_PATH}")
plt.show()

# %% [markdown]
# **Interpretation:**
#
# *   The heatmap above highlights the features most strongly correlated with thermal conductivity, such as `density` and `vickers_hardness`.
# *   We also observe high correlations between some independent features (e.g., `density` and `atomic_mass`). This indicates potential multicollinearity, which can affect the interpretability and stability of some linear models. This analysis is crucial for guiding feature selection and engineering in the subsequent modeling stages.

# %% [markdown]
# ## 3. Next Steps: Clustering and Unsupervised Learning
# 
# This notebook focused on the initial exploratory data analysis. The insights gained here—particularly the need for log transformations and the awareness of feature correlations—are foundational for the next steps.
# 
# For dimensionality reduction (PCA), clustering, and cluster composition analysis, please see the dedicated notebook:
# 
# **[2_clustering_and_pca.py](2_clustering_and_pca.py)**
# 
# That notebook continues the workflow by uncovering hidden structure in the data and generating cluster-based features for downstream modeling.
