#!/usr/bin/env python
# coding: utf-8

# %% [markdown]
# # Exploratory Data Analysis for Thermal Conductivity Prediction
#
# ## 1. Project Objective
#
# This notebook marks the first step in an end-to-end machine learning project to predict the thermal conductivity of inorganic materials. The primary goal of this initial phase—Exploratory Data Analysis (EDA)—is to thoroughly understand the dataset's structure, identify potential data quality issues, and uncover key relationships that will inform our subsequent modeling strategies.
#
# ## 2. Business Context
#
# In materials science, discovering new materials with optimal thermal properties is critical for developing next-generation electronics, batteries, and energy systems. However, the traditional process of synthesizing and testing materials is extremely slow and expensive. This project demonstrates how a data-driven approach can accelerate this discovery process, providing significant business value by reducing R&D costs and shortening time-to-market.
#
# ## 3. Technical Skills & Achievements Demonstrated
#
# - **Structured Project Setup:** The project is organized with a modular `src` directory, demonstrating best practices for creating reproducible and maintainable code.
# - **Efficient Data Processing:** Raw chemical formulas are converted into a rich set of over 100 physics-informed features. An efficient caching system (`.parquet`) is implemented to dramatically speed up subsequent data loading and processing.
# - **Insightful Visualization:** Professional, publication-quality visualizations are used to analyze feature distributions and correlations. The code demonstrates how to generate comprehensive, multi-page PDF reports while keeping the notebook summary clean and concise.
# - **Statistical Rigor:** Visual observations are validated with formal statistical tests (e.g., Shapiro-Wilk test for normality), confirming the need for specific preprocessing steps like log transformations.

# %% [markdown]
# ## 1. Environment Setup and Data Preparation
#
# This section sets up the analysis environment. We import standard libraries and our custom modules from the `src` directory, which contains reusable functions for data processing, visualization, and environment configuration. We then load the raw data, generate a rich feature set from the chemical formulas, and cache the result as a `.parquet` file to accelerate future runs. This modular approach is a key practice for building reproducible and maintainable data science workflows.

# %%
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
%matplotlib inline

# Ensure src directory is added to sys.path for modular imports
try:
    # Assumes the script is in the 'notebooks' directory
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    # Fallback for interactive environments (Jupyter, VSCode)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd()))

# Add the 'src' directory to the Python path
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
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
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots', '1_eda')
os.makedirs(PLOTS_DIR, exist_ok=True)

CACHE_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'featurized.parquet')
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
# To add statistical rigor to our visual analysis, we will now quantify our observations. This involves computing summary statistics and performing a formal normality test (Shapiro-Wilk) on the target variable, both before and after the log transformation.

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
# Finally, we examine the correlation structure of the dataset. A correlation matrix helps us identify which features are most strongly related to the target variable (`thermal_conductivity`). It also reveals potential multicollinearity (high correlation between predictor variables), an important consideration for feature selection in the modeling phase.

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
# ## 3. Conclusion and Next Steps
#
# This exploratory analysis has provided critical insights into the dataset's structure. We've confirmed the necessity of log-transforming our skewed target variable and identified key feature correlations that will inform our modeling strategy. The data is now understood and prepared for the next stage of the pipeline.
#
# The workflow continues in the next notebook, where we will use unsupervised learning techniques to uncover hidden structures in the data:
#
# **Next Notebook: [2_clustering_and_pca.py](2_clustering_and_pca.py)**
