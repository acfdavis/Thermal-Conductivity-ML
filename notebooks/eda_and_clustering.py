#!/usr/bin/env python
# coding: utf-8

# %% [markdown]
# # Exploratory Data Analysis and Clustering of Inorganic Materials
#
# This notebook-style script demonstrates how domain knowledge and modern data
# science tools can reveal structure in a thermal conductivity dataset. The goal
# is to perform exploratory analysis, reduce dimensionality, and uncover
# chemically meaningful clusters before building predictive models.

# %%
import os
import sys
import pandas as pd
import numpy as np
import plotly.io as pio
import plotly.express as px
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# add src directory
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from data import load_and_merge_data, impute_and_clean_data
from features import featurize_data
from utils import prepare_data_for_modeling
from viz import (
    plot_clusters,
    plot_pca_variance,
    plot_cluster_crystal_structure,
    plot_cluster_chemistry_distribution,
    plot_structure_by_chemistry,
    plot_pca_material_class,
    plot_numeric_histograms,
    plot_numeric_histograms_log,
)
from modeling import get_optimal_k

# plotting defaults
load_dotenv(os.path.join('..', '.env'))
pio.templates.default = "plotly_white"
pio.renderers.default = 'vscode'

pd.set_option('display.max_rows', 100)
CACHE_PATH = '../data/processed/featurized.pkl'

# %% [markdown]
# ## 1. Load and Featurize the Dataset

# %%
df_raw = load_and_merge_data()
df_clean = impute_and_clean_data(df_raw)
df = featurize_data(df_clean, composition_col='formula', cache_path=CACHE_PATH)
print(f"Featurized dataframe shape: {df.shape}")

df.head()

# %% [markdown]
# ### Basic Feature Distributions

# %%
plot_numeric_histograms(df)
plot_numeric_histograms_log(df)

# %% [markdown]
# ## 2. Prepare Data for PCA and Clustering

# %%
X, y = prepare_data_for_modeling(df, target_col='thermal_conductivity')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# ### Dimensionality Reduction

# %%
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(
    f"Original features: {X_train.shape[1]}, After PCA: {X_train_pca.shape[1]}"
)
fig = plot_pca_variance(pca)
fig.show()

# %% [markdown]
# ## 3. K-Means Clustering

# %%
optimal_k, silhouette_scores = get_optimal_k(X_train_scaled)
N_CLUSTERS = optimal_k if optimal_k else 3
print(f"Using {N_CLUSTERS} clusters")

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_train_scaled)

fig_2d, fig_3d = plot_clusters(X_train_scaled, cluster_labels, X_train.index, N_CLUSTERS)
fig_2d.show()
fig_3d.show()

# %% [markdown]
# ## 4. Cluster Composition Analysis

# %%
df_with_clusters = df.copy()
df_with_clusters['cluster_label'] = kmeans.predict(scaler.transform(X))

fig_struct = plot_cluster_crystal_structure(df_with_clusters)
fig_struct.show()

fig_chem = plot_cluster_chemistry_distribution(df_with_clusters)
fig_chem.show()

fig_combo = plot_structure_by_chemistry(df_with_clusters)
fig_combo.show()

analysis_df = df_with_clusters[['cluster_label', 'material_class', 'crystal_system_classified']]
fig_material = plot_pca_material_class(X_train_scaled, analysis_df.loc[X_train.index])
fig_material.show()

# %% [markdown]
# ### Room Temperature Focus

# %%
room_temp = df_with_clusters[(df_with_clusters['T [K]'] >= 290) & (df_with_clusters['T [K]'] <= 310)]
if not room_temp.empty:
    fig_rt = px.sunburst(
        room_temp.dropna(subset=['material_class']),
        path=['cluster_label', 'material_class'],
        title='Material Class Distribution by Cluster (Room Temperature)'
    )
    fig_rt.show()

# %% [markdown]
# The clustering reveals chemically coherent groups and highlights materials that
# behave differently around room temperature. These insights guide subsequent
# model development and data collection efforts.
