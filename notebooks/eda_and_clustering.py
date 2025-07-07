# %%
#!/usr/bin/env python
# coding: utf-8

# %%
# In[2]:

# %%
import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import plotly.express as px
import plotly.io as pio
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# %%
# Add src directory to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
# %%
# Import custom modules
from data import load_and_merge_data, impute_and_clean_data
from features import featurize_data
from utils import  prepare_cluster_analysis_data, prepare_data_for_modeling, load_cached_dataframe, cache_dataframe
from viz import plot_clusters
from modeling import get_optimal_k

# %%
# Try to load cached featurized data
from utils import load_cached_dataframe, cache_dataframe

# %%
# Load environment variables
load_dotenv(dotenv_path='../.env')

# %%
# Path to cache file (ensure this is correct)
cache_path = 'data/processed/featurized.csv'

# %%
# Load environment variables
load_dotenv(dotenv_path='../.env')

# %%
# Set Plotly template
pio.templates.default = "plotly_white"

# %%
# Add src to path to allow for module imports
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# %%
# Load the API key from environment variables
load_dotenv()
API_KEY = os.getenv("MP_API_KEY")

# %%
# To enable interactive plots in VSCode
pio.renderers.default = 'vscode'

# %%
# Set display options for pandas
pd.set_option('display.max_rows', 100)

# %%

# # Exploratory Data Analysis and Clustering of Inorganic Materials for Thermal Conductivity
# 
# **Author:** Angela C Davis  
# **Date:** June 30, 2025
# 
# This notebook demonstrates advanced exploratory data analysis (EDA) and unsupervised learning on a curated dataset of inorganic materials, with a focus on understanding the factors that govern thermal conductivity. Drawing on my background in materials science, I designed this project to showcase best practices in data wrangling, dimensionality reduction, clustering, and interactive visualization for scientific discovery.
# 
# **Project Motivation:**
# Thermal conductivity is a critical property for materials used in energy, electronics, and thermal management. By applying machine learning and clustering techniques, we can uncover hidden patterns, group materials with similar transport behavior, and generate hypotheses for further research or targeted materials design.
# 
# **Portfolio Context:**
# This project is part of my data science portfolio and is intended to demonstrate my ability to combine domain expertise with modern ML workflows. The research background and scientific context are further detailed in the accompanying project plan (see `reports/project_plan.ipynb`).
# 
# ---

# %%
# # Exploratory Data Analysis and Clustering of Materials for Thermal Conductivity
# 
# This notebook showcases a deep exploratory data analysis (EDA) of a materials dataset, focusing on identifying the underlying structure and characteristics of materials concerning their thermal conductivity. The goal is to use dimensionality reduction and clustering techniques to uncover patterns and group materials into meaningful classes.
# 
# **The key steps in this analysis are:**
# 
# 1.  **Data Loading and Featurization:** Combining multiple data sources and generating a rich feature set for each material.
# 2.  **Dimensionality Reduction with PCA:** Using Principal Component Analysis (PCA) to distill the most important information from the high-dimensional feature space.
# 3.  **Clustering with K-Means:** Applying the K-Means algorithm to group materials based on their featurized properties.
# 4.  **In-depth Cluster Analysis:** Investigating the composition of each cluster to understand what defines these material groups, looking at their chemical families, crystal structures, and temperature ranges.
# 5.  **Advanced Visualization:** Employing interactive plots to clearly communicate the relationships between material features and the identified clusters.

# %%
# # Data Loading and Preparation

# Standard library imports
import os
import sys

# Third-party imports
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

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
# Split the data for EDA
# We'll perform EDA on the training set to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%

print("Data preparation and featurization complete.")
df_featurized.head()

# %%

# # Clustering and PCA Analysis

# %%
# In[4]:

# %%

# 5. PCA for Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# %%
# Standardize the features before PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Use the same scaler to transform the test set

# %%
# Apply PCA
pca = PCA(n_components=0.95) # Retain 95% of the variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled) # Use the same PCA to transform the test set

# %%
print(f"Original number of features: {X_train.shape[1]}")
print(f"Number of features after PCA: {X_train_pca.shape[1]}")
print(f"Shape of X_train_pca: {X_train_pca.shape}")
print(f"Shape of X_test_pca: {X_test_pca.shape}")

# %%
# Create a scree plot
explained_variance = pca.explained_variance_ratio_
fig = px.bar(x=range(1, explained_variance.shape[0] + 1), y=explained_variance,
             labels={'x': 'Principal Component', 'y': 'Explained Variance Ratio'},
             title='Scree Plot')
fig.show()

# %%
# Determine the optimal number of clusters for K-Means
optimal_k, silhouette_scores = get_optimal_k(X_train_scaled)
print(f"The optimal number of clusters was found to be: {optimal_k}")

# %%
# For this analysis, we will use 3 clusters as requested.
n_clusters_set = optimal_k if optimal_k is not None else 3
print(f"Using {n_clusters_set} clusters for this analysis.")

# %%
# Perform K-Means clustering with 3 clusters
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=n_clusters_set, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_train_scaled)

# %%
# Visualize the clusters using PCA
fig_2d, fig_3d = plot_clusters(X_train_scaled, cluster_labels, X_train.index, n_clusters_set)
fig_2d.show()
fig_3d.show()

# %%

# ### Analyze Cluster Composition

# %%
# In[5]:

# %%

# Add cluster labels to the training data and prepare for analysis
X_train_clustered = X_train.copy()
X_train_clustered['cluster'] = kmeans.labels_

# %%
# Create a DataFrame with cluster labels for the analysis function
df_cluster_labels = pd.DataFrame({'cluster_label': kmeans.labels_}, index=X_train.index)

# %%
# Prepare the data for cluster analysis using the new utility function
# This function handles merging cluster labels and aggregating data
###delete df_cluster_analysis = prepare_cluster_analysis_data(df_featurized, df_cluster_labels)

df_with_clusters = df_featurized.merge(df_cluster_labels, left_index=True, right_index=True, how='left')

# %%
# Display the aggregated cluster analysis results
print("Aggregated Cluster Analysis Results:")
display(df_with_clusters)
analysis_df = df_with_clusters.copy()
# --- Detailed Analysis and Visualization ---
# Merge the cluster labels back into the training data for detailed plotting
#analysis_df = pd.merge(
#    X_train_clustered,
#    df_featurized['chemistry', 'crystal_structure'],
#    left_index=True,
#    right_index=True,
#    how='left'
#)
# %%
####print('chemistry' in df_featurized.columns)
####print(df_featurized.columns)

####print('chemistry' in df_cluster_analysis.columns)
####print(df_cluster_analysis.columns)

####analysis_df['chemistry'] = df_featurized.loc[analysis_df.index, 'chemistry']
####analysis_df['crystal_structure'] = df_featurized.loc[analysis_df.index, 'crystal_structure']


# %%
print(list(df_featurized))
print(df_featurized['crystal_structure'].describe())
print(df_featurized['MagpieData_mean_SpaceGroupNumber'].describe())
print(df_featurized['crystal_system'].describe())
print(df_featurized['spacegroup'].describe())

#print(df_featurized[[ 'crystal_structure', 'MagpieData_mean_SpaceGroupNumber']].notna().sum())

# %%
# ### Advanced Visualization:
# 
import pandas as pd
import plotly.express as px

# Filter and cast
df_plot = df_with_clusters.dropna(subset=['cluster_label', 'crystal_structure']).copy()
df_plot['crystal_structure'] = df_plot['crystal_structure'].astype(str)

# Count values per group
df_counts = df_plot.groupby(['cluster_label', 'crystal_structure']).size().reset_index(name='count')

# Normalize to percentages
df_total = df_counts.groupby('cluster_label')['count'].transform('sum')
df_counts['percent'] = df_counts['count'] / df_total * 100

# Plot
fig = px.bar(
    df_counts,
    x='cluster_label',
    y='percent',
    color='crystal_structure',
    title='Crystal Structure Composition by Cluster (Normalized)',
    labels={'percent': 'Percentage of Materials', 'cluster_label': 'Cluster'},
    barmode='stack',
    category_orders={"crystal_structure": ["Cubic", "Tetragonal", "Orthorhombic", "Hexagonal", "Trigonal", "Monoclinic", "Triclinic", "Unknown"]}
)

fig.update_layout(template="plotly_white", yaxis_tickformat='.0f')
fig.show()

# %%

# %%
# Visualize the chemistry distribution per cluster
import plotly.express as px

# Make a copy and clean
df_plot = df_with_clusters.dropna(subset=['cluster_label', 'chemistry']).copy()
df_plot['chemistry'] = df_plot['chemistry'].astype(str)

# Count chemistry types per cluster
df_chem_counts = df_plot.groupby(['cluster_label', 'chemistry']).size().reset_index(name='count')

# Normalize to percent within each cluster
df_chem_counts['percent'] = df_chem_counts.groupby('cluster_label')['count'].transform(lambda x: 100 * x / x.sum())

# Plot
fig = px.bar(
    df_chem_counts,
    x='cluster_label',
    y='percent',
    color='chemistry',
    barmode='stack',
    title="Chemical Class Distribution per Cluster (Normalized)",
    labels={'cluster_label': 'Cluster', 'percent': 'Percentage of Materials'},
    text_auto='.1f'
)

fig.update_layout(template="plotly_white", yaxis_tickformat='.0f')
fig.show()

# In[ ]:
print(list(df_featurized['formula']))
# %%


# %%
import seaborn as sns
import matplotlib.pyplot as plt

pivot = df_plot.pivot(index='formula', columns='temperature', values='cluster_label')
plt.figure(figsize=(12, 10))
sns.heatmap(pivot, cmap='tab10', annot=False, cbar_kws={'label': 'Cluster'})
plt.title("Cluster Assignment per Formula Across Temperature")
plt.xlabel("Temperature (K)")
plt.ylabel("Formula")
plt.tight_layout()
plt.show()


# %%

import pandas as pd
import plotly.express as px

# Clean and prep
df_struct = df_with_clusters.dropna(subset=['crystal_structure', 'chemistry']).copy()
df_struct['crystal_structure'] = df_struct['crystal_structure'].astype(str)
df_struct['chemistry'] = df_struct['chemistry'].astype(str)

# Count per chemistry/structure combo
df_count = df_struct.groupby(['chemistry', 'crystal_structure']).size().reset_index(name='count')

# Normalize to percent per chemistry class
df_count['percent'] = df_count.groupby('chemistry')['count'].transform(lambda x: 100 * x / x.sum())

# Sort categories if desired
structure_order = ["Cubic", "Tetragonal", "Orthorhombic", "Hexagonal", "Trigonal", "Monoclinic", "Triclinic", "Unknown"]

fig = px.bar(
    df_count,
    x='chemistry',
    y='percent',
    color='crystal_structure',
    title='Crystal Structure Composition per Chemistry Class (Normalized)',
    labels={'percent': 'Percentage of Materials'},
    category_orders={'crystal_structure': structure_order},
    barmode='stack'
)

fig.update_layout(template="plotly_white", yaxis_tickformat='.0f')
fig.show()



# %%
# ### 3D PCA Scatter Plot with Material Class Coloring
# 
# This plot provides a direct visual link between the clusters and the material types. By coloring the points in the 3D PCA space according to their `material_class`, we can visually inspect how well the clustering algorithm has separated materials into meaningful groups.

# %%
# In[ ]:

# %%

from sklearn.decomposition import PCA

# %%
# Perform PCA to get 3 components
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_scaled)

# %%
# Create a new DataFrame for plotting that includes PCA components and material classifications
plot_df = pd.DataFrame(X_train_pca, columns=['PC1', 'PC2', 'PC3'], index=X_train.index)
plot_df = plot_df.join(analysis_df[['cluster', 'material_class', 'crystal_system_classified']])

# %%
# Create an interactive 3D scatter plot
fig_3d_material = px.scatter_3d(
    plot_df.dropna(subset=['material_class']),
    x='PC1',
    y='PC2',
    z='PC3',
    color='material_class',  # Color points by their material class
    symbol='cluster',        # Use different symbols for each cluster
    title='3D PCA of Clusters Colored by Material Class',
    hover_data=['crystal_system_classified']
)

# %%
fig_3d_material.show()

# %%

# In[ ]:

# %%

# Analyze clusters for materials around room temperature (290-310 K)
room_temp_df = analysis_df[(analysis_df['T [K]'] >= 290) & (analysis_df['T [K]'] <= 310)]

# %%
print("--- Cluster Analysis at Room Temperature (290-310 K) ---")

# %%
# Visualize the material class distribution per cluster for room temperature
if not room_temp_df.empty:
    fig_rt = px.sunburst(
        room_temp_df.dropna(subset=['material_class']),
        path=['cluster', 'material_class'],
        title="Material Class Distribution by Cluster (Room Temperature)"
    )
    fig_rt.show()

# %%
    # Detailed breakdown
    for i in range(n_clusters_set):
        cluster_data_rt = room_temp_df[room_temp_df['cluster'] == i]
        if not cluster_data_rt.empty:
            print(f"\n--- Cluster {i} (Room Temperature) ---")
            print(f"Number of materials: {len(cluster_data_rt)}")
            print("Material Class Distribution:")
            print(cluster_data_rt['material_class'].value_counts(normalize=True).to_string())
else:
    print("No data available in the room temperature range for this analysis.")

# %%

# 
# ### Handling Small Clusters: The Case of Cluster 7
# 
# Our analysis reveals that Cluster 7 is exclusively composed of Uranium Oxide (U2O) measurements. While this is a fascinating insight—indicating that the feature space clearly separates this material from others—the cluster contains too few data points to train a reliable, specialized machine learning model.
# 
# **Strategy:**
# - **Insight:** We've identified a chemically distinct group that behaves differently from other materials in the dataset.
# - **Modeling:** We will exclude this cluster from the cluster-based modeling process. The data points from Cluster 7 would be better handled by a general model trained on the entire dataset, or flagged for further data collection.
# 
# This is a common scenario in data science where clustering provides valuable domain insights even when not all clusters are suitable for building individual models.
# 

# %%
# In[ ]:

# %%

print(analysis_df.columns)
print(analysis_df[['formula','temperature','cluster_label']].head())

#df_cluster_analysis = prepare_cluster_analysis_data(df_merged, df_cluster_labels)

# %%

# %% [markdown]
# ### Additional EDA: Histograms (Linear and Log), Missing Values

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Drop non-numeric for histogram visualization
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Plot histograms - linear scale
df[numeric_cols].hist(bins=30, figsize=(15, 10))
plt.suptitle("Histograms (Linear Scale)", fontsize=16)
plt.tight_layout()
plt.show()

# Plot histograms - log scale
df_log = df[numeric_cols].replace(0, np.nan).dropna()
df_log = df_log.applymap(lambda x: np.log10(x) if x > 0 else np.nan)

df_log.hist(bins=30, figsize=(15, 10))
plt.suptitle("Histograms (Log Scale)", fontsize=16)
plt.tight_layout()
plt.show()

# %%
