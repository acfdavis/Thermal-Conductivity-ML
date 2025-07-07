import pandas as pd
import numpy as np
import re
import os
import hashlib
import requests
from matminer.datasets import load_dataset
from sklearn.impute import SimpleImputer
from typing import Dict, List, Union, Callable, Any


def cached_fetch_thermoml_xml(url: str, cache_dir: str = "./cache") -> str:
    os.makedirs(cache_dir, exist_ok=True)
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{url_hash}.xml")

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        response = requests.get(url)
        response.raise_for_status()
        xml_content = response.text
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(xml_content)
        return xml_content

def prepare_cluster_analysis_data(df_merged, df_cluster_labels):
    """
    Merges cluster labels with the main dataframe and aggregates data for cluster analysis.

    Args:
        df_merged (pd.DataFrame): The main dataframe with features.
        df_cluster_labels (pd.DataFrame): Dataframe with cluster labels.

    Returns:
        pd.DataFrame: A dataframe with aggregated cluster analysis data.
    """
    # Merge cluster labels with material class and crystal system
    df_merged_clusters = pd.merge(df_merged, df_cluster_labels, left_index=True, right_index=True)

    # Define aggregations
    agg_dict: Dict[str, Union[List[str], Callable[[Any], Any]]] = {
        'thermal_conductivity': ['mean', 'std'],
        'temperature': ['mean', 'std'],
    }

    # Add optional columns to aggregation dictionary if they exist
    optional_cols: Dict[str, Union[List[str], Callable[[Any], Any]]] = {
        'crystal_system_classified': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown',
        'mp_density': ['mean', 'std'],
        'MagpieData mean MolarMass': ['mean', 'std'],
        'mp_band_gap': ['mean', 'std']
    }

    for col, agg_funcs in optional_cols.items():
        if col in df_merged_clusters.columns:
            agg_dict[col] = agg_funcs

    # Aggregate by cluster and other features
    df_cluster_analysis = df_merged_clusters.groupby('cluster_label').agg(agg_dict).reset_index()

    # Flatten MultiIndex columns
    df_cluster_analysis.columns = ['_'.join(col).strip() for col in df_cluster_analysis.columns.values]

    # Rename columns for clarity
    rename_dict = {
        'cluster_label_': 'cluster_label',
        'thermal_conductivity_mean': 'mean_kappa',
        'thermal_conductivity_std': 'std_kappa',
        'temperature_mean': 'mean_T',
        'temperature_std': 'std_T',
    }
    
    # Add optional columns to rename dictionary
    optional_rename = {
        'crystal_system_classified_<lambda>': 'dominant_crystal_system',
        'mp_density_mean': 'mean_density',
        'mp_density_std': 'std_density',
        'MagpieData mean MolarMass_mean': 'mean_molar_mass',
        'MagpieData mean MolarMass_std': 'std_molar_mass',
        'mp_band_gap_mean': 'mean_band_gap',
        'mp_band_gap_std': 'std_band_gap'
    }

    for old_name, new_name in optional_rename.items():
        if old_name in df_cluster_analysis.columns:
            rename_dict[old_name] = new_name
            
    df_cluster_analysis = df_cluster_analysis.rename(columns=rename_dict)

    return df_cluster_analysis


def prepare_data_for_modeling(df, target_col='thermal_conductivity'):
    """
    Prepare the data for modeling by selecting features, handling missing values.

    Args:
        df (pd.DataFrame): The featurized dataframe.
        target_col (str): The name of the target variable column.

    Returns:
        pd.DataFrame: The feature matrix (X).
        pd.Series: The target vector (y).
    """
    df_model = df.copy()
    
    # Drop rows where target is NaN
    df_model.dropna(subset=[target_col], inplace=True)
    
    y = df_model[target_col]
    
    # Select feature columns - exclude non-feature columns
    non_feature_cols = [
        target_col, 'formula', 'source', 'composition_obj', 
        'crystal_system', 'mp_spacegroup'
    ]
    
    # Get all numeric columns as features, dropping non-feature columns if they are numeric
    X = df_model.select_dtypes(include=np.number).drop(columns=[col for col in non_feature_cols if col in df_model.select_dtypes(include=np.number).columns], errors='ignore')

    # Impute any remaining NaNs in features
    # Get columns with NaN values
    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        print(f"Imputing NaN values in columns: {nan_cols}")
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
    
    return X, y

def cache_dataframe(df, cache_path):
    """Cache a DataFrame to disk as a parquet file."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)

def load_cached_dataframe(cache_path):
    """Load a cached DataFrame from disk if it exists, else return None."""
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    return None

def clear_cache(cache_path):
    """Delete a cached file if it exists."""
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"Cache cleared: {cache_path}")
    else:
        print(f"No cache found at: {cache_path}")
