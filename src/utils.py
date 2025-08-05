import pandas as pd
import numpy as np
import re
import os
import hashlib
import requests
from sklearn.impute import SimpleImputer
from typing import Dict, List, Union, Callable, Any, cast
from scipy.stats import shapiro


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
    agg_dict: Dict[str, Union[str, List[str], Callable[..., Any]]] = {
        'thermal_conductivity': ['mean', 'std'],
        'temperature': ['mean', 'std'],
    }

    # Add optional columns to aggregation dictionary if they exist
    optional_cols: Dict[str, Union[str, List[str], Callable[..., Any]]] = {
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
    import numpy as np
    import pandas as pd

    # Convert all object columns to string to avoid ArrowInvalid errors
    for col in df.select_dtypes(include=['object', 'category']):
        df[col] = df[col].astype(str)
    # Also handle columns with custom types (e.g., Enums)
    for col in df.columns:
        if df[col].apply(lambda x: not isinstance(x, (str, int, float, np.integer, np.floating, type(None)))).any():
            df[col] = df[col].astype(str)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)
    
def load_cached_dataframe(cache_path: str, verbose: bool = True) -> pd.DataFrame | None:
    """
    Load a cached DataFrame from disk if it exists, else return None.

    Args:
        cache_path (str): Path to the cached parquet file.
        verbose (bool): Whether to print status messages.

    Returns:
        pd.DataFrame | None: The loaded DataFrame, or None if not found or failed to load.
    """
    import traceback
    base, ext = os.path.splitext(cache_path)
    tried = []
    # Try Parquet first
    if os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            if verbose:
                print(f"Loaded cached DataFrame from {cache_path} (parquet)")
            return df
        except Exception as e:
            print(f"Warning: Failed to load cached DataFrame from {cache_path} (parquet): {e}")
            tried.append(f"parquet: {e}")
            # Optionally, remove corrupted cache file
            try:
                os.remove(cache_path)
                print(f"Corrupted cache removed: {cache_path}")
            except Exception:
                pass
    # Try Pickle
    pkl_path = base + ".pkl"
    if os.path.exists(pkl_path):
        try:
            df = pd.read_pickle(pkl_path)
            if verbose:
                print(f"Loaded cached DataFrame from {pkl_path} (pickle)")
            return df
        except Exception as e:
            print(f"Warning: Failed to load cached DataFrame from {pkl_path} (pickle): {e}")
            tried.append(f"pickle: {e}")
    # Try CSV
    csv_path = base + ".csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if verbose:
                print(f"Loaded cached DataFrame from {csv_path} (csv)")
            return df
        except Exception as e:
            print(f"Warning: Failed to load cached DataFrame from {csv_path} (csv): {e}")
            tried.append(f"csv: {e}")
    if tried:
        print(f"All attempts to load cached DataFrame failed: {tried}")
    else:
        if verbose:
            print(f"No cache file found for {cache_path} (tried .parquet, .pkl, .csv)")
    return None

def clear_cache(cache_path):
    """Delete a cached file if it exists."""
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"Cache cleared: {cache_path}")
    else:
        print(f"No cache found at: {cache_path}")

def style_df(df: pd.DataFrame):
    """
    Apply professional styling to a pandas DataFrame for clear presentation.

    Args:
        df (pd.DataFrame): The DataFrame to style.

    Returns:
        Styler: A styled DataFrame object for display.
    """
    # Only format float columns, leave others (like 'Model') as is
    float_cols = df.select_dtypes(include=['float', 'float64', 'float32']).columns
    format_dict = {col: "{:.3f}" for col in float_cols}
    # Simple, clean style: light header, subtle borders, no color gradients
    return df.style.format(format_dict, na_rep="-") \
        .set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#f5f5f5'), ('color', '#222'), ('font-weight', 'bold'), ('border', '1px solid #ccc')]},
            {'selector': 'td', 'props': [('border', '1px solid #ddd'), ('padding', '6px')]}
        ])

def save_plot(fig, path):
    """Save a Plotly or Matplotlib figure to disk and print a confirmation."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        fig.write_image(path)
        print(f"Plot saved to {path}")
    except AttributeError:
        # Try matplotlib
        try:
            fig.savefig(path)
            print(f"Plot saved to {path}")
        except Exception as e:
            print(f"Failed to save plot: {e}")

def setup_environment(env_path=None):
    """Load environment variables and set plotting defaults."""
    import plotly.io as pio
    from dotenv import load_dotenv
    if env_path is None:
        env_path = os.path.join('..', '.env')
    load_dotenv(env_path)
    pio.templates.default = "plotly_white"
    pio.renderers.default = 'vscode'
    import pandas as pd
    pd.set_option('display.max_rows', 100)

def get_feature_list(df, exclude=None):
    """Return a list of feature columns, excluding specified columns."""
    if exclude is None:
        exclude = []
    numeric_cols = df.select_dtypes(include=np.number).columns
    return [col for col in numeric_cols if col not in exclude]

def log_and_print(msg):
    """Print and log a message (for notebook consistency)."""
    print(msg)
    # Optionally, add logging to file or notebook cell here

def display_markdown(md_str):
    """Display markdown in a notebook cell programmatically."""
    from IPython.display import display, Markdown
    display(Markdown(md_str))

def process_and_featurize(cache_path, composition_col='formula'):
    """
    Load, clean, and featurize raw data, then cache the result.
    Args:
        cache_path (str): Path to cache the featurized DataFrame.
        composition_col (str): Name of the column with chemical formulas.
    Returns:
        pd.DataFrame: The featurized DataFrame.
    """
    from data import load_and_merge_data, impute_and_clean_data
    from features import featurize_data
    df_raw = load_and_merge_data()
    df_clean = impute_and_clean_data(df_raw)
    return featurize_data(df_clean, composition_col=composition_col, cache_path=cache_path)


def load_or_process_dataframe(cache_path: str) -> pd.DataFrame:
    """Load a DataFrame from cache if it exists, else process and cache it."""
    df = load_cached_dataframe(cache_path)
    if df is not None:
        print(f"Loaded cached DataFrame from {cache_path}")
        return df
    df = process_and_featurize(cache_path)
    cache_dataframe(df, cache_path)
    print(f"Processed and cached DataFrame to {cache_path}")
    return df

def load_selected_features(df, features_path):
    """
    Loads a list of selected features from a JSON file and applies it to the given DataFrame.
    Returns the filtered DataFrame and the feature list.
    """
    import json
    with open(features_path, 'r') as f:
        selected_features = json.load(f)
    available_selected_features = [f for f in selected_features if f in df.columns]
    return df[available_selected_features], available_selected_features

def validate_feature_significance(X, y, cluster_labels=None):
    """
    Perform statistical tests to validate the significance of features.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        cluster_labels (pd.Series, optional): Cluster labels to validate.

    Returns:
        dict: Dictionary containing p-values for each feature.
    """
    from scipy.stats import f_oneway

    results = {}
    for feature in X.columns:
        if cluster_labels is not None:
            # Convert cluster_labels to pandas Series if it's a numpy array
            if isinstance(cluster_labels, np.ndarray):
                cluster_labels = pd.Series(cluster_labels, index=y.index)

            # Perform ANOVA for cluster labels
            grouped = [y[cluster_labels == label] for label in cluster_labels.unique()]
            results[feature] = f_oneway(*grouped).pvalue
        else:
            # Perform correlation test for numeric features
            results[feature] = X[feature].corr(y)

    return results

def perform_normality_tests(df: pd.DataFrame, columns: List[str] = None, sample_size: int = 5000) -> pd.DataFrame:
    """
    Perform Shapiro-Wilk normality tests on specified columns of a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        columns (List[str], optional): List of column names to test. If None, all numeric columns are tested.
        sample_size (int): Maximum sample size for the Shapiro-Wilk test (default: 5000).

    Returns:
        pd.DataFrame: A DataFrame with columns ['Feature', 'P-Value', 'Is Normal'] summarizing the test results.
    """
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()

    results = []
    for col in columns:
        data = df[col].dropna()
        if len(data) > sample_size:
            data = data.sample(sample_size, random_state=42)
        stat, p_value = shapiro(data)
        results.append({
            'Feature': col,
            'P-Value': p_value,
            'Is Normal': p_value >= 0.05
        })

    return pd.DataFrame(results)