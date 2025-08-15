import pandas as pd
import numpy as np
import re
import os
from matminer.datasets import load_dataset
from sklearn.impute import SimpleImputer

def load_and_standardize_citrine() -> pd.DataFrame:
    """Loads and standardizes the Citrine dataset."""
    print("Loading Citrine dataset...")
    df = load_dataset("citrine_thermal_conductivity")
    print(f"Citrine dataset shape before dropping NaNs: {df.shape}")
    df = df.dropna(subset=["k_expt"]).copy()
    print(f"Citrine dataset shape after dropping NaNs: {df.shape}")
    return (
        df.rename(columns={"k_expt": "thermal_conductivity", "k_condition": "temperature"})
        .assign(source="citrine")
    )

def load_and_standardize_ucsb() -> pd.DataFrame:
    """Loads and standardizes the UCSB dataset."""
    print("Loading UCSB dataset...")
    df = load_dataset("ucsb_thermoelectrics")
    print(f"UCSB dataset shape: {df.shape}")
    return (
        df.rename(columns={"composition": "formula", "temperature": "temperature", "kappa": "thermal_conductivity"})
        .assign(source="ucsb")
    )

def _standardize_nist_property_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the 'property' column in the NIST dataframe to simple labels
    like 'thermalconductivity' and 'density'.
    """
    if 'property' not in df.columns:
        return df

    def clean_name(name):
        name_lower = str(name).lower()
        if 'thermal conductivity' in name_lower:
            return 'thermalconductivity'
        if 'density' in name_lower:
            return 'density'
        return None  # Return None for properties we don't care about

    df['property'] = df['property'].apply(clean_name)
    df.dropna(subset=['property'], inplace=True)
    return df

def load_and_standardize_nist(project_root: str) -> pd.DataFrame:
    """
    Loads NIST ThermoML data, standardizes property names, and pivots the data
    to have properties as columns.
    """
    print("Loading NIST dataset...")
    data_file = os.path.join(project_root, 'data', 'raw', 'thermoml_data.parquet')
    if not os.path.exists(data_file):
        print(f"NIST data file not found at {data_file}")
        return pd.DataFrame()
    print("Reading NIST data file...")
    df_data = pd.read_parquet(data_file)
    print(f"NIST dataset shape before filtering: {df_data.shape}")
    # Filter for crystalline solids first
    if 'phase' in df_data.columns:
        df_data = df_data[df_data['phase'].str.contains('crystal', case=False, na=False)].copy()
    print(f"NIST dataset shape after filtering for crystalline solids: {df_data.shape}")
    # Standardize property names BEFORE pivoting
    df_data = _standardize_nist_property_names(df_data)
    print(f"NIST dataset shape after standardizing property names: {df_data.shape}")

    if df_data.empty:
        print("No relevant properties found in NIST crystalline data.")
        return pd.DataFrame()

    # Pivot the table to get properties as columns
    id_vars = ['formula', 'temperature']
    id_vars = [col for col in id_vars if col in df_data.columns]
    
    df_pivot = df_data.pivot_table(
        index=id_vars,
        columns='property',
        values='value',
        aggfunc='first'  # Use the first valid measurement found
    ).reset_index()

    # Rename columns for consistency
    rename_map = {
        'thermalconductivity': 'thermal_conductivity'
    }
    df_pivot.rename(columns=rename_map, inplace=True)
    df_pivot['source'] = 'nist_thermoml'
    
    return df_pivot

def parse_temperature(val):
    """Parses temperature values from strings into Kelvin."""
    if pd.isnull(val):
        return np.nan
    s = str(val).lower().strip()
    # Look for numeric values, handling Celsius/Fahrenheit if specified
    m = re.search(r"([-+]?[0-9]*\.?[0-9]+)\s*°?\s*([ckf])?", s)
    if m:
        temp, unit = float(m.group(1)), m.group(2)
        if unit == 'c':
            return temp + 273.15
        elif unit == 'f':
            return (temp - 32) * 5/9 + 273.15
        return temp  # Assume Kelvin if no unit
    if "room" in s or "ambient" in s:
        return 298.15
    return np.nan

def load_and_merge_data(project_root: str, drop_missing: bool = True) -> pd.DataFrame:
    """
    Loads, merges, and cleans data from Citrine, UCSB, and NIST sources.
    """
    print("Loading datasets...")
    citrine_df = load_and_standardize_citrine()
    ucsb_df = load_and_standardize_ucsb()
    nist_df = load_and_standardize_nist(project_root)
    
    print(f"Citrine data loaded: {citrine_df.shape}")
    print(f"UCSB data loaded: {ucsb_df.shape}")
    print(f"NIST data loaded: {nist_df.shape}")

    # Combine all datasets
    df_all = pd.concat([citrine_df, ucsb_df, nist_df], ignore_index=True)

    # Clean temperature
    df_all["temperature"] = df_all["temperature"].apply(parse_temperature)
    
    # Drop duplicates, keeping the most complete record by prioritizing sources
    df_all['source_cat'] = pd.Categorical(df_all['source'], categories=['citrine', 'ucsb', 'nist_thermoml'], ordered=True)
    df_all = df_all.sort_values('source_cat')
    df_all = df_all.drop_duplicates(subset=['formula', 'temperature'], keep='first')
    df_all = df_all.drop(columns='source_cat')

    # Final cleaning of target variable
    if drop_missing:
        df_all = df_all.dropna(subset=["formula", "temperature", "thermal_conductivity"]).reset_index(drop=True)

    # Group by formula and temperature to get a single entry with all available info
    agg_dict = {
        "thermal_conductivity": "mean",
        "source": lambda s: ",".join(sorted(set(s)))
    }

    df_final = df_all.groupby(['formula', 'temperature']).agg(agg_dict).reset_index()
    print(f"Shape after merging and cleaning: {df_final.shape}")
    
    return df_final

def impute_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and imputes missing values in the DataFrame.
    """
    df_cleaned = df.copy()
    df_cleaned["temperature"] = df_cleaned["temperature"].apply(parse_temperature)
    
    if df_cleaned["temperature"].isnull().any():
        print("Warning: Imputing missing temperature values with mean.")
        imputer = SimpleImputer(strategy="mean")
        df_cleaned["temperature"] = imputer.fit_transform(df_cleaned[["temperature"]])
        
    df_cleaned = df_cleaned.dropna(subset=["formula", "thermal_conductivity"]).reset_index(drop=True)
    return df_cleaned
