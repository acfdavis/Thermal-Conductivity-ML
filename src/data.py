import pandas as pd
import numpy as np
import re
import os
import joblib
from matminer.datasets import load_dataset
from sklearn.impute import SimpleImputer

def load_and_standardize_citrine() -> pd.DataFrame:
    df = load_dataset("citrine_thermal_conductivity")
    if df is None:
        return pd.DataFrame()
    df = df.dropna(subset=["k_expt"]).copy()
    return (
        df.rename(columns={
            "k_expt": "thermal_conductivity",
            "k_condition": "temperature"
        })
        .assign(source="citrine")
    )

def load_and_standardize_ucsb() -> pd.DataFrame:
    df = load_dataset("ucsb_thermoelectrics")
    if df is None:
        return pd.DataFrame()
    df = (
        df.rename(columns={
            "composition": "formula",
            "temperature": "temperature",
            "kappa": "thermal_conductivity"
        })
        .assign(source="ucsb")
    )
    return df

def _filter_thermal_conductivity_data(df_data):
    print("Original shape:", df_data.shape)
    
    if "property" not in df_data.columns:
        print("Missing 'property' column")
        return pd.DataFrame()

    conductivity_df = df_data[df_data['property'].str.contains('Thermal conductivity', case=False, na=False)]
    print("Conductivity rows:", conductivity_df.shape)

    if "phase" not in conductivity_df.columns:
        print("Missing 'phase' column")
        return pd.DataFrame()

    thermal_conductivity_crystalline = conductivity_df[
        conductivity_df['phase'].str.contains('crystal', case=False, na=False)
    ]
    print("Crystalline rows:", thermal_conductivity_crystalline.shape)

    if thermal_conductivity_crystalline.empty:
        print("No crystalline thermal conductivity data found.")
        return pd.DataFrame()

    cleaned_df = thermal_conductivity_crystalline.dropna(axis=1, how='all')
    standardized_df = cleaned_df.rename(columns={
        'value': 'thermal_conductivity',
        'temperature': 'temperature',
        'formula': 'formula'
    }).assign(source="nist_thermoml")

    return standardized_df


def load_and_standardize_nist() -> pd.DataFrame:
    """Load NIST ThermoML data, filter for thermal conductivity of crystalline solids."""
    data_file = r"c:\Users\angel\thermal_conductivity\ml_conductivity_project_user_ready\data\raw\thermoml_data.pkl"


    df_data = joblib.load(data_file)

    if isinstance(df_data, pd.DataFrame) and 'property' in df_data.columns:
        df_data = df_data.dropna(subset=['property'])

    return _filter_thermal_conductivity_data(df_data)



def parse_temperature(val):
    if pd.isnull(val):
        return np.nan
    s = str(val).lower().strip()
    # Check for Celsius, Kelvin, or Fahrenheit
    m = re.search(r"([-+]?[0-9]*\.?[0-9]+)\s*°?\s*([ckf])", s)
    if m:
        temp = float(m.group(1)); unit = m.group(2)
        if unit == 'c':
            return temp + 273.15
        elif unit == 'f':
            return (temp - 32) * 5/9 + 273.15
        else:  # unit == 'k'
            return temp
    # Check for plain number
    m2 = re.search(r"([-+]?[0-9]*\.?[0-9]+)", s)
    if m2:
        return float(m2.group(1))
    # Check for room temperature keywords
    if "room" in s or "ambient" in s or "standard" in s:
        return 298.15
    return np.nan

def load_and_merge_data(drop_missing=True):
    """
    Load and combine datasets from Citrine, UCSB, and NIST.
    Optionally drop rows missing formula, temperature, or thermal_conductivity.
    Returns a cleaned, merged DataFrame ready for feature engineering.
    """
    print("Loading datasets...")
    citrine_df = load_and_standardize_citrine()
    print(f"Citrine data loaded: {citrine_df.shape}")
    ucsb_df = load_and_standardize_ucsb()
    print(f"UCSB data loaded: {ucsb_df.shape}")
    nist_df = load_and_standardize_nist()
    print(f"NIST data loaded: {nist_df.shape}")

    datasets_to_combine = []
    if not citrine_df.empty:
        datasets_to_combine.append(citrine_df)
    if not ucsb_df.empty:
        datasets_to_combine.append(ucsb_df)
    if not nist_df.empty:
        datasets_to_combine.append(nist_df)

    if datasets_to_combine:
        df_all = pd.concat(datasets_to_combine, ignore_index=True)
        print(f"Combined dataset shape: {df_all.shape}")
        # Standardize temperature and drop duplicates
        df_all["temperature"] = df_all["temperature"].apply(parse_temperature)
        df_all = df_all.drop_duplicates(subset=["formula", "temperature"], keep="first").reset_index(drop=True)
        # Optionally drop rows missing any critical field
        if drop_missing:
            df_all = df_all.dropna(subset=["formula", "temperature", "thermal_conductivity"]).reset_index(drop=True)
        # Group and average kappa for duplicate entries
        df_feat = (
            df_all
            .groupby(["formula", "temperature"])
            .agg({
                "thermal_conductivity": "mean",
                "source": lambda s: ",".join(sorted(set(s)))
            })
            .reset_index()
        )
        print(f"Shape after grouping and cleaning: {df_feat.shape}")
        return df_feat
    else:
        print("No datasets were successfully loaded.")
        return pd.DataFrame()

def impute_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and impute missing values in the DataFrame.
    Applies temperature parsing and imputes missing temperature values (only if justified).
    Drops rows missing formula or thermal_conductivity.
    Use with caution: imputation should only be used if physically justified.
    """
    df_cleaned = df.copy()
    df_cleaned["temperature"] = df_cleaned["temperature"].apply(parse_temperature)
    # Only impute if a strong physical/statistical justification exists
    if df_cleaned["temperature"].isnull().any():
        print("Warning: Imputing missing temperature values with mean. Ensure this is physically justified.")
        imputer = SimpleImputer(strategy="mean")
        df_cleaned["temperature"] = imputer.fit_transform(df_cleaned[["temperature"]])
    df_cleaned = df_cleaned.dropna(subset=["formula", "thermal_conductivity"]).reset_index(drop=True)
    return df_cleaned
