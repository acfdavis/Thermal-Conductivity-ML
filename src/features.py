"""
features.py

This module contains functions to generate and transform features for thermal conductivity modeling.
"""

import os
import re
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data as jdata
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import (
    ElementProperty,
    IonProperty,
    Stoichiometry,
    ValenceOrbital,
)
from matminer.featurizers.structure import (
    BondFractions,
    EwaldEnergy,
    MaximumPackingEfficiency,
    SiteStatsFingerprint,
    StructuralHeterogeneity,
)
from mp_api.client import MPRester
from pymatgen.core import Composition, Structure
from tqdm import tqdm

from data import impute_and_clean_data, load_and_merge_data


def featurize_data(df, composition_col='formula', cache_path=None):
    """
    Applies a series of matminer featurizers to the input dataframe.

    The function performs the following steps:
    1. Adds composition-based features using Magpie, Stoichiometry, and Valence Orbital analyses.
    2. Normalizes column names for consistency.
    3. Enriches the dataframe with materials properties from the Materials Project (if API key is available).
    4. Enriches the dataframe with features from the JARVIS-DFT database.
    5. Classifies the material type based on the composition.
    6. Classifies the crystal structure based on the space group number.

    If `cache_path` is provided and exists, load the cached result.
    Otherwise, run the pipeline and save to cache (if path provided).

    Args:
        df (pd.DataFrame): Input dataframe containing material formulas.
        cache_path (str, optional): Path to cache the features as a parquet file.

    Returns:
        pd.DataFrame: DataFrame enriched with features.
    """
    import pandas as pd
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        return pd.read_parquet(cache_path)

    print("Starting feature engineering...")

    if composition_col not in df.columns:
        raise ValueError(f"Composition column '{composition_col}' not found in DataFrame.")

    df_comp = add_composition_features(df, composition_column=composition_col)
    print("Composition features added.")

    # Normalize column names immediately after they are created
    df_comp = normalize_column_names(df_comp)
    print("Column names normalized.")

    api_key = os.getenv("MP_API_KEY")
    if api_key:
        df_comp = add_materials_project_features(df_comp, api_key)
        print("Materials Project features added.")
    else:
        print("Materials Project API key not found. Skipping these features.")

    df_jarvis = add_jarvis_features(df_comp)
    print("JARVIS features added.")

    # Add chemistry column using classify_material on formula
    df_jarvis['chemistry'] = df_jarvis[composition_col].apply(classify_material)

    # Add crystal_structure column using a more robust approach
    df_jarvis['crystal_structure'] = df_jarvis['MagpieData_mean_SpaceGroupNumber'].apply(classify_spacegroup_by_number)

    # Initialize the column with 'Unknown'
    #df_jarvis['crystal_structure'] = 'Unknown'

    # First, try to use the crystal_system from Materials Project if it exists
    #if 'crystal_system' in df_jarvis.columns:
    #    # Use .loc to avoid SettingWithCopyWarning
    #    df_jarvis.loc[df_jarvis['crystal_system'].notna(), 'crystal_structure'] = df_jarvis['crystal_system']

    # For any remaining 'Unknown' values, try to use the Magpie spacegroup number
    #if 'MagpieData_mean_SpaceGroupNumber' in df_jarvis.columns:
    #    # Identify rows that are still 'Unknown' and have a valid spacegroup number
    #    unknown_mask = (df_jarvis['crystal_structure'] == 'Unknown') & (df_jarvis['MagpieData_mean_SpaceGroupNumber'].notna())
    #    # Apply the classification function only to those rows
    #    df_jarvis.loc[unknown_mask, 'crystal_structure'] = df_jarvis.loc[unknown_mask, 'MagpieData_mean_SpaceGroupNumber'].apply(classify_spacegroup_by_number)

    # Create a unified density column
    df_jarvis["density"] = (
        df_jarvis["mp_density"]
        if "mp_density" in df_jarvis
        else pd.Series(dtype=float)
    ).combine_first(
        df_jarvis["jarvis_density"]
        if "jarvis_density" in df_jarvis
        else pd.Series(dtype=float)
    )
    # Coerce to numeric, coercing errors will set invalid parsing as NaN
    df_jarvis["density"] = pd.to_numeric(df_jarvis["density"], errors="coerce")


    print("Feature engineering complete.")
    
    # Convert any special types (like Enums) to strings before caching
    if 'crystal_system' in df_jarvis.columns:
        df_jarvis['crystal_system'] = df_jarvis['crystal_system'].astype(str)

    if cache_path:
        # Ensure the parent directory exists before attempting to write the parquet file.
        parent_dir = os.path.dirname(cache_path)
        if parent_dir:
            try:
                os.makedirs(parent_dir, exist_ok=True)
                # Helpful debug message in case directory needed to be created
                if not os.path.exists(parent_dir):
                    print(f"Created cache directory {parent_dir}")
            except Exception as e:
                # Non-fatal: we'll still try to write and let pandas raise a clear error if it fails
                print(f"Warning: could not create cache directory {parent_dir}: {e}")

        try:
            df_jarvis.to_parquet(cache_path, index=False)
            print(f"Features cached to {cache_path}.")
        except Exception as e:
            print(f"Error caching to Parquet: {e}")
            # Fallback: convert all object columns to string and retry
            print("Attempting to convert all object columns to string and re-caching...")
            for col in df_jarvis.select_dtypes(include=['object']).columns:
                df_jarvis[col] = df_jarvis[col].astype(str)
            df_jarvis.to_parquet(cache_path, index=False)
            print(f"Features successfully cached to {cache_path} after type conversion.")


    return df_jarvis



def add_composition_features(df, composition_column='formula'):
    """
    Adds features based on elemental properties of the composition, including Magpie, Stoichiometry, and ValenceOrbital.
    """
    from pymatgen.core import Composition
    from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital

    print("Preparing composition features for", len(df), "rows")
    df['composition'] = df[composition_column].apply(Composition)

    ep_featurizer = ElementProperty.from_preset("magpie", impute_nan=True)
    ep_featurizer.set_n_jobs(1)  # correct way to disable multiprocessing
    df = ep_featurizer.featurize_dataframe(df, col_id="composition")

    stoich_featurizer = Stoichiometry()
    stoich_featurizer.set_n_jobs(1)
    df = stoich_featurizer.featurize_dataframe(df, col_id="composition")

    valence_featurizer = ValenceOrbital(impute_nan=True)
    valence_featurizer.set_n_jobs(1)
    df = valence_featurizer.featurize_dataframe(df, col_id="composition")

    if df is not None:
        df = df.drop("composition", axis=1)
    return df

def normalize_column_names(df):
    df.columns = [re.sub(r"[\s\[\]<>]", "_", col) for col in df.columns]
    return df


def add_materials_project_features(df, api_key):
    """
    Enriches the DataFrame with materials properties from the Materials Project,
    including crystal system and spacegroup.

    Args:
        df (pd.DataFrame): DataFrame with a 'formula' column.
        api_key (str): Your Materials Project API key.

    Returns:
        pd.DataFrame: DataFrame with added features.
    """
    from mp_api.client import MPRester
    from tqdm import tqdm

    materials_data_list = []
    with MPRester(api_key=api_key) as mpr:
        unique_formulas = df["formula"].unique()
        print(f"Starting Materials Project query for {len(unique_formulas)} unique formulas...")

        batch_size = 100
        for i in tqdm(range(0, len(unique_formulas), batch_size), desc="Fetching Materials Project Data"):
            batch_formulas = unique_formulas[i:i+batch_size]
            try:
                docs = mpr.materials.summary.search(
                    formula=list(batch_formulas),
                    fields=[
                        "material_id", "formula_pretty", "density", "volume",
                        "band_gap", "is_metal", "energy_above_hull", "symmetry"
                    ]
                )

                docs_by_formula = {}
                for doc in docs:
                    f = doc.formula_pretty
                    # Get the current best doc for this formula
                    current_best_doc = docs_by_formula.get(f)
                    
                    # If there is no doc yet, or the new doc is better, update
                    if not current_best_doc or (
                        getattr(doc, 'energy_above_hull', float('inf')) <
                        getattr(current_best_doc, 'energy_above_hull', float('inf'))
                    ):
                        docs_by_formula[f] = doc

                for doc in docs_by_formula.values():
                    symmetry = getattr(doc, "symmetry", None)
                    materials_data_list.append({
                        "formula": doc.formula_pretty,
                        "mp_formula": doc.formula_pretty,
                        "material_id": str(doc.material_id),
                        "mp_density": getattr(doc, "density", None),
                        "mp_volume": getattr(doc, "volume", None),
                        "mp_band_gap": getattr(doc, "band_gap", None),
                        "mp_is_metal": getattr(doc, "is_metal", None),
                        "mp_energy_above_hull": getattr(doc, "energy_above_hull", None),
                        "crystal_system": getattr(symmetry, "crystal_system", None).name if symmetry and hasattr(symmetry, "crystal_system") and hasattr(getattr(symmetry, "crystal_system"), "name") else None,
                        "mp_spacegroup": getattr(symmetry, "symbol", None) if symmetry else None,
                    })
            except Exception as e:
                print(f"An error occurred during Materials Project query for batch {i//batch_size + 1}: {e}")

    print(f"Finished Materials Project query. Fetched data for {len(materials_data_list)} materials.")

    if materials_data_list:
        mp_df = pd.DataFrame(materials_data_list)
        # Use the original formula for merging to handle any inconsistencies
        df = pd.merge(df, mp_df, on="formula", how="left")

    return df


def add_jarvis_features(df):
    """
    Enrich the dataframe with thermal-transport-relevant features from the JARVIS-DFT database.
    Includes density, elastic constants, bond lengths, packing, anisotropy, etc.
    """
    jdft_3d = jdata('dft_3d')
    formula_to_jarvis = {item['formula']: item for item in jdft_3d}
    jarvis_data = []
    formulas = df['formula'].unique()
    for formula in tqdm(formulas, desc="Fetching JARVIS Data"):
        if formula in formula_to_jarvis:
            item = formula_to_jarvis[formula]
            # Compute density from structure
            try:
                structure = Structure.from_dict(item['atoms'])
                density_val = structure.density  # g/cm^3
            except Exception:
                density_val = None
            jarvis_data.append({
                'formula': formula,
                'jarvis_bulk_modulus': item.get('bulk_modulus_vrh'),
                'jarvis_shear_modulus': item.get('shear_modulus_vrh'),
                'jarvis_band_gap': item.get('optb88vdw_bandgap'),
                'jarvis_formation_energy': item.get('formation_energy_peratom'),
                'jarvis_debye_temp': item.get('debye_temp'),
                'jarvis_density': density_val,
                'jarvis_avg_mass': item.get('avg_mass'),
                'jarvis_natoms': item.get('natoms'),
                'jarvis_packing_fraction': item.get('packing_fraction'),
                'jarvis_min_bond_length': item.get('min_bond_length'),
                'jarvis_avg_bond_length': item.get('avg_bond_length'),
                'jarvis_elastic_anisotropy': item.get('elastic_anisotropy'),
                'jarvis_is_metal': item.get('is_metal'),
                'jarvis_eps_electronic': item.get('eps_electronic'),
                'jarvis_eps_total': item.get('eps_total')
            })
    results_df = pd.DataFrame(jarvis_data)
    if not results_df.empty:
        results_df.dropna(
            how='all',
            subset=[
                'jarvis_bulk_modulus', 'jarvis_shear_modulus', 'jarvis_band_gap',
                'jarvis_formation_energy', 'jarvis_debye_temp', 'jarvis_density',
                'jarvis_avg_mass', 'jarvis_natoms', 'jarvis_packing_fraction'
            ],
            inplace=True
        )
        df = pd.merge(df, results_df, on='formula', how='left')
    print(f"Successfully fetched data for {len(results_df)} of {len(formulas)} unique formulas from JARVIS.")
    return df



def add_pca_features(X, n_components=5):
    """
    Apply PCA to the feature set and add the top principal components as new features.

    Args:
        X (pd.DataFrame): Feature matrix.
        n_components (int): Number of principal components to retain.

    Returns:
        pd.DataFrame: Feature matrix with PCA components added.
    """
    pca = PCA(n_components=n_components, random_state=42)
    pca_components = pca.fit_transform(X)
    pca_df = pd.DataFrame(
        pca_components, columns=[f'pca_component_{i+1}' for i in range(n_components)], index=X.index
    )
    return pd.concat([X, pca_df], axis=1)


def classify_spacegroup_by_number(sg_number):
    """Classifies the crystal system based on the space group number."""
    if pd.isna(sg_number):
        return 'Unknown'
    sg_number = int(sg_number)
    if 1 <= sg_number <= 2:
        return 'Triclinic'
    elif 3 <= sg_number <= 15:
        return 'Monoclinic'
    elif 16 <= sg_number <= 74:
        return 'Orthorhombic'
    elif 75 <= sg_number <= 142:
        return 'Tetragonal'
    elif 143 <= sg_number <= 167:
        return 'Trigonal'
    elif 168 <= sg_number <= 194:
        return 'Hexagonal'
    elif 195 <= sg_number <= 230:
        return 'Cubic'
    return 'Unknown'

def classify_material(formula):
    from pymatgen.core import Composition

    try:
        composition = Composition(formula)
        elements = {el.symbol for el in composition.elements}

        if len(elements) == 1:
            return "Elemental"
        elif "O" in elements:
            return "Oxide"
        elif elements & {"S", "Se", "Te"}:
            return "Chalcogenide"
        elif elements & {"F", "Cl", "Br", "I"}:
            return "Halide"
        elif "N" in elements:
            return "Nitride"
        elif "C" in elements:
            return "Carbide"
        elif "B" in elements:
            return "Boride"
        elif elements & {"P", "As", "Sb", "Bi"}:
            return "Pnictide"
        elif "H" in elements:
            return "Hydride"
        else:
            return "Intermetallic/Alloy"
    except Exception:
        return "Unknown"

def density(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines 'mp_density' and 'jarvis_density' columns into a single 'density' column.
    Prioritizes 'mp_density' if available, otherwise uses 'jarvis_density'.
    Drops the original density columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing 'mp_density' and 'jarvis_density'.

    Returns:
        pd.DataFrame: DataFrame with combined 'density' column.
    """
    df['density'] = df[['mp_density', 'jarvis_density']].apply(
        lambda row: row['mp_density'] if pd.notnull(row['mp_density']) else row['jarvis_density'], axis=1
    )
    df.drop(columns=['mp_density', 'jarvis_density'], inplace=True, errors='ignore')
    return df
