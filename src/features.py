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

from .data import impute_and_clean_data, load_and_merge_data #remove "." when running notebook 4


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

    # Normalize accidental leading/trailing whitespace in column headers (e.g. ' space_group')
    original_cols = df.columns.tolist()
    df.columns = [c.strip() for c in df.columns]
    if original_cols != df.columns.tolist():
        print("Stripped whitespace from column names:", original_cols, "->", df.columns.tolist())

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

    skip_jarvis = os.getenv("TCML_SKIP_JARVIS") == "1"
    if skip_jarvis:
        print("Skipping JARVIS feature retrieval (TCML_SKIP_JARVIS=1).")
        df_jarvis = df_comp.copy()
    else:
        df_jarvis = add_jarvis_features(df_comp)
        print("JARVIS features added.")

    # Add chemistry column using classify_material on formula
    df_jarvis['chemistry'] = df_jarvis[composition_col].apply(classify_material)

    # Determine crystal structure. If user supplied an explicit space_group column (1-230), prefer that.
    user_sg_col = None
    for cand in ['space_group', 'space_group_number', 'spacegroup', 'spacegroup_number']:
        if cand in df_jarvis.columns:
            user_sg_col = cand
            break

    if user_sg_col:
        # Coerce to numeric safely
        df_jarvis[user_sg_col] = pd.to_numeric(df_jarvis[user_sg_col], errors='coerce')
        df_jarvis['crystal_structure'] = df_jarvis[user_sg_col].apply(classify_spacegroup_by_number)
        df_jarvis.rename(columns={user_sg_col: 'user_space_group_number'}, inplace=True)
        print(f"Used user-provided space group numbers from column '{user_sg_col}' to assign crystal_structure.")
    else:
        # Fallback to Magpie-derived mean space group number
        df_jarvis['crystal_structure'] = df_jarvis['MagpieData_mean_SpaceGroupNumber'].apply(classify_spacegroup_by_number)
        print("Assigned crystal_structure from MagpieData_mean_SpaceGroupNumber (no user space_group provided).")

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
    Enriches the DataFrame with materials properties from the Materials Project on a
    ROW-BY-ROW basis to handle polymorphs correctly.

    For each row, it attempts to find a structure in Materials Project that matches
    both the formula and the user-provided space group. If no exact match is found,
    it falls back to the lowest-energy polymorph for that formula.
    """
    from mp_api.client import MPRester
    from tqdm import tqdm

    materials_data_list = []
    print(f"Starting row-by-row Materials Project query for {len(df)} entries...")

    with MPRester(api_key=api_key) as mpr:
        # Use tqdm to iterate with a progress bar
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Querying MP"):
            formula = row['formula']
            # Use the already-normalized space group column if it exists
            user_sg = row.get('user_space_group_number')

            docs = []
            # 1. Try to find an exact match for formula AND space group
            if pd.notna(user_sg):
                try:
                    docs = mpr.materials.summary.search(
                        formula=formula,
                        symmetry_space_group_number=int(user_sg),
                        fields=["material_id", "formula_pretty", "density", "volume",
                                "band_gap", "is_metal", "energy_above_hull", "symmetry"]
                    )
                except Exception:
                    docs = [] # Query failed, proceed to fallback

            # 2. Fallback: If no exact match, query by formula only
            if not docs:
                try:
                    docs = mpr.materials.summary.search(
                        formula=formula,
                        fields=["material_id", "formula_pretty", "density", "volume",
                                "band_gap", "is_metal", "energy_above_hull", "symmetry"]
                    )
                except Exception as e:
                    print(f"Warning: MP query failed for formula {formula}: {e}")
                    docs = []

            # 3. Process the results
            if docs:
                # Pick the best available structure (lowest energy above hull)
                docs.sort(key=lambda d: getattr(d, "energy_above_hull", float("inf")))
                chosen = docs[0]
                symm = getattr(chosen, "symmetry", None)
                
                # Check if the chosen structure matches the user's requested space group
                mp_sg = getattr(symm, "space_group_number", None)
                sg_match = pd.notna(user_sg) and pd.notna(mp_sg) and int(user_sg) == int(mp_sg)

                materials_data_list.append({
                    "mp_formula": chosen.formula_pretty,
                    "material_id": str(chosen.material_id),
                    "mp_density": getattr(chosen, "density", None),
                    "mp_volume": getattr(chosen, "volume", None),
                    "mp_band_gap": getattr(chosen, "band_gap", None),
                    "mp_is_metal": getattr(chosen, "is_metal", None),
                    "mp_energy_above_hull": getattr(chosen, "energy_above_hull", None),
                    "crystal_system": getattr(symm, "crystal_system", None).name if symm and getattr(symm, "crystal_system", None) else None,
                    "mp_spacegroup_number": mp_sg,
                    "mp_spacegroup_match_user": sg_match
                })
            else:
                # Append a dictionary of NaNs if no data was found
                materials_data_list.append({col: None for col in [
                    "mp_formula", "material_id", "mp_density", "mp_volume", "mp_band_gap",
                    "mp_is_metal", "mp_energy_above_hull", "crystal_system", "mp_spacegroup_number",
                    "mp_spacegroup_match_user"
                ]})

    print(f"Finished Materials Project query. Processed {len(materials_data_list)} rows.")
    if materials_data_list:
        mp_df = pd.DataFrame(materials_data_list, index=df.index)
        df = pd.concat([df, mp_df], axis=1)

    return df


def add_jarvis_features(df):
    """
    Enrich the dataframe with features from JARVIS-DFT on a ROW-BY-ROW basis.
    It prioritizes structures matching the user-provided space group.
    """
    from pymatgen.core import Structure
    from tqdm import tqdm

    # Pre-process the JARVIS database for efficient lookup
    jdft_3d = jdata('dft_3d')
    formula_to_jarvis_list = {}
    for item in jdft_3d:
        formula = item['formula']
        if formula not in formula_to_jarvis_list:
            formula_to_jarvis_list[formula] = []
        formula_to_jarvis_list[formula].append(item)

    jarvis_data_list = []
    print(f"Starting row-by-row JARVIS query for {len(df)} entries...")

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Fetching JARVIS Data"):
        formula = row['formula']
        user_sg = row.get('user_space_group_number')
        
        candidates = formula_to_jarvis_list.get(formula, [])
        chosen_item = None

        # 1. Try to find an exact match for formula AND space group
        if pd.notna(user_sg) and candidates:
            sg_matched = [item for item in candidates if item.get('spacegroup_number') == int(user_sg)]
            if sg_matched:
                # Among matches, pick lowest formation energy
                sg_matched.sort(key=lambda item: item.get('formation_energy_peratom', float('inf')))
                chosen_item = sg_matched[0]

        # 2. Fallback: If no exact match, pick the lowest formation energy polymorph
        if chosen_item is None and candidates:
            candidates.sort(key=lambda item: item.get('formation_energy_peratom', float('inf')))
            chosen_item = candidates[0]

        # 3. Process the chosen item
        if chosen_item:
            try:
                structure = Structure.from_dict(chosen_item['atoms'])
                density_val = structure.density
            except Exception:
                density_val = None
            
            jarvis_data_list.append({
                'jarvis_bulk_modulus': chosen_item.get('bulk_modulus_vrh'),
                'jarvis_shear_modulus': chosen_item.get('shear_modulus_vrh'),
                'jarvis_band_gap': chosen_item.get('optb88vdw_bandgap'),
                'jarvis_formation_energy': chosen_item.get('formation_energy_peratom'),
                'jarvis_debye_temp': chosen_item.get('debye_temp'),
                'jarvis_density': density_val,
            })
        else:
            # Append a dictionary of NaNs if no data was found
            jarvis_data_list.append({col: None for col in [
                'jarvis_bulk_modulus', 'jarvis_shear_modulus', 'jarvis_band_gap',
                'jarvis_formation_energy', 'jarvis_debye_temp', 'jarvis_density'
            ]})

    print(f"Finished JARVIS query. Processed {len(jarvis_data_list)} rows.")
    if jarvis_data_list:
        jarvis_df = pd.DataFrame(jarvis_data_list, index=df.index)
        df = pd.concat([df, jarvis_df], axis=1)

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
