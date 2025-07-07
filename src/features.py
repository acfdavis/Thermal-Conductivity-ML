"""
features.py

This module contains functions to generate and transform features for thermal conductivity modeling.
"""

import os
import re
from dotenv import load_dotenv
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from jarvis.db.figshare import data as jdata
from tqdm import tqdm
import pandas as pd
import joblib
from mp_api.client import MPRester


# Load environment variables from .env file
load_dotenv()
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital

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

    valence_featurizer = ValenceOrbital()
    valence_featurizer.set_n_jobs(1)
    df = valence_featurizer.featurize_dataframe(df, col_id="composition")

    df = df.drop('composition', axis=1)
    return df

def normalize_column_names(df):
    df.columns = [re.sub(r"[\s\[\]<>]", "_", col) for col in df.columns]
    return df

def run():
    # Your pipeline logic goes here, e.g.
    df_raw = load_and_merge_data()
    df_clean = impute_and_clean_data(df_raw)
    df_features = add_composition_features(df_clean)
    print(df_features.head())

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    run()

    
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
                    if f not in docs_by_formula or (
                        getattr(doc, 'energy_above_hull', float('inf')) <
                        getattr(docs_by_formula[f], 'energy_above_hull', float('inf'))
                    ):
                        docs_by_formula[f] = doc

                for doc in docs_by_formula.values():
                    symmetry = getattr(doc, "symmetry", None)
                    materials_data_list.append({
                        "formula": doc.formula_pretty,
                        "mp_formula": doc.formula_pretty,
                        "material_id": str(doc.material_id),
                        "mp_density": getattr(doc, 'density', None),
                        "mp_volume": getattr(doc, 'volume', None),
                        "mp_band_gap": getattr(doc, 'band_gap', None),
                        "is_metal": getattr(doc, 'is_metal', None),
                        "energy_above_hull": getattr(doc, 'energy_above_hull', None),
                        "crystal_system": getattr(symmetry, 'crystal_system', None) if symmetry else None,
                        "spacegroup": getattr(symmetry, 'number', None) if symmetry else None
                    })

            except Exception as e:
                print(f"An error occurred during Materials Project query for batch {i//batch_size}: {e}")

    print(f"Finished Materials Project query. Fetched data for {len(materials_data_list)} materials.")

    if materials_data_list:
        materials_df = pd.DataFrame(materials_data_list)
        materials_df = materials_df.drop_duplicates(subset=['formula'])
        df = pd.merge(df, materials_df, on="formula", how="left")

    return df


def add_jarvis_features(df):
    """
    Enrich the dataframe with features from the JARVIS-DFT database.
    """
    jdft_3d = jdata('dft_3d')
    formula_to_jarvis = {item['formula']: item for item in jdft_3d}
    jarvis_data = []
    formulas = df['formula'].unique()
    for formula in tqdm(formulas, desc="Fetching JARVIS Data"):
        if formula in formula_to_jarvis:
            item = formula_to_jarvis[formula]
            jarvis_data.append({
                'formula': formula,
                'jarvis_bulk_modulus': item.get('bulk_modulus_vrh'),
                'jarvis_shear_modulus': item.get('shear_modulus_vrh'),
                'jarvis_band_gap': item.get('optb88vdw_bandgap'),
                'jarvis_formation_energy': item.get('formation_energy_peratom'),
                'jarvis_debye_temp': item.get('debye_temp'),
                'jarvis_eps_electronic': item.get('eps_electronic'),
                'jarvis_eps_total': item.get('eps_total')
            })

    results_df = pd.DataFrame(jarvis_data)
    if not results_df.empty:
        results_df.dropna(how='all', subset=['jarvis_bulk_modulus', 'jarvis_shear_modulus', 'jarvis_band_gap', 'jarvis_formation_energy', 'jarvis_debye_temp', 'jarvis_eps_electronic', 'jarvis_eps_total'], inplace=True)
        df = pd.merge(df, results_df, on='formula', how='left')
    print(f"Successfully fetched data for {len(results_df)} of {len(formulas)} unique formulas from JARVIS.")
    return df



def featurize_data(df, composition_col='formula', cache_path=None):
    """
    Orchestrates the feature engineering pipeline.
    
    If `cache_path` is provided and exists, load the cached result.
    Otherwise, run the pipeline and save to cache (if path provided).
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        return joblib.load(cache_path)

    print("Starting feature engineering...")

    if composition_col not in df.columns:
        raise ValueError(f"Composition column '{composition_col}' not found in DataFrame.")

    df_comp = add_composition_features(df, composition_column=composition_col)
    print("Composition features added.")

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

    # Add crystal_structure column
    if 'crystal_system' in df_jarvis.columns:
        df_jarvis['crystal_structure'] = df_jarvis['crystal_system'].fillna('Unknown')
    elif 'MagpieData_mean_SpaceGroupNumber' in df_jarvis.columns:
        df_jarvis['crystal_structure'] = df_jarvis['MagpieData_mean_SpaceGroupNumber'].apply(classify_spacegroup_by_number)
    else:
        df_jarvis['crystal_structure'] = 'Unknown'


    print("Feature engineering complete.")
    # Normalize column names
    df_jarvis = normalize_column_names(df_jarvis)
    if cache_path:
        joblib.dump(df_jarvis, cache_path)
        print(f"Features cached to {cache_path}.")

    return df_jarvis



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
