import os
import subprocess
import pandas as pd
import pytest

# Define project root and paths to artifacts
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SCRIPT_PATH = os.path.join(PROJECT_ROOT, 'scripts', 'predict_from_csv.py')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'tuned_xgboost_model.joblib')
SCALER_PATH = os.path.join(PROJECT_ROOT, 'models', 'scaler.joblib')
FEATURES_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'selected_features_xgb.json')
TEST_INPUT_PATH = os.path.join(PROJECT_ROOT, 'tests', 'temp_test_input.csv')
TEST_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'tests', 'temp_test_output.csv')

@pytest.fixture(scope="module")
def setup_teardown_test_files():
    """
    Pytest fixture to create a temporary input file for the test
    and clean it up after the test runs.
    """
    # --- SETUP ---
    # Create a sample input DataFrame that matches the required format
    sample_data = {
        "formula": ["NaCl", "Fe2O3"],
        "temperature": [300, 400],
        "pressure": [101325, 101325],
        "phase": ["solid", "solid"]
    }
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv(TEST_INPUT_PATH, index=False)
    
    # Yield control to the test function
    yield
    
    # --- TEARDOWN ---
    # Clean up the created files
    if os.path.exists(TEST_INPUT_PATH):
        os.remove(TEST_INPUT_PATH)
    if os.path.exists(TEST_OUTPUT_PATH):
        os.remove(TEST_OUTPUT_PATH)

def test_prediction_script_runs_successfully(setup_teardown_test_files):
    """
    Tests the full execution of the predict_from_csv.py script.
    """
    # Check if necessary artifacts exist before running the test
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}. Please run notebook 5."
    assert os.path.exists(SCALER_PATH), f"Scaler file not found at {SCALER_PATH}. Please run notebook 5."
    assert os.path.exists(FEATURES_PATH), f"Features file not found at {FEATURES_PATH}. Please run notebook 4."

    # Construct the command to run the script
    command = [
        "python", SCRIPT_PATH,
        "--model", MODEL_PATH,
        "--scaler", SCALER_PATH,
        "--features", FEATURES_PATH,
        "--input", TEST_INPUT_PATH,
        "--output", TEST_OUTPUT_PATH
    ]
    
    # Execute the script as a subprocess
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    
    # Assert that the script ran successfully
    assert result.returncode == 0, f"Script failed to run. Error: {result.stderr}"
    assert "Success! Predictions saved" in result.stdout, f"Script did not report success. Output: {result.stdout}"
    
    # Assert that the output file was created
    assert os.path.exists(TEST_OUTPUT_PATH), "Prediction script did not create the output file."
    
    # Validate the contents of the output file
    output_df = pd.read_csv(TEST_OUTPUT_PATH)
    
    # Check for the prediction column
    assert "predicted_thermal_conductivity" in output_df.columns, "Output CSV is missing the prediction column."
    
    # Check that predictions are not null
    assert output_df["predicted_thermal_conductivity"].isnull().sum() == 0, "Predictions contain null values."
    
    # Check that the number of predictions matches the number of inputs
    assert len(output_df) == 2, "Number of predictions does not match number of inputs."
