import pandas as pd

def convert_csv_to_parquet(csv_path, parquet_path):
    """
    Reads a pandas DataFrame from a .csv file, cleans up mixed-type columns,
    and saves it to a .parquet file.

    Args:
        csv_path (str): The file path of the input .csv file.
        parquet_path (str): The file path for the output .parquet file.
    """
    try:
        # 1. Read the .csv file. low_memory=False helps pandas guess dtypes better.
        print(f"Reading CSV file from: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)

        # 2. Identify and fix mixed-type columns.
        # The DtypeWarning mentioned columns 0, 6, and 69.
        # We will convert them to strings to ensure type consistency.
        problem_columns_indices = [0, 6, 69]
        for col_idx in problem_columns_indices:
            # Check if the column index is valid
            if col_idx < len(df.columns):
                col_name = df.columns[col_idx]
                print(f"Converting column '{col_name}' (index {col_idx}) to string type to resolve mixed types.")
                df[col_name] = df[col_name].astype(str)

        # 3. Write the cleaned DataFrame to a .parquet file
        print(f"Writing parquet file to: {parquet_path}")
        df.to_parquet(parquet_path, engine='pyarrow', index=False)

        print("\nConversion successful!")
        print(f"DataFrame shape: {df.shape}")

    except FileNotFoundError:
        print(f"Error: The file was not found at {csv_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Usage Example ---
# Define the input and output file paths without extra quotes.
input_csv = 'C:\\Users\\angel\\thermal_conductivity\\ml_conductivity_project\\data\\raw\\thermoml_data.csv'
output_parquet = 'C:\\Users\\angel\\Thermal-Conductivity-ML\\data\\raw\\thermoml_data.parquet'

# Run the conversion function
convert_csv_to_parquet(input_csv, output_parquet)