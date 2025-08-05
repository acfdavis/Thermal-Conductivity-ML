import pandas as pd
import numpy as np
from src.utils import perform_normality_tests

def test_perform_normality_tests():
    # Create a sample DataFrame
    data = {
        'normal_dist': np.random.normal(0, 1, 1000),
        'uniform_dist': np.random.uniform(0, 1, 1000),
        'skewed_dist': np.random.exponential(1, 1000)
    }
    df = pd.DataFrame(data)

    # Run the normality test
    results = perform_normality_tests(df)

    # Print the results
    print(results)

    # Assertions to validate the function
    assert 'Feature' in results.columns
    assert 'P-Value' in results.columns
    assert 'Is Normal' in results.columns
    assert len(results) == len(df.columns)

if __name__ == "__main__":
    test_perform_normality_tests()
