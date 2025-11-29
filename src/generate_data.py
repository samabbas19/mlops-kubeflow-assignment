"""
Script to generate/download the Boston Housing dataset
"""
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

def generate_boston_data():
    """
    Generate a dataset similar to Boston Housing using California Housing
    or create synthetic data
    """
    # Using California Housing as alternative (Boston dataset deprecated)
    data = fetch_california_housing()

    # Create DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Save to CSV
    df.to_csv('data/raw_data.csv', index=False)
    print(f"Dataset created with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())

if __name__ == "__main__":
    generate_boston_data()
