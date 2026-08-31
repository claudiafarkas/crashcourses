"""Shared dataset loaders for the foundations and models notebooks."""

import os
import kagglehub
import pandas as pd


def load_housing_data() -> pd.DataFrame:
    """Download the Kaggle Housing dataset and return it as a DataFrame."""
    dataset_path = kagglehub.dataset_download("yasserh/housing-prices-dataset")

    csv_files = [
        os.path.join(root, filename)
        for root, _, files in os.walk(dataset_path)
        for filename in files
        if filename.lower().endswith(".csv")
    ]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {dataset_path}")

    return pd.read_csv(csv_files[0])