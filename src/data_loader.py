"""
====================================================
Module : data_loader.py
Project: Airline Passenger Forecasting
Purpose: Load and validate the dataset
====================================================
"""

from pathlib import Path

import pandas as pd

# Project root = the folder that contains /src, /Data, /models
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "Data" / "airline-passengers.csv"


class DataLoader:
    """
    A class responsible for loading the time series dataset.
    """

    def __init__(self, file_path=None):
        """
        Parameters
        ----------
        file_path : str or Path, optional
            Path of the CSV dataset. Defaults to Data/airline-passengers.csv
            inside the project, so this works the same on any machine
            (local, Streamlit Cloud, etc.) without editing the code.
        """
        self.file_path = Path(file_path) if file_path else DEFAULT_DATA_PATH

    def load_data(self, verbose=False):
        """
        Load the dataset and return a DataFrame.

        Parameters
        ----------
        verbose : bool
            When True, print dataset diagnostics (head, shape, dtypes...).
            Kept quiet by default so it doesn't spam a deployed app's logs.

        Returns
        -------
        pandas.DataFrame
        """
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at:\n{self.file_path}\n"
                "Make sure the Data/ folder was pushed to the repo."
            )

        df = pd.read_csv(self.file_path)

        if verbose:
            print("Dataset Loaded Successfully.")
            print(df.head())
            print(df.shape)
            print(df.info())
            print(df.isnull().sum())

        # Normalize column names/types
        df["Month"] = pd.to_datetime(df["Month"])
        df.set_index("Month", inplace=True)

        return df


if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_data(verbose=True)
    print("\nProcessed Dataset")
    print(df.head())
