"""
====================================================
Module : preprocessing.py
Project: Airline Passenger Forecasting
Purpose: Scale the dataset using MinMaxScaler
====================================================
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_SCALER_PATH = MODELS_DIR / "scaler.pkl"


class Preprocessor:
    """
    Preprocess the time series dataset.
    """

    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def scale_data(self, df, save_scaler=True):
        """
        Scale the Passengers column.

        Parameters
        ----------
        df : pandas.DataFrame
        save_scaler : bool
            When True (default), persist the fitted scaler to
            models/scaler.pkl. Set False at inference time if you
            only want to *load* an existing scaler elsewhere.

        Returns
        -------
        scaled_df : pandas.DataFrame
        """
        # Support either column naming: "Passengers" or lowercase "passengers"
        col_name = "Passengers" if "Passengers" in df.columns else "passengers"

        scaled_values = self.scaler.fit_transform(df[[col_name]])

        scaled_df = pd.DataFrame(
            scaled_values,
            columns=["Passengers"],
            index=df.index,
        )

        if save_scaler:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.scaler, DEFAULT_SCALER_PATH)

        return scaled_df


if __name__ == "__main__":
    from src.data_loader import DataLoader

    loader = DataLoader()
    df = loader.load_data()

    preprocessor = Preprocessor()
    scaled_df = preprocessor.scale_data(df)

    print("\nScaled Dataset")
    print(scaled_df.head())
