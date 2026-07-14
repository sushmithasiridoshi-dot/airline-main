"""
====================================================
Module : forecast.py
Project: Airline Passenger Forecasting
Purpose: Forecast Future Passenger Counts
====================================================
"""

from pathlib import Path

import numpy as np
import joblib

from tensorflow.keras.models import load_model

from src.data_loader import DataLoader
from src.preprocessing import Preprocessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_model.keras"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"


class Forecaster:

    def __init__(self, model_path=None, scaler_path=None, sequence_length=12):
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.scaler_path = Path(scaler_path) if scaler_path else SCALER_PATH
        self.sequence_length = sequence_length

    def forecast(self, future_months=12):
        loader = DataLoader()
        df = loader.load_data()

        preprocessor = Preprocessor()
        scaled_df = preprocessor.scale_data(df, save_scaler=False)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at:\n{self.model_path}\n"
                "Run `python -m src.train` first, or add the "
                "lstm_model.keras file to the models/ folder."
            )

        model = load_model(str(self.model_path))

        if not self.scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler not found at:\n{self.scaler_path}\n"
                "Run training/preprocessing first, or add scaler.pkl "
                "to the models/ folder."
            )

        scaler = joblib.load(self.scaler_path)

        last_sequence = scaled_df.values[-self.sequence_length:]

        future_predictions = []

        for _ in range(future_months):
            input_data = last_sequence.reshape(1, self.sequence_length, 1)
            prediction = model.predict(input_data, verbose=0)
            future_predictions.append(prediction[0, 0])
            last_sequence = np.vstack((last_sequence[1:], prediction))

        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_predictions = scaler.inverse_transform(future_predictions)

        return future_predictions


if __name__ == "__main__":
    forecaster = Forecaster()
    future = forecaster.forecast(future_months=12)

    print("\nNext 12 Month Forecast\n")
    for i, value in enumerate(future, start=1):
        print(f"Month {i} : {value[0]:.2f}")
