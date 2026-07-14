"""
====================================================
Module : predict.py
Project: Airline Passenger Forecasting
Purpose: Predict Passenger Counts
====================================================
"""

from pathlib import Path

import joblib

from tensorflow.keras.models import load_model

from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.sequence_generator import SequenceGenerator
from src.train_test_split import TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_model.keras"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"


class Predictor:

    def __init__(self, model_path=None, scaler_path=None):
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.scaler_path = Path(scaler_path) if scaler_path else SCALER_PATH

    def predict(self):
        loader = DataLoader()
        df = loader.load_data()

        preprocessor = Preprocessor()
        scaled_df = preprocessor.scale_data(df, save_scaler=False)

        generator = SequenceGenerator(sequence_length=12)
        X, y = generator.create_sequences(scaled_df)

        splitter = TimeSeriesSplit(train_size=0.80)
        X_train, X_test, y_train, y_test = splitter.split(X, y)

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

        predictions = model.predict(X_test, verbose=0)

        predictions = scaler.inverse_transform(predictions)
        y_test = scaler.inverse_transform(y_test)

        return y_test, predictions


if __name__ == "__main__":
    predictor = Predictor()
    actual, predicted = predictor.predict()

    print("\nFirst 10 Predictions\n")
    for i in range(10):
        print(f"Actual : {actual[i][0]:.2f}   Predicted : {predicted[i][0]:.2f}")
