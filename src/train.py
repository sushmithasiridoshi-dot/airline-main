"""
====================================================
Module : train.py
Project: Airline Passenger Forecasting
Purpose: Train the LSTM Model
====================================================
"""

from pathlib import Path

from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.sequence_generator import SequenceGenerator
from src.train_test_split import TimeSeriesSplit
from src.model import ModelBuilder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


class ModelTrainer:

    def __init__(self):
        pass

    def train(self):
        loader = DataLoader()
        df = loader.load_data()

        preprocessor = Preprocessor()
        scaled_df = preprocessor.scale_data(df)

        generator = SequenceGenerator(sequence_length=12)
        X, y = generator.create_sequences(scaled_df)

        splitter = TimeSeriesSplit(train_size=0.80)
        X_train, X_test, y_train, y_test = splitter.split(X, y)

        builder = ModelBuilder(model_type="lstm", input_shape=(12, 1))
        model = builder.build_model()

        print("\nTraining Started...\n")

        history = model.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=8,
            validation_data=(X_test, y_test),
            verbose=1,
        )

        print("\nTraining Completed Successfully.")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.save(str(MODELS_DIR / "lstm_model.keras"))

        print("\nModel Saved Successfully.")

        return model, history


if __name__ == "__main__":
    trainer = ModelTrainer()
    model, history = trainer.train()
