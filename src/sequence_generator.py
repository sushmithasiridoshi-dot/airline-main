"""
====================================================
Module : sequence_generator.py
Project: Airline Passenger Forecasting
Purpose: Generate sequences for RNN/LSTM
====================================================
"""

import numpy as np


class SequenceGenerator:
    """
    Convert a time series into input-output sequences.
    """

    def __init__(self, sequence_length=12):
        self.sequence_length = sequence_length

    def create_sequences(self, scaled_df):
        X = []
        y = []

        data = scaled_df.values

        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])

        X = np.array(X)
        y = np.array(y)

        return X, y


if __name__ == "__main__":
    from src.data_loader import DataLoader
    from src.preprocessing import Preprocessor

    loader = DataLoader()
    df = loader.load_data()

    preprocessor = Preprocessor()
    scaled_df = preprocessor.scale_data(df)

    generator = SequenceGenerator(sequence_length=12)
    X, y = generator.create_sequences(scaled_df)

    print(f"Input Shape : {X.shape}")
    print(f"Output Shape : {y.shape}")
