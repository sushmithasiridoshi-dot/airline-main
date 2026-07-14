"""
====================================================
Module : train_test_split.py
Project: Airline Passenger Forecasting
Purpose: Split sequences into Train and Test sets
====================================================
"""


class TimeSeriesSplit:

    def __init__(self, train_size=0.8):
        self.train_size = train_size

    def split(self, X, y):
        split_index = int(len(X) * self.train_size)

        X_train = X[:split_index]
        X_test = X[split_index:]

        y_train = y[:split_index]
        y_test = y[split_index:]

        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from src.data_loader import DataLoader
    from src.preprocessing import Preprocessor
    from src.sequence_generator import SequenceGenerator

    loader = DataLoader()
    df = loader.load_data()

    preprocessor = Preprocessor()
    scaled_df = preprocessor.scale_data(df)

    generator = SequenceGenerator(sequence_length=12)
    X, y = generator.create_sequences(scaled_df)

    splitter = TimeSeriesSplit(train_size=0.8)
    X_train, X_test, y_train, y_test = splitter.split(X, y)

    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")
