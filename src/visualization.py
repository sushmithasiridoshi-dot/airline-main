"""
====================================================
Module : visualization.py
Project: Airline Passenger Forecasting
Purpose: Visualize Model Performance (offline/CLI use)
====================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


class Visualizer:
    """
    Visualize the model performance. Saves PNGs to outputs/.
    Not used by the Streamlit app (which renders interactive Plotly
    charts instead) — this is for offline/CLI diagnostics.
    """

    def __init__(self):
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    def plot_training_loss(self, history):
        plt.figure(figsize=(10, 5))
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Training Loss vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(OUTPUTS_DIR / "loss_curve.png")
        plt.close()

    def plot_predictions(self, actual, predicted):
        plt.figure(figsize=(12, 6))
        plt.plot(actual, label="Actual", linewidth=2)
        plt.plot(predicted, label="Predicted", linewidth=2)
        plt.title("Actual vs Predicted Passenger Count")
        plt.xlabel("Time")
        plt.ylabel("Passengers")
        plt.legend()
        plt.grid(True)
        plt.savefig(OUTPUTS_DIR / "prediction.png")
        plt.close()

    def plot_future_forecast(self, future_values):
        plt.figure(figsize=(12, 6))
        plt.plot(future_values, marker="o", linewidth=2)
        plt.title("Future Passenger Forecast")
        plt.xlabel("Future Months")
        plt.ylabel("Passengers")
        plt.grid(True)
        plt.savefig(OUTPUTS_DIR / "forecast.png")
        plt.close()
