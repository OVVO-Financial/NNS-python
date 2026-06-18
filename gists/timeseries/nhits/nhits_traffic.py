"""N-HiTS is a neural breakthrough -- NNS forecasts the same traffic series with no network.

N-HiTS (Neural Hierarchical Interpolation for Time Series, AAAI 2023) is a deep
learning model -- multi-rate input sampling plus hierarchical interpolation, built
on N-BEATS -- that reports ~20% better long-horizon accuracy than Transformer
forecasters at a fraction of the compute. It still needs a GPU-era training loop.

This script forecasts the last 120 steps of the hourly traffic-volume series with
NNS.ARMA -- no neural network, no training loop, no hyperparameter search -- and
lands a competitive MAE in a handful of lines. Seasonal periods are discovered
directly from the data by nns_seas.

See: https://www.datasciencewithmarco.com/blog/all-about-n-hits-the-latest-breakthrough-in-time-series-forecasting
"""

import matplotlib

matplotlib.use("Agg")  # headless; drop this line for an interactive window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

import nns as NNS

# Load data
url = (
    "https://raw.githubusercontent.com/OVVO-Financial/NNS/"
    "Data-and-Simulation-Routines/Datasets/traffic_data.csv"
)
df = pd.read_csv(url)

# Columns are 'date_time' and 'traffic_volume'
y = df["traffic_volume"].values

# Train / test split -- hold out the last 120 points
train_set = y[:-120]
test_set = y[-120:]

# Discover seasonal periods from the training data
seas_result = NNS.nns_seas(train_set, plot=False)
periods = seas_result.get("periods", [])

# Optimize the seasonal ARMA combination (MAE objective to match the benchmark)
nns_estimates = NNS.nns_arma_optim(
    variable=train_set,
    h=120,
    seasonal_factor=periods,
    obj_fn=lambda predicted, actual: np.mean(np.abs(predicted - actual)),
    objective="min",
    plot=False,
    negative_values=False,
)

# Evaluate
mae = mean_absolute_error(test_set, nns_estimates["results"])
print(f"MAE on traffic data: {mae}")

# Plot actual vs. forecast
plt.figure(figsize=(10, 6))
plt.plot(test_set, label="Actual Traffic", color="black", linewidth=2)
plt.plot(nns_estimates["results"], label="NNS Forecast", color="blue", linewidth=2)
plt.legend()
plt.title("Traffic Volume Forecast (Actual Data)")
plt.tight_layout()
plt.savefig("nhits_traffic_forecast.png", dpi=120)  # so the README can show the figure
plt.show()
