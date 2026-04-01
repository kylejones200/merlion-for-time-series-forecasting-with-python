# Description: Short example for Merlion for Time Series Forecasting with Python.



from data_io import read_csv
from merlion.evaluate.forecast import ForecastMetric
from merlion.models.anomaly.isolation_forest import IsolationForest, IsolationForestConfig
from merlion.models.defaults import DefaultForecasterConfig
from merlion.models.factory import ModelFactory
from merlion.models.forecast.arima import Arima, ArimaConfig
from merlion.models.forecast.arima import ArimaConfig
from merlion.models.forecast.prophet import Prophet, ProphetConfig
from merlion.models.forecast.prophet import ProphetConfig
from merlion.transform.moving_average import MovingAverage, DifferenceTransform
from merlion.transform.normalize import MeanVarNormalize
from merlion.transform.resample import TemporalResample
from merlion.transform.sequence import TransformSequence
from merlion.utils import TimeSeries
from merlion.utils.time_series import TimeSeries
from scipy.stats import norm
from sklearn.model_selection import TimeSeriesSplit
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)



# Load dataset
url = "https://raw.githubusercontent.com/kylejones200/time_series/main/ercot_load_data.csv"
df = read_csv(url)
# Convert time column to datetime format
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
# Resample to hourly frequency
df = df.resample('H').mean()
df['values'] = df['values'].interpolate()
# Convert to Merlion TimeSeries format
ts = TimeSeries.from_pd(df)
logger.info(ts)


# Initialize Prophet with optimized hyperparameters
prophet_config = ProphetConfig(
    add_seasonality="auto",
    weekly_seasonality=True,
    daily_seasonality=True,
    changepoint_prior_scale=0.05  # Better trend detection
)
prophet_model = Prophet(prophet_config)
# Initialize ARIMA (manually tuned order)
arima_model = Arima(ArimaConfig(order=(2, 1, 2), target_seq_index=0))

# Split data into training and test sets
train_ratio = 0.8  # 80% training, 20% testing
split_idx = int(len(df) * train_ratio)
train_data = TimeSeries.from_pd(df.iloc[:split_idx])
test_data = TimeSeries.from_pd(df.iloc[split_idx:])

# Train the models
prophet_model.train(train_data)
arima_model.train(train_data)


# Generate forecasts
prophet_forecast, _ = prophet_model.forecast(test_data.time_stamps)
arima_forecast, _ = arima_model.forecast(test_data.time_stamps)
# Compute sMAPE
prophet_smape = ForecastMetric.sMAPE.value(test_data, prophet_forecast)
arima_smape = ForecastMetric.sMAPE.value(test_data, arima_forecast)
logger.info(f"Prophet sMAPE: {prophet_smape:.2f}")
logger.info(f"ARIMA sMAPE: {arima_smape:.2f}")


# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(test_data.to_pd(), label="Actual")
plt.plot(prophet_forecast.to_pd(), label="Prophet Forecast", linestyle="--")
plt.plot(arima_forecast.to_pd(), label="ARIMA Forecast", linestyle="--")
plt.legend()
plt.title("Prophet vs ARIMA Forecasting")
plt.show()


# Initialize an Isolation Forest model with the correct config
config = IsolationForestConfig()
anomaly_model = IsolationForest(config)

# Train the model on the dataset
anomaly_model.train(train_data)

# Generate anomaly scores
anomalies = anomaly_model.get_anomaly_label(test_data)
scores = anomaly_model.get_anomaly_score(test_data)

# Plot anomaly scores
plt.figure(figsize=(10, 6))
plt.plot(test_data.to_pd(), label="Original Data")
plt.plot(scores.to_pd(), label="Anomaly Scores", color="red", linestyle="--")
plt.legend()
plt.title("Anomaly Detection with Merlion")
plt.show()


# Instantiate models correctly
prophet_model = Prophet(ProphetConfig(
    add_seasonality="auto",
    weekly_seasonality=True,
    daily_seasonality=True,
    changepoint_prior_scale=0.05
))
arima_model = Arima(ArimaConfig(order=(2, 1, 2), target_seq_index=0))

# Train the models
prophet_model.train(train_data)
arima_model.train(train_data)

# Compare models on performance metrics
results = []
for model, name in zip([arima_model, prophet_model], ["Merlion ARIMA", "Prophet"]):
    forecast, _ = model.forecast(test_data.time_stamps)
    smape = ForecastMetric.sMAPE.value(test_data, forecast)
    results.append({"Model": name, "sMAPE": smape})

# Convert results to DataFrame
comparison_df = pd.DataFrame(results)
logger.info(comparison_df)

0       Merlion ARIMA   6.038623
1       Prophet         25.010984

# Import necessary libraries

# Load ERCOT dataset with hourly resampling and outlier removal
url = "https://raw.githubusercontent.com/kylejones200/time_series/main/ercot_load_data.csv"
df = read_csv(url)

# Convert time column to datetime format
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Resample to hourly frequency
df = df.resample('H').mean()
df['values'] = df['values'].interpolate()

# Remove outliers (values beyond 3 standard deviations)
df["z_score"] = (df["values"] - df["values"].mean()) / df["values"].std()
df = df[df["z_score"].abs() < 3]
df.drop(columns=["z_score"], inplace=True)

# Convert to Merlion TimeSeries format
ts = TimeSeries.from_pd(df)

# Optimize Data Splitting using TimeSeriesSplit (Cross-validation)
tscv = TimeSeriesSplit(n_splits=5)
train_idx, test_idx = list(tscv.split(df))[-1]  # Use the last split
train_data = TimeSeries.from_pd(df.iloc[train_idx])
test_data = TimeSeries.from_pd(df.iloc[test_idx])

# Function to create a model using ModelFactory

def get_model(model_type="prophet", transform=None):
    config_dict = {
        "prophet": ProphetConfig(
            add_seasonality="auto",
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.05,  # Better trend detection
            transform=TransformSequence([TemporalResample(), transform]) if transform else None
        ),
        "arima": ArimaConfig(order=(2, 1, 2), target_seq_index=0),
        "default": DefaultForecasterConfig()
    }

    model_mapping = {
        "prophet": "merlion.models.forecast.prophet:Prophet",
        "arima": "merlion.models.forecast.arima:Arima",
        "default": "merlion.models.defaults:DefaultForecaster"
    }

    if model_type not in model_mapping:
        raise ValueError(f"Invalid model type: {model_type}")

    return ModelFactory.create(model_mapping[model_type], **config_dict[model_type].to_dict())


# Function to evaluate and visualize forecasts
def eval_model(model, train_data, test_data, title):
    forecast_horizon = min(len(test_data), 168)  # Forecast up to 7 days (168 hours)
    t = test_data.time_stamps[:forecast_horizon]

    model.train(train_data)
    yhat_test, test_err = model.forecast(t)

    smape_value = ForecastMetric.sMAPE.value(test_data, yhat_test)

    # Confidence Intervals
    if hasattr(model, "forecast") and test_err is not None:
        ci_multiplier = 1.96  # 95% confidence
        lb = (yhat_test.to_pd() - ci_multiplier * test_err.to_pd().abs()).values.flatten()
        ub = (yhat_test.to_pd() + ci_multiplier * test_err.to_pd().abs()).values.flatten()

        # Ensure confidence intervals have the same length as timestamps
        min_length = min(len(t), len(lb), len(ub))
        t = t[:min_length]
        lb = lb[:min_length]
        ub = ub[:min_length]

    logger.info(f"{title} - sMAPE: {smape_value:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(test_data.to_pd(), label="Actual")
    plt.plot(yhat_test.to_pd(), label="Forecast", linestyle="--")

    if hasattr(model, "forecast") and test_err is not None:
        plt.fill_between(t, lb, ub, color="gray", alpha=0.3, label="Confidence Interval")

    plt.legend()
    plt.title(f"{title} - sMAPE: {smape_value:.2f}")
    plt.show()

    return yhat_test

# Run Prophet Model Without Transformations
logger.info("No transform...")
base = eval_model(get_model("prophet"), train_data, test_data, title="No Transform")

# Apply Normalization
logger.info("Normalize...")
norm = eval_model(get_model("prophet", MeanVarNormalize()), train_data, test_data, title="Mean-Variance Normalize")

# Apply Moving Average Transform
logger.info("Moving Average...")
ma = eval_model(get_model("prophet", MovingAverage(n_steps=12)), train_data, test_data, title="Moving Average Transform")

# Apply Seasonal Differencing
logger.info("Seasonal Differencing...")
diff = eval_model(get_model("prophet", DifferenceTransform()), train_data, test_data, title="Seasonal Differencing Transform")

# Run Merlion ARIMA Model
logger.info("\n=== ARIMA Model ===")
arima_results = eval_model(get_model("arima"), train_data, test_data, title="Merlion ARIMA")

# Run Default Forecaster (Baseline Model)
logger.info("\n=== Default Forecaster (Baseline Model) ===")
default_results = eval_model(get_model("default"), train_data, test_data, title="Default Forecaster")

# Create a table of sMAPE values
smape_values = {
    "Prophet (No Transform)": ForecastMetric.sMAPE.value(test_data, base),
    "Prophet (Mean-Variance Normalize)": ForecastMetric.sMAPE.value(test_data, norm),
    "Prophet (Moving Average Transform)": ForecastMetric.sMAPE.value(test_data, ma),
    "Prophet (Seasonal Differencing)": ForecastMetric.sMAPE.value(test_data, diff),
    "Merlion ARIMA": ForecastMetric.sMAPE.value(test_data, arima_results),
    "Default Forecaster": ForecastMetric.sMAPE.value(test_data, default_results),
}

# Convert to a DataFrame
smape_table = pd.DataFrame(list(smape_values.items()), columns=["Model", "sMAPE"]).sort_values(by="sMAPE")
logger.info(smape_table)

Model      sMAPE
5                  Default Forecaster   4.698401
4                       Merlion ARIMA   6.038623
1   Prophet (Mean-Variance Normalize)  20.132062
0              Prophet (No Transform)  25.010984
2  Prophet (Moving Average Transform)  29.058623
3     Prophet (Seasonal Differencing)  47.833924
