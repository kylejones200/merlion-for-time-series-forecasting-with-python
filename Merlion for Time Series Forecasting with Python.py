"""Generated from Jupyter notebook: Merlion for Time Series Forecasting with Python

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import inspect
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from merlion.evaluate.forecast import (
    ForecastEvaluator,
    ForecastEvaluatorConfig,
    ForecastMetric,
)
from merlion.models.automl.autosarima import AutoSarima, AutoSarimaConfig
from merlion.models.automl.seasonality_mixin import SeasonalityLayer
from merlion.models.defaults import DefaultForecaster, DefaultForecasterConfig
from merlion.models.forecast.arima import Arima
from merlion.models.forecast.prophet import Prophet, ProphetConfig
from merlion.models.forecast.sarima import Sarima
from merlion.ts_datasets.forecast import M4
from merlion.utils import TimeSeries
from merlion.utils.time_series import TimeSeries
from scipy.stats import norm
from ts_datasets.forecast import M4


def train_and_forecast(model, train_data, test_data):
    model.train(train_data)
    forecast = model.forecast(time_stamps=test_data.time_stamps)
    evaluator = ForecastEvaluator(model)
    eval_result = evaluator.evaluate(ground_truth=test_data, predict=forecast)
    return (forecast, eval_result)


def assuming_your_data_and_values_are_already_define() -> None:
    df = pd.DataFrame({"timestamp": data, "value": values})

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df.set_index("timestamp", inplace=True)

    ts = TimeSeries.from_pd(df)

    print(ts)


def initialize_an_arima_model() -> None:
    config = Arima.Config(order=(5, 1, 0))

    model = Arima(config)


def create_some_sample_data() -> None:
    dates = pd.date_range(start="2020-01-01", end="2022-12-31", freq="D")

    values = np.random.randn(len(dates)).cumsum()

    df = pd.DataFrame({"timestamp": dates, "value": values})

    df.set_index("timestamp", inplace=True)


def visualize_the_time_series_and_draw_a_dotted_line() -> None:
    logging.basicConfig(level=logging.DEBUG)

    time_series, metadata = M4("Hourly")[0]

    train_data = TimeSeries.from_pd(time_series[metadata.trainval])

    test_data = TimeSeries.from_pd(time_series[~metadata.trainval])

    fig = plt.figure(figsize=(10, 6))

    ax = fig.add_subplot(111)

    ax.plot(time_series)

    ax.axvline(metadata[metadata.trainval].index[-1], ls="--", lw="2", c="k")

    plt.show()

    print(
        f"{len(train_data)} points in train split, {len(test_data)} points in test split."
    )

    config1 = AutoSarimaConfig(
        max_forecast_steps=len(train_data),
        order=("auto", "auto", "auto"),
        seasonal_order=("auto", "auto", "auto", "auto"),
        approximation=True,
        maxiter=5,
    )

    model1 = SeasonalityLayer(model=AutoSarima(model=Sarima(config1)))

    train_pred, train_err = model1.train(
        train_data,
        train_config={"enforce_stationarity": True, "enforce_invertibility": True},
    )

    forecast1, stderr1 = model1.forecast(len(test_data))

    smape1 = ForecastMetric.sMAPE.value(ground_truth=test_data, predict=forecast1)

    print(f"Full AutoSarima with approximation sMAPE is {smape1:.4f}")


def train_the_model_on_the_dataset() -> None:
    model.train(ts)

    evaluator = ForecastEvaluator(model, ts)

    forecast, error = evaluator.get_predicted_time_series()

    print(forecast)


def create_some_sample_data_2() -> None:
    dates = pd.date_range(start="2020-01-01", end="2022-12-31", freq="D")

    values = np.random.randn(len(dates)).cumsum()

    df = pd.DataFrame({"timestamp": dates, "value": values})

    df.set_index("timestamp", inplace=True)

    ts = TimeSeries.from_pd(df)

    train_data = ts[: int(0.8 * len(ts))]

    test_data = ts[int(0.8 * len(ts)) :]

    prophet_config = ProphetConfig()

    model = Prophet(prophet_config)

    model.train(train_data)

    prediction = model.forecast(n_periods=len(test_data))

    evaluator = ForecastEvaluator(model, train_data, test_data)

    eval_result = evaluator.evaluate()

    print(eval_result)


def after_creating_and_training_the_model() -> None:
    print(inspect.signature(model.forecast))


def create_some_sample_data_3() -> None:
    dates = pd.date_range(start="2020-01-01", end="2022-12-31", freq="D")

    values = np.random.randn(len(dates)).cumsum()

    df = pd.DataFrame({"timestamp": dates, "value": values})

    df.set_index("timestamp", inplace=True)

    ts = TimeSeries.from_pd(df)

    train_data = ts[: int(0.8 * len(ts))]

    test_data = ts[int(0.8 * len(ts)) :]

    prophet_config = ProphetConfig()

    model = Prophet(prophet_config)

    model.train(train_data)

    last_timestamp = train_data.time_stamps[-1]

    future_timestamps = pd.date_range(
        start=last_timestamp, periods=len(test_data) + 1, freq="D"
    )[1:]

    prediction = model.forecast(future_timestamps)

    evaluator = ForecastEvaluator(model, train_data, test_data)

    eval_result = evaluator.evaluate()

    print(eval_result)


def plot_the_original_data_and_forecast() -> None:
    plt.figure(figsize=(10, 6))

    plt.plot(ts.to_pd(), label="Actual")

    plt.plot(forecast.to_pd(), label="Forecast", linestyle="--")

    plt.fill_between(
        forecast.time_stamps,
        forecast.values - error.values,
        forecast.values + error.values,
        color="gray",
        alpha=0.3,
        label="Confidence Interval",
    )

    plt.legend()

    plt.show()


def create_some_sample_data_4() -> None:
    dates = pd.date_range(start="2020-01-01", end="2022-12-31", freq="D")

    values = np.random.randn(len(dates)).cumsum()

    df = pd.DataFrame({"timestamp": dates, "value": values})

    df.set_index("timestamp", inplace=True)

    ts = TimeSeries.from_pd(df)

    train_data = ts[: int(0.8 * len(ts))]

    test_data = ts[int(0.8 * len(ts)) :]

    prophet_config = ProphetConfig()

    model = Prophet(prophet_config)

    model.train(train_data)

    last_timestamp = train_data.time_stamps[-1]

    future_timestamps = np.arange(
        last_timestamp + 86400, last_timestamp + 86400 * (len(test_data) + 1), 86400
    )

    forecast, error = model.forecast(time_stamps=future_timestamps, return_iqr=True)

    eval_config = ForecastEvaluatorConfig()

    evaluator = ForecastEvaluator(model, eval_config)

    eval_result = evaluator.evaluate(ground_truth=test_data, predict=forecast)

    print(eval_result)

    plt.figure(figsize=(12, 6))

    plt.plot(ts.to_pd(), label="Actual", color="blue")

    plt.plot(forecast.to_pd(), label="Forecast", linestyle="--", color="red")

    plt.fill_between(
        forecast.to_pd().index,
        (forecast - error).to_pd().iloc[:, 0],
        (forecast + error).to_pd().iloc[:, 0],
        color="gray",
        alpha=0.3,
        label="Confidence Interval",
    )

    plt.title("Time Series Forecast")

    plt.xlabel("Date")

    plt.ylabel("Value")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()


def create_some_sample_data_5() -> None:
    dates = pd.date_range(start="2020-01-01", end="2022-12-31", freq="D")

    values = np.random.randn(len(dates)).cumsum()

    df = pd.DataFrame({"timestamp": dates, "value": values})

    df.set_index("timestamp", inplace=True)

    ts = TimeSeries.from_pd(df)

    train_data = ts[: int(0.8 * len(ts))]

    test_data = ts[int(0.8 * len(ts)) :]

    prophet_config = ProphetConfig()

    model = Prophet(prophet_config)

    model.train(train_data)

    last_timestamp = train_data.time_stamps[-1]

    future_timestamps = np.arange(
        last_timestamp + 86400, last_timestamp + 86400 * (len(test_data) + 1), 86400
    )

    forecast = model.forecast(time_stamps=future_timestamps)

    eval_config = ForecastEvaluatorConfig()

    evaluator = ForecastEvaluator(model, eval_config)

    eval_result = evaluator.evaluate(ground_truth=test_data, predict=forecast)

    print(eval_result)

    plt.figure(figsize=(12, 6))

    plt.plot(ts.to_pd(), label="Actual", color="blue")

    plt.plot(forecast.to_pd(), label="Forecast", linestyle="--", color="red")

    confidence_interval = forecast.to_pd().std() * 1.96

    plt.fill_between(
        forecast.to_pd().index,
        forecast.to_pd().iloc[:, 0] - confidence_interval,
        forecast.to_pd().iloc[:, 0] + confidence_interval,
        color="gray",
        alpha=0.3,
        label="Confidence Interval",
    )

    plt.title("Time Series Forecast")

    plt.xlabel("Date")

    plt.ylabel("Value")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()


def create_some_sample_data_6() -> None:
    dates = pd.date_range(start="2020-01-01", end="2022-12-31", freq="D")

    values = np.random.randn(len(dates)).cumsum()

    df = pd.DataFrame({"timestamp": dates, "value": values})

    df.set_index("timestamp", inplace=True)

    ts = TimeSeries.from_pd(df)

    train_data = ts[: int(0.8 * len(ts))]

    test_data = ts[int(0.8 * len(ts)) :]

    prophet_config = ProphetConfig()

    model = Prophet(prophet_config)

    model.train(train_data)

    last_timestamp = train_data.time_stamps[-1]

    future_timestamps = np.arange(
        last_timestamp + 86400, last_timestamp + 86400 * (len(test_data) + 1), 86400
    )

    forecast_tuple = model.forecast(time_stamps=future_timestamps)

    forecast = forecast_tuple[0]

    eval_config = ForecastEvaluatorConfig()

    evaluator = ForecastEvaluator(model, eval_config)

    eval_result = evaluator.evaluate(ground_truth=test_data, predict=forecast)

    print(eval_result)

    plt.figure(figsize=(12, 6))

    plt.plot(ts.to_pd(), label="Actual", color="blue")

    plt.plot(forecast.to_pd(), label="Forecast", linestyle="--", color="red")

    confidence_interval = forecast.to_pd().std() * 1.96

    plt.fill_between(
        forecast.to_pd().index,
        forecast.to_pd().iloc[:, 0] - confidence_interval,
        forecast.to_pd().iloc[:, 0] + confidence_interval,
        color="gray",
        alpha=0.3,
        label="Confidence Interval",
    )

    plt.title("Time Series Forecast")

    plt.xlabel("Date")

    plt.ylabel("Value")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()


def create_sample_data() -> None:
    np.random.seed(42)

    dates = pd.date_range(start="2020-01-01", end="2022-12-31", freq="D")

    trend = np.linspace(0, 10, len(dates))

    seasonality = 5 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365)

    noise = np.random.normal(0, 1, len(dates))

    values = trend + seasonality + noise

    df = pd.DataFrame({"timestamp": dates, "value": values})

    df.set_index("timestamp", inplace=True)

    ts = TimeSeries.from_pd(df)

    train_data = ts[:-365]

    test_data = ts[-365:]

    arima_model = Arima(order=(1, 1, 1))

    arima_forecast, arima_eval = train_and_forecast(arima_model, train_data, test_data)

    prophet_model = Prophet()

    prophet_forecast, prophet_eval = train_and_forecast(
        prophet_model, train_data, test_data
    )

    plt.figure(figsize=(15, 10))

    plt.plot(train_data.to_pd(), label="Training Data", color="blue")

    plt.plot(test_data.to_pd(), label="Test Data", color="green")

    plt.plot(
        arima_forecast.to_pd(), label="ARIMA Forecast", color="red", linestyle="--"
    )

    plt.plot(
        prophet_forecast.to_pd(),
        label="Prophet Forecast",
        color="purple",
        linestyle="--",
    )

    plt.title("Time Series Forecasting: ARIMA vs Prophet")

    plt.xlabel("Date")

    plt.ylabel("Value")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()

    print("ARIMA Evaluation:")

    print(arima_eval)

    print("\nProphet Evaluation:")

    print(prophet_eval)


def data_loader_returns_pandas_dataframes_which_we_c() -> None:
    time_series, metadata = M4(subset="Hourly")[0]

    train_data = TimeSeries.from_pd(time_series[metadata.trainval])

    test_data = TimeSeries.from_pd(time_series[~metadata.trainval])

    "We can then initialize and train Merlion’s DefaultForecaster, which is an forecasting model that balances performance with efficiency. We also obtain its predictions on the test split.\n"

    model = DefaultForecaster(DefaultForecasterConfig())

    model.train(train_data=train_data)

    test_pred, test_err = model.forecast(time_stamps=test_data.time_stamps)

    "Next, we visualize the model’s predictions.\n"

    fig, ax = model.plot_forecast(time_series=test_data, plot_forecast_uncertainty=True)

    plt.show()

    "Finally, we quantitatively evaluate the model. sMAPE measures the error of the prediction on a scale of 0 to 100 (lower is better), while MSIS evaluates the quality of the 95% confidence band on a scale of 0 to 100 (lower is better).\n"

    smape = ForecastMetric.sMAPE.value(ground_truth=test_data, predict=test_pred)

    lb = TimeSeries.from_pd(
        test_pred.to_pd() + norm.ppf(0.025) * test_err.to_pd().values
    )

    ub = TimeSeries.from_pd(
        test_pred.to_pd() + norm.ppf(0.975) * test_err.to_pd().values
    )

    msis = ForecastMetric.MSIS.value(
        ground_truth=test_data, predict=test_pred, insample=train_data, lb=lb, ub=ub
    )

    print(f"sMAPE: {smape:.4f}, MSIS: {msis:.4f}")


def main() -> None:
    assuming_your_data_and_values_are_already_define()
    initialize_an_arima_model()
    create_some_sample_data()
    visualize_the_time_series_and_draw_a_dotted_line()
    train_the_model_on_the_dataset()
    create_some_sample_data_2()
    after_creating_and_training_the_model()
    create_some_sample_data_3()
    plot_the_original_data_and_forecast()
    create_some_sample_data_4()
    create_some_sample_data_5()
    create_some_sample_data_6()
    create_sample_data()
    data_loader_returns_pandas_dataframes_which_we_c()


if __name__ == "__main__":
    main()
