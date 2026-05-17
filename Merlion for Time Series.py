"""Generated from Jupyter notebook: Merlion for Time Series

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import logging

import matplotlib.pyplot as plt
from merlion.utils.time_series import TimeSeries
from ts_datasets.forecast import M4


def main():
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
    merlion.models.automl.seasonality_mixin


def main() -> None:
    main()


if __name__ == "__main__":
    main()
