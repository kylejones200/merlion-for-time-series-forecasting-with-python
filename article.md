# Merlion for Time Series Forecasting with Python Exploring a new approach/library

::::### Merlion for Time Series Forecasting with Python 

#### Exploring a new approach/library
Merlion is an open-source Python library designed for time series
forecasting and anomaly detection. Developed by Salesforce, it
simplifies the end-to-end workflow of time series analysis by
integrating data preprocessing, model training, evaluation, and
visualization into a single framework.

Merlion supports statistical models, machine learning approaches, and
deep learning models. Merlion requires time series data in Pandas
DataFrame format with timestamps.

### Let's build an example.
I'm using data from Ercot on energy demand in Texas.


Merlion provides several forecasting models, including ARIMA and
Prophet.


Normally you would want ARIMA to be auto tuned. I had trouble getting
Merlion to do that --- a task easily done with pmdarima.


Merlion has several features for measuring forecast accuracy. I'm usin
sMAPE (symetric mean absolute percentage error).


Merlion has built in visuzlation tools but I'm using matplotlob instead
so I can have more flexibility.


### Anomaly Detection with Merlion
Merlion supports **both supervised and unsupervised** anomaly detection
models.



The scores are basically zero because this is a well structured dataset.

### Model Comparison
Merlion simplifies benchmarking multiple models.



### Full implementation


Overall, the default forecaster is the best and captures the
fluctuations in the data well. ARIMA has a low sMAPE but it clearly
doesn't fit the data well.

::::Merlion simplifies time series forecasting and anomaly detection in
Python with built-in evaluation and visualization tools. For the ERCOT
dataset, the Default Forecaster and Merlion ARIMA deliver the best
results. Prophet performs well with proper tuning, especially when using
Mean-Variance Normalization. However, the Box-Cox Transform
significantly increases error.

I spent a long time fiddling with Merlion. I wanted it to be amazing but
I don't love it and don't plan to use it for more projects.
