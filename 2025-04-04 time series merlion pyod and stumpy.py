import anomsmith as am
import numpy as np
import pandas as pd
import plotsmith as ps


def main() -> None:
    url = 'https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine_temperature_system_failure.csv'

    df = pd.read_csv(url)

    df = df.rename(columns={df.columns[0]: 'value'})

    df['time'] = np.arange(len(df))

    anomaly_threshold = df['value'].quantile(0.98)

    df['anomaly'] = (df['value'] > anomaly_threshold).astype(int)

    models = ['isolation_forest', 'level_shift', 'stumpy', 'merlion', 'alibi']

    results = am.compare_models(df['value'].values, ground_truth=df['anomaly'].values, models=models, return_details=True)

    print('Anomaly Detection Model Comparison:')

    print(results.summary)

    ps.plot_anomaly_comparison(df, value_col='value', time_col='time', ground_truth=df['anomaly'], results=results, title='Anomaly Detection Model Comparison - NASA SMAP Dataset', figsize=(14, 8))

    ps.plot_anomaly_results(df, value_col='value', time_col='time', results=results, models=models, title='Individual Model Anomaly Detections')

    # Summary: anomsmith/plotsmith replace manual per-library setup with compare_models()
    # and plot_anomaly_comparison() for unified anomaly detection workflows.

if __name__ == "__main__":
    main()
