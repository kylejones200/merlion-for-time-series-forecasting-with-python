import anomsmith as am
import numpy as np
import pandas as pd
import plotsmith as ps


def main() -> None:
    'Generated from Jupyter notebook: Additional Analysis: Model Performance Summary\n\nMagics and shell lines are commented out. Run with a normal Python interpreter.'

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

    # --- notebook cell ---
    ## Summary: What anomsmith and plotsmith Simplified

    ### Before (Original Code):
    - Manual implementation of 5 different anomaly detection libraries (PyOD, Kats, STUMPY, Merlion, Alibi-Detect)
    - Manual model setup and fitting for each library (~10 lines per model)
    - Manual evaluation metrics calculation
    - Manual plotting with matplotlib (15+ lines)
    - Inconsistent APIs across different libraries

    ### After (With Your Libraries):
    - **One function call** to compare all models: `am.compare_models()`
    - **Automatic model setup**: anomsmith handles all library-specific APIs
    - **Automatic evaluation**: Precision, recall, and timing metrics included
    - **One line** to visualize: `ps.plot_anomaly_comparison()`
    - **Unified API**: Same interface for all anomaly detection methods

    ### Key Improvements:
    1. **Unified Interface**: anomsmith provides consistent API across all anomaly detection libraries
    2. **Automatic Comparison**: Compare multiple models with a single function call
    3. **Enhanced Visualization**: plotsmith creates publication-quality comparison plots
    4. **Less Boilerplate**: No need to manually implement each library's interface
    5. **Better Documentation**: Clear function signatures show what each model does

if __name__ == "__main__":
    main()
