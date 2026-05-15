"""Generated from Jupyter notebook: Additional Analysis: Model Performance Summary

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

# Install required libraries
# Note: anomsmith provides a unified API for anomaly detection across multiple libraries
# ! pip install anomsmith plotsmith


# --- code cell ---

import numpy as np
import pandas as pd
import anomsmith as am
import plotsmith as ps

# anomsmith provides a unified API for anomaly detection across multiple libraries
# plotsmith provides enhanced visualization capabilities

# Load NASA SMAP dataset
url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine_temperature_system_failure.csv"
df = pd.read_csv(url)

# anomsmith can automatically load and preprocess data from URLs
# df = am.load_dataset(url)  # Alternative: automatic loading

# Prepare data - anomsmith handles this automatically in unified API
df = df.rename(columns={df.columns[0]: "value"})
df["time"] = np.arange(len(df))

# Create anomaly labels (assuming threshold-based ground truth)
anomaly_threshold = df["value"].quantile(0.98)
df["anomaly"] = (df["value"] > anomaly_threshold).astype(int)

# anomsmith provides a unified API to compare multiple anomaly detection methods
# No need to manually implement each library's interface!
models = ['isolation_forest', 'level_shift', 'stumpy', 'merlion', 'alibi']
results = am.compare_models(
    df["value"].values,
    ground_truth=df["anomaly"].values,
    models=models,
    return_details=True
)

# Display results - anomsmith provides comprehensive metrics automatically
print("Anomaly Detection Model Comparison:")
print(results.summary)

# plotsmith makes it easy to visualize anomaly detection results
# Automatically creates publication-quality plots comparing all models
ps.plot_anomaly_comparison(
    df,
    value_col="value",
    time_col="time",
    ground_truth=df["anomaly"],
    results=results,
    title="Anomaly Detection Model Comparison - NASA SMAP Dataset",
    figsize=(14, 8)
)

# plotsmith can also create detailed comparison plots for each model
ps.plot_anomaly_results(
    df,
    value_col="value",
    time_col="time",
    results=results,
    models=models,
    title="Individual Model Anomaly Detections"
)


# --- code cell ---

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
