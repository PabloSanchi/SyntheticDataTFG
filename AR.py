import pandas as pd
from statsmodels.tsa.api import AutoReg
from ydata_profiling import ProfileReport

filename = 'selectedCols.csv'
columns = [
    'wind_direction_true', 
    'wind_speed', 
    'wave_height', 
    'swell_direction', 
    'swell_height'
]

original_data = pd.read_csv(filename, usecols=columns)

# Use AR model to simulate the time series process of each column of the data
lags = 10
synthetic_data = pd.DataFrame()
for col in original_data.columns:
    ar_model = AutoReg(original_data[col], lags=lags)
    ar_result = ar_model.fit()
    synthetic_data[col] = ar_result.predict(start=len(original_data[col]), end=len(original_data[col])+999)


# Save the synthetic data as a new CSV file
synthetic_data.to_csv('DATA/synthetic_data_ar.csv', index=False)

# create a html pandas-profile report
profile = ProfileReport(synthetic_data, title="Synthetic Data inspector AR", explorative=True)
profile.to_file("REPORTS/synthetic_data_inspector_ar.html")
