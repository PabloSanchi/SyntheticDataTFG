'''
    import pandas_profiling` 
    is going to be deprecated by April 1st.
    Please use `import ydata_profiling` instead.
'''

from sklearn.neighbors import KernelDensity
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
# from pandas_profiling import ProfileReport

filename = 'selectedCols.csv'
colums = [
    'wind_direction_true', 
    'wind_speed', 
    'wave_height', 
    'swell_direction', 
    'swell_height'
]

original_data = pd.read_csv(filename, usecols=colums)

# Use KDE to estimate the joint probability density function of the data
kde = KernelDensity(bandwidth=0.1)
kde.fit(original_data.values)

# Generate synthetic data by sampling from the KDE model
synthetic_data = kde.sample(1000)

# Save the synthetic data as a new CSV file
synthetic_data = pd.DataFrame(synthetic_data, columns=original_data.columns)
synthetic_data.to_csv('DATA/synthetic_data_kde.csv', index=False)

# create a html pandas-profile report
profile = ProfileReport(synthetic_data, title="Synthetic Data inspector KDE", explorative=True)
profile.to_file("REPORTS/synthetic_data_inspector_kde.html")