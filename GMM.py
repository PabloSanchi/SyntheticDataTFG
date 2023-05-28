'''
Using generative models: 
    - Not neural network-based, 
    - Statistical modeling. 
    - Generative models:
        - Gaussian Mixture Models (GMM), 
        - Kernel Density Estimation (KDE),
        - AutoRegressive
    Using GMM
'''

'''
    import pandas_profiling` 
    is going to be deprecated by April 1st.
    Please use `import ydata_profiling` instead.
'''

from sklearn.mixture import GaussianMixture
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

# gmm => Gaussian Mixture Model
gmm = GaussianMixture(n_components=3)
gmm.fit(original_data)
synthetic_data = gmm.sample(n_samples=1000)[0]
# synthetic_data = gmm.sample(n_samples=1_000_000)[0]

# numpy array to pandas dataframe
synthetic_data = pd.DataFrame(synthetic_data, columns=colums)
print(synthetic_data.head(10))

# remove duplicates
synthetic_data = synthetic_data.drop_duplicates() 

# save csv
synthetic_data.to_csv('DATA/synthetic_data_gmm.csv')
# synthetic_data.to_csv('synthetic_data_gmm.csv')

# create a html pandas-profile report
profile = ProfileReport(synthetic_data, title="Synthetic Data inspector GMM", explorative=True)
profile.to_file("REPORTS/synthetic_data_inspector_gmm.html")
# profile.to_file("synthetic_data_inspector_GMM.html")
