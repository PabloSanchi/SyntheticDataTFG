from ctgan import CTGAN, load_demo
import pandas as pd

filename = 'selectedCols.csv'
colums = [
    'wind_direction_true', 
    'wind_speed', 
    'wave_height', 
    'swell_direction', 
    'swell_height'
]

original_data = pd.read_csv(filename, usecols=colums)

# epochs lower than 1000 will not generate good results
# 3000 gives decent results
# 5000 gives good results
ctgan = CTGAN(epochs=5000)

ctgan.fit(original_data, colums)

synthetic_data = ctgan.sample(1000)
synthetic_data.to_csv('DATA/synthetic_data_gan.csv')