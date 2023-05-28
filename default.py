import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport

def create_csv_file(df):
    df.to_csv(f'DATA/{output_file}', index=False)
    
def na_to_mean(df, colums):
    for col in colums:
        mean_value = int(df[col].mean())
        df[col].fillna(value=mean_value, inplace=True)
    return df

filename = 'original_data_full.csv'
output_file = 'selectedCols.csv'

'''
Column names:
wind_direction_true
wind_speed
wave_height
swell_direction
swell_height
timestamp
'''

colums = ['wind_direction_true', 'wind_speed', 'wave_height', 'swell_direction', 'swell_height']

# read data from csv file using the specified column names
df = pd.read_csv(f'DATA/{filename}', usecols=colums, decimal=',')
df = df.drop_duplicates()

# replace NaN values with the mean of the column
print(df.head())
df = na_to_mean(df, colums)
print(df.head())
# write data into another csv file
create_csv_file(df)

profile = ProfileReport(df, title='Data inspector', explorative=True)
#Displaying the report
# a = profile.to_widgets()
# write a in a html file
profile.to_file("REPORTS/original_data_inspector.html")