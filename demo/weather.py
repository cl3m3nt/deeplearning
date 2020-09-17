import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd 
import logging
import datetime
import seaborn as sns
from window import WindowGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


# Get Data
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)

csv_path, _ = os.path.splitext(zip_path)

# Create Dataframe
df_init = pd.read_csv(csv_path)
def df_info(dataframe:pd.DataFrame):
    print(f'There are {len(list(df_init.columns))} columns within Dataframe.')
    print(f'There are {len(df_init)} records within Dataframe')
    print(f'Columns name are:\n{list(df_init.columns)}')
    print(f'Descriptive Statistics are: {df_init.describe()}')
    print(f'Data samples:\n{df_init.head()}')

########### Pre-process Data ###########

df_init = pd.read_csv(csv_path)
df = df_init.copy()
# Sub-sampling data to keep only 1 record every hour
df = df[5::6] # this breaks natural index 0,1,2 etc...
def df_time_sample(dataframe:pd.DataFrame):
    for i in range(0,10):
        print(dataframe['Date Time'].iloc[i])

# Temp Dataframe + Datetime pre-processing
df['Date'] = pd.to_datetime(df['Date Time'],format='%d.%m.%Y %H:%M:%S')
df['Year'] =df['Date'].dt.year
df['Month'] =df['Date'].dt.month
df['Day'] =df['Date'].dt.day
df['Hour'] =df['Date'].dt.hour
df['Minute'] =df['Date'].dt.minute
df['Second'] =df['Date'].dt.second

# Groupby operation example
df_by_year =df.groupby(['Year']).mean ()
df_by_month =df.groupby(['Month']).mean()
df_by_day =df.groupby(['Day']).mean()
df_by_year_month =df.groupby(['Year','Month']).mean ()
df_by_year_month_day =df.groupby(['Year','Month','Day']).mean ()

# Visualize some feature: Temperature, Pressure, Rho
# Overall Time Series
def plot_all_TPR(dataframe:pd.DataFrame):
    plot_cols = ['T (degC)', 'p (mbar)', 'rho (g/m**3)']
    plot_features_df = dataframe[plot_cols] 
    plot_features_df.index = dataframe['Date'] #Date to become index
    _ = plot_features_df.plot(subplots=True)

def plot_480_TPR(dataframe:pd.DataFrame):
    plot_cols = ['T (degC)', 'p (mbar)', 'rho (g/m**3)']
    plot_features_df = dataframe[plot_cols][:480]
    plot_features_df.index = dataframe['Date'][:480]
    _ = plot_features_df.plot(subplots=True)

# Data Cleanup
def clean_wv(dataframe:pd.DataFrame):
    dataframe['wv (m/s)'].loc[(dataframe['wv (m/s)']==-9999.0)] = 0
    return dataframe

def clean_maxwv(dataframe:pd.DataFrame):
    dataframe['max. wv (m/s)'].loc[(dataframe['max. wv (m/s)']==-9999.0)] = 0
    return dataframe

clean_wv(df)
clean_maxwv(df)


# Feature Engineering: Wind Vector & Max Wind Vector = wv + wd 
df['wd (rad)'] = df['wd (deg)'] * np.pi/180
df['Wx'] =  df['wv (m/s)'] * np.cos(df['wd (rad)'])
df['Wy'] =  df['wv (m/s)'] * np.sin(df['wd (rad)'])
df['max Wx'] =  df['max. wv (m/s)'] * np.cos(df['wd (rad)'])
df['max Wy'] =  df['max. wv (m/s)'] * np.sin(df['wd (rad)'])
Wx = df['Wx']
Wy = df['Wy']
WmaxX = df['max Wx']
WmaxY = df['max Wy']

def plot_WxWy(wx,wy):
    plt.hist2d(wx,wy,bins=(50,50),vmax=400)
    plt.show()

def plot_WmaxX_WmaxY(w_maxx,w_maxy):
    plt.hist2d(w_maxx,w_maxy,bins=(50,50),vmax=400)
    plt.show()

# Feature Engineering: Time as a 2 x Components second Vector
# This is because Day & Year, like Wind angle, are meaningfully reprenseted as circular
fft = tf.signal.rfft(df['T (degC)'])
f_per_dataset = np.arange(0, len(fft))

n_samples_h = len(df['T (degC)'])
hours_per_year = 24*365.2524
years_per_dataset = n_samples_h/(hours_per_year)
f_per_year = f_per_dataset/years_per_dataset

def plot_frequency():
    plt.step(f_per_year, np.abs(fft))
    plt.xscale('log')
    plt.ylim(0, 400000)
    plt.xlim([0.1, max(plt.xlim())])
    plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])
    _ = plt.xlabel('Frequency (log scale)')

date_time = df['Date']
timestamp_s = date_time.map(datetime.datetime.timestamp)

day = 60*60*24 # number of seconds per day
year = day * 365.2425 # number of seconds per year
logger.info(f'There are {day} seconds within a day')
logger.info(f'There are {year} seconds within a year')

df['x Day'] = np.cos(timestamp_s * (2 * np.pi /day))
df['y Day'] = np.sin(timestamp_s * (2 * np.pi /day))
df['x Year'] = np.cos(timestamp_s * (2 * np.pi /year))
df['y Year'] = np.sin(timestamp_s * (2 * np.pi /year))

# Split Data: Train, Validation, Test
columns_indices = {name: i for i, name in enumerate(df.columns)}
n = len(df)

df.pop('Date Time')
df.pop('Date')
df_train = df[0:int(n*0.7)]
df_val = df[int(n*0.7):int(n*0.9)]
df_test = df[int(n*0.9):]

# Normalize Data
train_mean = df_train.mean()
train_std = df_train.std()

df_train = (df_train - train_mean)/train_std
df_val = (df_val - train_mean)/train_std
df_test = (df_test - train_mean)/train_std

def plot_normalized():
    df_std = (df - train_mean) / train_std
    df_std = df_std.melt(var_name='Column', value_name='Normalized')
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    _ = ax.set_xticklabels(df.keys(), rotation=90)
########### End Pre-process Data ###########


###########  Time Series Dataset ###########
w1 = WindowGenerator(input_width=24, label_width=1, shift=24,train_df=df_train,val_df=df_val,test_df=df_test,label_columns=['T (degC)'])
print(w1)