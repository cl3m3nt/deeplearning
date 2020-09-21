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
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Hour'] = df['Date'].dt.hour
df['Minute'] = df['Date'].dt.minute
df['Second'] = df['Date'].dt.second

# Groupby operation example
def group_by():
    df_by_year = df.groupby(['Year']).mean ()
    df_by_month = df.groupby(['Month']).mean()
    df_by_day = df.groupby(['Day']).mean()
    df_by_year_month = df.groupby(['Year','Month']).mean ()
    df_by_year_month_day = df.groupby(['Year','Month','Day']).mean ()

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

# Remove unessary feature
df.pop('Date Time')
df.pop('Date')
df.pop('Year')
df.pop('Month')
df.pop('Day')
df.pop('Hour')
df.pop('Minute')
df.pop('Second')

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


###########  Time Series Dataset Pre-Processing ###########
# Using WindowGenerator Object that holds: train,val,test sets as DataFrame and Labels
# 24 x input_feature, 24 x offset (horizon?) step, label 1 step
w1 = WindowGenerator(input_width=24, label_width=1, shift=24,train_df=df_train,val_df=df_val,test_df=df_test,label_columns=['T (degC)'])
print(w1)
# 24 x input_feature, 1 x offset step (horizon?), label 1 step = horizon = 1
w2 = WindowGenerator(input_width=47,label_width=1,shift=1,train_df=df_train,val_df=df_val,test_df=df_test,label_columns=['T (degC)'])
print(w2)

# Work out with Dataframe, Numpy, Tensor: Cast df to tf
def df_to_tensorSlice_tf(dataframe:pd.DataFrame,label:str)->pd.DataFrame:
    data_df = dataframe.copy()
    y_data_df = data_df.pop(label)
    y_data_np = np.array(y_data_df)
    x_data_np = np.array(data_df)
    dataset_tf = tf.data.Dataset.from_tensor_slices((x_data_np,y_data_np))
    return dataset_tf

def tf_to_np(dataset:object)->np.array:
    data = dataset.take(1)
    for feature,label in data:
        feature_np = feature.numpy()
        label_np = label.numpy()
    return feature_np,label_np

# This function create Time Series batch of TF Data
def df_to_batch_TS_tf(dataframe:pd.DataFrame,batch_size)->object:
    data_batch_TS_tf = tf.keras.preprocessing.timeseries_dataset_from_array(dataframe,targets=None,sequence_length=48,batch_size=batch_size)
    return data_batch_TS_tf

# Check Time Series Batch dimension
def check_TS_dim(ts_tf_batch:object)->tuple:
    sample = ts_tf_batch.take(1)
    for data in sample:
        dim = data.numpy().shape
    return dim


# Split Data based on indices (not column name as working with TF)
def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)
  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])
  return inputs, labels

# Testing Windowing + Split process
final_data = df_to_batch_TS_tf(df,64) # create 48-length window batches of size 64
print(len(final_data))
final_sample = final_data.take(1) # take a single batch of 64 x 48-length windows of 23 x columns
for e in final_sample:
    data = e
sample_inputs, sample_labels = w2.split_window(data)
print(sample_inputs.shape)
print(type(sample_inputs))
print(sample_labels.shape)
print(type(sample_labels))


# Adding split window function to Window Generator object
WindowGenerator.split_window = split_window

# Stack three slices, the length of the total window:
example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                           np.array(train_df[100:100+w2.total_window_size]),
                           np.array(train_df[200:200+w2.total_window_size])])


example_inputs, example_labels = w2.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'labels shape: {example_labels.shape}')

def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset


# Create Final train_data dataset, using make_dataset helper function
# make_dataset creates, batch of windowed data, then split window into input and labels
train_data = w2.make_dataset(w2.train_df)
test = train_data.take(1)
for input,label in test:
    print(input)
    print(label)


@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example