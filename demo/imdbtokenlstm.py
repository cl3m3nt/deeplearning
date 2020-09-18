import tensorflow as tf 
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import datetime
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Get Data from TFDS as Tensorflow dataset
imdb = tfds.load('imdb_reviews',as_supervised=True) # loads imdb as tensor Dataset
dataset_train , dataset_test = imdb['train'],imdb['test']

# Print out sample values
def get_sample_value(dataset):
    sample = dataset.take(1)
    for text,label in sample:
        print(f'Text Data is:\n{text}')
        print(f'Label is:\n{label}')
    return

# Data Pre-processing: convert TF dataset to List & Numpy arrays
x_train = []
y_train = []
x_test = []
y_test = []
for text,label in dataset_train:
    x_train.append(str(text.numpy()))
    y_train.append(label.numpy())

for text,label in dataset_test:
    x_test.append(str(text.numpy()))
    y_test.append(label.numpy())

# Pre-process Label list to numpy array
y_train = np.array(y_train) 
y_test = np.array(y_test)

# Pre-Process Text String list to numpy array: Tokenize, Word_index, Padding
vocab_size = 10000
oov_tok = '<OOV>'
# Tokenizing to token from 1 to 10000
tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_tok)
tokenizer.fit_on_texts(x_train)
# Word_Index of length 86539
word_index = tokenizer.word_index
logger.info(f'Word_index length is: {len(word_index)}')
# Convert Text to Numeric Sequence
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
# Padding to max length 120
max_length = 120
trunc_type = 'post'
x_train = pad_sequences(x_train,maxlen=max_length,truncating=trunc_type)
x_test = pad_sequences(x_test,maxlen=max_length,truncating=trunc_type)

# Define model
embedding_dim = 16

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8,return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8)),
    tf.keras.layers.Dense(units=1,activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['acc']
)

model.summary()

# Logdir + Callback
log_dir = "./tensorboard/imdbLogs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train Model
model.fit(x_train,y_train,
    validation_data=(x_test,y_test),
    callbacks=tensorboard_callback,
    epochs=5,
    batch_size=256
    )
