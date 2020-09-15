import tensorflow as tf 
import tensorflow.keras.datasets as datasets
from tensorflow.keras.preprocessing.sequence import pad_sequences
import datetime
import logging
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Get Data
reuters = datasets.reuters
(x_train,y_train),(x_test,y_test)=reuters.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Explore Data Sample
logger.info(f'Printing out first data:\n {x_train[0]} \nAnd associated label:\n{y_train[0]}')

# Build word_index
word_index = reuters.get_word_index()
logger.info(f'There are {len(word_index)} words within word_index')
def wi_info(wi:dict):
    wi_iterator = iter(wi)
    for i in range(0,5):
        word = next(wi_iterator)
        index = wi[word]
        print(word,index)


# Helper to reverse sequence encoding
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# Data Pre-Processing: Padding x_train and x_test
max_length = 120
trunc_type = 'post'
x_train = pad_sequences(x_train,max_length,truncating=trunc_type)
x_test = pad_sequences(x_test,max_length,truncating=trunc_type)

# Get info on Data
def get_info(text_data:np.ndarray):
    logger.info(f'Summary statistics on data sample: {stats.describe(text_data)}')
    for i in range(0,5):
        logger.info(f'{max(x_train[i])}')


# Define model
vocab_size = len(word_index)+1
embedding_dim = 16
max_length = 120

model = tf.keras.models.Sequential([
   tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
   tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8,return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8)),
    tf.keras.layers.Dense(units=46,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Logdir + Callback
log_dir = "./tensorboard/reutersLogs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train model
model.fit(x_train,y_train,
    validation_data=(x_test,y_test),
    callbacks=tensorboard_callback,
    epochs=5
    )
