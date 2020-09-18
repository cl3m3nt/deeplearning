import tensorflow as tf 
import tensorflow.keras.datasets as datasets
from tensorflow.keras.preprocessing.sequence import pad_sequences
import datetime
import numpy as np
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

#Get Data
imdb = datasets.imdb
(x_train,y_train),(x_test,y_test) = imdb.load_data() # loads imdb as numpy
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Explore Data Sample
logger.info(f'Printing out first data:\n {x_train[0]} \nAnd associated label:\n{y_train[0]}')

# Word_index
word_index = imdb.get_word_index()
logger.info(f'There are {len(word_index)} words within imdb word_index')
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

# Pre-Process Data: Padding x_train, x_test to same length
max_length = 120
trunc_type = 'post'
x_train = pad_sequences(x_train,maxlen=max_length,truncating=trunc_type)
x_test = pad_sequences(x_test,maxlen=max_length,truncating=trunc_type)
print(x_train.shape)
print(x_test.shape)

# Get info on Data
def getInfo(text_data:np.ndarray):
    logger.info(f'Summary statistics on data sample {stats.describe(text_data)}')
    for i in range(0,5):
          logger.info(f'Max index value {max(x_train[i])}')  

# Define LSTM Model
vocab_size = len(word_index)+1
embedding_dim = 16
max_length = 120

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8,return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8)),
    tf.keras.layers.Dense(units=1,activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics = ['accuracy']
)

model.summary()


# Logdir + Callback
log_dir = "./tensorboard/imdbLogs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train Model
model.fit(x_train,y_train,
    validation_data=(x_test,y_test),
    callbacks=tensorboard_callback,
    epochs=5
    )        

    