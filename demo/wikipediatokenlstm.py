import tensorflow as tf 
import tensorflow_datasets as tfds 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import datetime
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Get Data
wikipedia = tfds.load('wikipedia_toxicity_subtypes',as_supervised=True)
dataset_train, dataset_test = wikipedia['train'],wikipedia['test']

# Get Sample
def get_sample_value(dataset):
    sample = dataset.take(1)
    for text, label in sample:
        print(f'Text Data is:\n{text}')
        print(f'Label is:\n{label}')
    return text,label

# Data Preprocessing
# TF Dataset to list 
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

# Labels as numpy array
y_train = np.array(y_train)
y_test = np.array(y_test)

# Tokenizer
vocab_size = 10000
oov_tok = '<OOV>'
tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_tok)
tokenizer.fit_on_texts(x_train)
# Word_index (not really useful here)
word_index = tokenizer.word_index
print(f'Length of word_index: {len(word_index)}')
# Text to Sequence using tokenizer
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
# Padding Sequence to 120 length
max_length = 120
trunc_type = 'post'
x_train = pad_sequences(x_train,maxlen=max_length,truncating=trunc_type)
x_test = pad_sequences(x_test,maxlen=max_length,truncating=trunc_type)
logger.info(f'x_train shape: {x_train.shape}')
logger.info(f'x_test shape: {x_test.shape}')

# Define Model
embedding_dim = 16

model=tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8,return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8)),
    tf.keras.layers.Dense(units=1,activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Logdir + Callback
log_dir = "./tensorboard/wikipediaLogs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train model
model.fit(x_train,y_train,
    validation_data=(x_test,y_train),
    callbacks=tensorboard_callback,
    epochs=5,
    batch_size=32
    )
