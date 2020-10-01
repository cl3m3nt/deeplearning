import tensorflow as tf 
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
import string

# Get Data
imdb = tfds.load('imdb_reviews',as_supervised=True)
train_dataset = imdb['train']
test_dataset = imdb['test']

# Get Example
sample = train_dataset.take(1)
for text,label in sample:
    raw_text = text
    raw_label = label
    print(raw_text)
    print(raw_label)

# Data Pre-Processing
# Cast TF tensor to String + Numpy
x_train, y_train, x_test, y_test = [],[],[],[]

for text,label in train_dataset:
    x_train.append(str(text.numpy()))
    y_train.append(label.numpy())

for text,label in test_dataset:
    x_test.append(str(text.numpy()))
    y_test.append(label.numpy())

# optionnaly remove < br> tags
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation), '')

# Cast y label to numpy
y_train=np.array(y_train)
y_test=np.array(y_test)

# Tokenize
vocab_size  = 10000
oov_token='<OOV>'
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size,oov_token=oov_token)
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

# Word Index
word_index = tokenizer.word_index # size 86539

# Pad Sequence
max_length=120
trunc_type = 'post'
x_train = pad_sequences(x_train,maxlen=max_length,truncating=trunc_type)
x_test = pad_sequences(x_test,maxlen=max_length,truncating=trunc_type)

# Modeling
embedding_dim = 16
def build_imdb_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8)),
        tf.keras.layers.Dense(units=10,activation='relu'),
        tf.keras.layers.Dense(units=1,activation='sigmoid')
    ])
    return model

# Remark: Embedding Layer is a look up table that maps integer to semantic vector
# https://www.tensorflow.org/tutorials/text/word_embeddings#using_the_embedding_layer


def compile_fit(model):
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['acc']
    )
    history = model.fit(x_train,y_train,
        validation_data = (x_test,y_test),
        epochs=5,
        batch_size=32,
    )
    return history

imdb_model = build_imdb_model()
history = compile_fit(imdb_model)


def before_after_embedding(model,input_data):
    embedding_layer = model.layers[0]
    output_data = embedding_layer(input_data)
    print(f'input data: {input_data}')
    print(f'embedding data: {output_data}')
    return output_data

    