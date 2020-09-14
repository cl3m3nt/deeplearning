import tensorflow as tf 
import tensorflow.keras.datasets as datasets 
from tensorflow.keras.preprocessing.sequence import pad_sequences
import datetime

# Get Data
reuters = datasets.reuters
(x_train,y_train),(x_test,y_test)=reuters.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Explore Data Sample
print(f'Printing out first data:\n {x_train[0]} \nAnd associated label:\n{y_train[0]}')


# Word_index: associate a word to a number
# The Word index can be manually build from original text
word_index = reuters.get_word_index()
print(f'There are {len(word_index)} words in word_index ')
wi_iterator = iter(word_index)
for i in range(0,5):
    word = next(wi_iterator)
    index = word_index[word]
    print(word)
    print(index)

# Helper to reverse sequence encoding
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# Data Pre-Processing
max_length = 120
trunc_type = 'post'
x_train = pad_sequences(x_train,maxlen=max_length,truncating=trunc_type)
x_test = pad_sequences(x_test,maxlen=max_length,truncating=trunc_type)

# Define Model
vocab_size = 30982
embedding_dim = 16
max_length = 120

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=8,activation='relu'),
    tf.keras.layers.Dense(units=46,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
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


