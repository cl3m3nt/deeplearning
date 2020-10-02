import tensorflow as tf 
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np
import os
import re
import string
import shutil
import io

# Get Data from TFDS
imdb = tfds.load('imdb_reviews',as_supervised=True)
train_data,test_data = imdb['train'],imdb['test']

# Visualize Data
def sample_data(data):
    sample = data.take(1)
    for text,label in sample:
        print(text)
        print(label)

# Data preprocessing with Cast, Token, Pad on tfds data
x_train,y_train,x_test,y_test = [],[],[],[]
for text, label in train_data:
    x_train.append(str(text.numpy()))
    y_train.append(label.numpy())
for text, label in test_data:
    x_test.append(str(text.numpy()))
    y_test.append(label.numpy())

y_train = np.array(y_train)
y_test = np.array(y_test)

vocab_size=10000
oov_token = '<OOV>'

def tokenize(data):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size,oov_token=oov_token)
    tokenizer.fit_on_texts(data)
    return tokenizer
tokenizer = tokenize(x_train)

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)


max_length=120
trunc_type='post'
x_train = pad_sequences(x_train,maxlen=max_length,truncating=trunc_type)
x_test = pad_sequences(x_test,maxlen=max_length,truncating=trunc_type)


# Data Preprocessing from directory
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url,
                                    untar=True, cache_dir='.',
                                    cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
os.listdir(dataset_dir)

train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

batch_size = 1024
seed = 123

train_ds = tf.keras.preprocessing.text_dataset_from_directory('aclImdb/train', batch_size=batch_size, validation_split=0.2, 
    subset='training', seed=seed)
val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', batch_size=batch_size, validation_split=0.2, 
    subset='validation', seed=seed)

def sample_batch(batch_data):
    batch_sample = batch_data.take(1)
    for text,label in batch_sample:
        for i in range(0,5):
            print(label[i].numpy(),text.numpy()[i])

sample_batch(train_ds)

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Create a custom standardization function to strip HTML break tags '<br />'.
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation), '')

vocab_size = 10000
sequence_length = 120

vectorize_layer = TextVectorization(
    standardize = custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length
)

x_train = train_ds.map(lambda x,y:x)
vectorize_layer.adapt(x_train)

# Build Model with vectorize layer
embedding_dim = 16
model = tf.keras.Sequential([
    vectorize_layer,
    tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=sequence_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(8,activation='relu'),
    tf.keras.layers.Dense(units=1,activation='sigmoid')
])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")


def compile_fit(model):
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics = ['acc']
    )

    history = model.fit(train_ds,
        validation_data = val_ds,
        epochs=5,
        callbacks=tensorboard_callback
    )
    
    return history

history = compile_fit(model)
model.summary()

# Get Vocabulary & Embeddings
vocab = vectorize_layer.get_vocabulary()
print(vocab[:10])
# Get weights matrix of layer named 'embedding'
weights = model.get_layer('embedding').get_weights()[0]
print(weights.shape) 

# Save Vocab & Embedding to Disk
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

for num, word in enumerate(vocab):
    if num == 0: continue # skip padding token from vocab
    vec = weights[num]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()

try:
    from google.colab import files
except ImportError:
    pass
else:
    files.download('vecs.tsv')
    files.download('meta.tsv')
