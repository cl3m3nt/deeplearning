import tensorflow as tf 
import numpy as np 
import os
import time

# Get Data
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file,'rb').read().decode(encoding='utf-8')

# Explore Data
def data_sample():
    print(text[:250])  
    vocab = sorted(set(text))
    print(f'Unique characters: {len(vocab)}')
    print(f'Total characters: {len(text)}')

# Data Pre-Processing
vocab = sorted(set(text))
char_2_idx = {u:i for i,u in enumerate(vocab)}
idx_2_char = np.array(vocab)
# Convert all text characters to in
text_as_int = np.array([char_2_idx[c] for c in text])

def show_char_2_idx():
    print('{')
    for char,_ in zip(char_2_idx, range(20)):
        print('  {:4s}: {:3d},'.format(repr(char), char_2_idx[char]))
    print('  ...\n}')

def show_mapping(text,boundary):
    # Show how the first 13 characters from the text are mapped to integers
    print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:boundary]), text_as_int[:boundary]))


# Create Dataset
seq_length = 100
examples_per_epoch = len(text) // (seq_length+1)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

def head_tf_character(number=5):
    for i in char_dataset.take(number):
        print(idx_2_char[i.numpy()])

sequences = char_dataset.batch(seq_length+1,drop_remainder=True)

def head_tf_sequences(number=5):
    for item in sequences.take(number):
        print(repr(''.join(idx_2_char[item.numpy()])))

def split_input_target(chunk):
  input_text = chunk[:-1]
  target_text = chunk[1:]
  return input_text, target_text

dataset = sequences.map(split_input_target)

def head_dataset(number=5):
    for input,target in dataset.take(number):
        print('Input Data',repr(''.join(idx_2_char[input.numpy()])))
        print('Target Data',repr(''.join(idx_2_char[target.numpy()])))


def head_prediction(number=5):
    for input,target in dataset.take(1):
        for i,(input_idx,target_idx) in enumerate(zip(input[:5],target[:5])):
            print("Step {:4d}".format(i))
            print("  input: {} ({:s})".format(input_idx, repr(idx_2_char[input_idx])))
            print("  expected output: {} ({:s})".format(target_idx, repr(idx_2_char[target_idx])))

# Modeling
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE,drop_remainder=True)

vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 128

def build_model(vocab_size,embedding_dim,rnn_units,batch_size):
    rnn_model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size,embedding_dim,batch_input_shape=[batch_size,None]),
        tf.keras.layers.GRU(rnn_units,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return rnn_model


def compile_fit(model,EPOCHS):
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['acc']
    )
    
    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    history = model.fit(dataset,
        epochs=EPOCHS,
        callbacks=checkpoint_callback
    )

    return history


rnn_model = build_model(vocab_size,embedding_dim,rnn_units,BATCH_SIZE)

history = compile_fit(rnn_model,10)


# Generate Text
checkpoint_dir = './training_checkpoints'
model = build_model(vocab_size,embedding_dim,rnn_units,batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1,None]))
model.summary()

def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char_2_idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
    predictions = model(input_eval)
    # remove the batch dimension
    predictions = tf.squeeze(predictions, 0)

    # using a categorical distribution to predict the character returned by the model
    predictions = predictions / temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

    # We pass the predicted character as the next input to the model
    # along with the previous hidden state
    input_eval = tf.expand_dims([predicted_id], 0)

    text_generated.append(idx_2_char[predicted_id])

  return (start_string + ''.join(text_generated))


print(generate_text(model, start_string=u"ROMEO: "))