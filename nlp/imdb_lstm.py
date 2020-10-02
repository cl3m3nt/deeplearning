import tensorflow as tf 
import tensorflow_datasets as tfds 

# Get Data
imdb,info = tfds.load('imdb_reviews/subwords8k',as_supervised=True,with_info=True)
train_dataset,test_dataset = imdb['train'],imdb['test']

# Encode and Decode string
encoder = info.features['text'].encoder
print(f'Vocabulary size: {encoder.vocab_size}')

sample_string = 'Hello Tensorflow'
encoded_string = encoder.encode(sample_string)
original_string = encoder.decode(encoded_string)
print(f'Encoded String: {encoded_string}')
print(f'Orginal String: {original_string}')
for index in encoded_string:
  print('{} ----> {}'.format(index, encoder.decode([index])))

# Create Dataset
BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE)

test_dataset = test_dataset.padded_batch(BATCH_SIZE)

# Model 
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size,16),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8)),
    tf.keras.layers.Dense(8,activation='relu'),
    tf.keras.layers.Dense(1)
])

def compile_fit(model):
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['acc']
    )

    history = model.fit(train_dataset,
        validation_data = test_dataset,
        epochs=5
    )


history = compile_fit(model)

test_loss, test_acc = model.evaluate(test_dataset)
print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

