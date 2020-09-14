import tensorflow as tf 
import tensorflow.keras.datasets as datasets
import matplotlib.pyplot as plt 
import numpy as np 

# Get Data
(x_train,y_train),(x_test,y_test) = datasets.cifar10.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Pre-Process Data
x_train = x_train/255.0
x_test = x_test/255.0

# Define Model
model2 = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(32,32,3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=8, activation='relu'),
    tf.keras.layers.Dense(units=10,activation='softmax')
])

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(32,32,3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=10,activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['acc']
)

# Tensorboard logging
log_dir = "./tensorboard/cifarLogs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train Model
model.fit(x=x_train, 
          y=y_train, 
          epochs=10, 
          validation_data=(x_test, y_test), 
          callbacks=[tensorboard_callback])