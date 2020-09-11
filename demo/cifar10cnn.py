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

tf.keras.backend.clear_session()

model2 = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(32,32,3)),
    tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=10,activation='softmax')
])

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(32,32,3)),
    tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=10,activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['acc']
)

# Train model
history = model.fit(x_train,
                    y_train,
                    validation_data=(x_test,y_test),
                    epochs=10
)