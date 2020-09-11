import tensorflow as tf 
import tensorflow.keras.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

# Get Data
(x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Visualize Data
plt.imshow(x_train[0])
plt.title(f'label value:{y_train[0]}')
plt.show()

# Preprocess Data
x_train = x_train/255.0
x_train = tf.expand_dims(x_train,axis=3)
x_test = x_test/255.0
x_test = tf.expand_dims(x_test,axis=3)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# Define Model
model2 = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28,28,1)),
    tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=10,activation='softmax')
])

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28,28,1)),
    tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=10,activation='softmax')
])

model.summary()

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['acc']
)

# Train model
history = model.fit(x_train,
                    y_train,
                    batch_size=32,
                    validation_data=(x_test,y_test),
                    epochs=10
)


