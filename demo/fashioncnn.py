import tensorflow as tf 
import tensorflow.keras.datasets as datasets
import datetime


# Get Data
(x_train,y_train),(x_test,y_test) = datasets.fashion_mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Pre-Process Data
x_train = x_train/255.0
x_test = x_test/255.0

# Add 1 extra-dim for Convolution
x_train = tf.expand_dims(x_train,axis=3)
x_test = tf.expand_dims(x_test,axis=3)

# Define CNN Model
tf.keras.backend.clear_session()

model2 = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28,28,1)),
    tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=8,activation='relu'),
    tf.keras.layers.Dense(units=10,activation='softmax')
])

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28,28,1)),
    tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=8,activation='relu'),
    tf.keras.layers.Dense(units=10,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )

model.summary()

# Log & Callback
log_dir = "./tensorboard/fashionLogs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# Train Model
model = model.fit(x_train,y_train,
    validation_data=(x_test,y_test),
    callbacks=tensorboard_callback,
    epochs=10
    )