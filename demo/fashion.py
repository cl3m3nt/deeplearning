import tensorflow as tf 
import tensorflow.keras.datasets as datatasets
import datetime

# Get Data
(x_train,y_train),(x_test,y_test) = datatasets.fashion_mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Pre-Process Data
x_train = x_train/255.0
x_test = x_test/255.0

# Define Model
tf.keras.backend.clear_session()

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28,28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=8,activation='relu'),
    tf.keras.layers.Dense(units=10,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

model.summary()

# Log & Callback
log_dir = "./tensorboard/fashionLogs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train model
model.fit(x_train,
    y_train,
    validation_data=(x_test,y_test),
    callbacks=tensorboard_callback,
    epochs=10
)



