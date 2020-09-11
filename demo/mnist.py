import tensorflow as tf 
import tensorflow.keras.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

(x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()

print(len(x_train))
print(x_train.shape)

print(len(y_train))
print(y_train.shape)

plt.imshow(x_train[0])
plt.show()


# Image Pre-Processing
x_train = x_train / 255.0
x_test = x_test / 255.0

# Model Definition
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(units=10,activation='relu'),
    tf.keras.layers.Dense(units=10,activation='softmax')
])

model.summary()

# Model Compile & Train
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['acc']
)

history = model.fit(x_train,
                    y_train,
                    validation_data=(x_test,y_test),
                    epochs=10)

# Prediction
predictions = []
for i in range(0,10):
    test = tf.expand_dims(x_test[i],axis=0)
    prediction = model.predict(test)
    predictions.append(prediction)
print(f'predictons on 5 x Test images')
for i in range(0,10):
    print(predictions[i])
    print(np.argmax(predictions[i][0]))