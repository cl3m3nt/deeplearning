import tensorflow as tf 
import numpy as np

x = np.array([0,1,2,3,4,5,6,7,8,8,10],dtype=float)
y = np.array([0,2,4,6,8,10,12,14,16,18,20],dtype=float)

print(x)
print(y)


# Linear Activation Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(input_shape=[1],units=10,activation='linear'),
    tf.keras.layers.Dense(units = 1,activation='linear')
])

'''
# Sigmoid Activation Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(input_shape=[1],units=10,activation='sigmoid'),
    tf.keras.layers.Dense(units = 1,activation='linear')
])

# RELU Activation Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=10,activation='relu',input_shape=[1]),
    tf.keras.layers.Dense(units=1,activation='linear')
])
'''

model.summary()

model.compile(
    optimizer = 'adam',
    loss = 'mse',
    metrics = ['acc']
)

history = model.fit(x,y,epochs=50)

prediction = model.predict([11.0])
print(f'prediction by model: {prediction}')

