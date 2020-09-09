import tensorflow as tf 
import numpy as np

x = np.array([0,1,2,3,4,5,6,7,8,9,10],dtype = float)
y = np.array([0,2,4,6,8,10,12,14,16,18,20], dtype = float)

print(f'x values: {x}')
print(f'y values: {y}')

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1,input_shape=[1])
])

model.summary()

model.compile(
    optimizer = 'SGD',
    loss = 'mse',
    metrics = ['acc']
)

history = model.fit(x,y,epochs=50)

prediction = model.predict([11.0])
print(f'prediction by model: {prediction}')