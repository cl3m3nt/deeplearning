import tensorflow as tf 
import tensorflow.keras.datasets as datasets
import matplotlib.pyplot as plt

# Get Data
fashion = datasets.fashion_mnist
(x_train,y_train),(x_test,y_test) = fashion.load_data()

# Preprocess Data
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = tf.expand_dims(x_train,axis=3)
x_test = tf.expand_dims(x_test,axis=3)


# Analyze Sample
def plot_sample(sample_img,sample_label):
    plt.imshow(sample_img)
    plt.title(f'Label value: {sample_label}')
    plt.show()


def conv_net_1():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),input_shape=(28,28,1)),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=10,activation='softmax')
    ])
    return model


def conv_net_2():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),input_shape=(28,28,1)),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3)),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=10,activation='softmax')
    ])
    return model

conv_model = conv_net_1()
conv_model.summary()

conv_model_2 = conv_net_2()
conv_model_2.summary()

def compile_fit(model):
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['acc']
    )

    history = model.fit(x_train,y_train,
    validation_data=(x_test,y_test),
    epochs=5,
    )
    
    return history

history = compile_fit(conv_model)
history_2 = compile_fit(conv_model_2)