import tensorflow as tf 
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
import tensorflow.keras.datasets as datasets
import matplotlib.pyplot as plt

# https://keras.io/guides/transfer_learning/

# Get Data
cifar10 = datasets.cifar10
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

# Preprocess Data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Analyze Sample
def plot_sample(sample_img,sample_label):
    plt.imshow(sample_img)
    plt.title(f'Label value: {sample_label}')
    plt.show()


# Define VGGs architecture
def transfer_vgg19():
    # Load VGG19 architecture
    vgg19 = VGG19(input_shape=(32,32,3),weights='imagenet',include_top=False)
    vgg19.summary()
    for layer in vgg19.layers:
        layer.trainable = False

    # Define custom model
    inputs = tf.keras.Input(shape=(32,32,3))
    x = vgg19(inputs,training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10,activation='softmax')(x)
    custom_vgg19 = tf.keras.Model(inputs,outputs)
    return custom_vgg19


def transfer_vgg16():
    # Load VGG19 architecture
    vgg16 = VGG16(input_shape=(32,32,3),weights='imagenet',include_top=False)
    vgg16.summary()
    for layer in vgg19.layers:
        layer.trainable = False

    # Define custom model
    inputs = tf.keras.Input(shape=(32,32,3))
    x = vgg19(inputs,training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10,activation='softmax')(x)
    custom_vgg16 = tf.keras.Model(inputs,outputs)
    return custom_vgg16


# Compile & Train model
def compile_fit(model):
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics = ['acc']
    )

    model.fit(x_train,y_train,
        validation_data = (x_test,y_test),
        epochs=5,
        batch_size=64
    )
    return model

custom_vgg16 = transfer_vgg16()
history_vgg16 = compile_fit(custom_vgg16)

custom_vgg19 = transfer_vgg19()
history_vgg19 = compile_fit(custom_vgg19)


