import tensorflow as tf 
from tensorflow.keras.applications import ResNet50,ResNet50V2,ResNet101,ResNet101V2,ResNet152,ResNet152V2
import tensorflow.keras.datasets as datasets
import matplotlib.pyplot as plt
import datetime

# Get Data
cifar10 = datasets.cifar10
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

# Plot Data
def plot_sample(sample,title):
    plt.imshow(sample)
    plt.title(f'Label:{title}, shape:{sample.shape}')
    plt.show()

# Preprocess Data
x_train = x_train/255.0
x_test = x_test/255.0

def transfer_resnet50():
    resnet50 = ResNet50(include_top=False,weights='imagenet',input_shape=(32,32,3))
    resnet50.summary()

    inputs = tf.keras.Input(shape=(32,32,3))
    x = resnet50(inputs,training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(units=10,activation='softmax')(x)
    custom_resnet50 = tf.keras.Model(inputs,outputs)

    return custom_resnet50


def transfer_resnet50v2():
    resnet50v2 = ResNet50V2(include_top=False,weights='imagenet',input_shape=(32,32,3))
    resnet50v2.summary()

    inputs = tf.keras.Input(shape=(32,32,3))
    x = resnet50v2(inputs,training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(units=10,activation='softmax')(x)
    custom_resnet50v2 = tf.keras.Model(inputs,outputs)

    return custom_resnet50v2


def transfer_resnet101():
    resnet101 = ResNet101(include_top=False,weights='imagenet',input_shape=(32,32,3))
    resnet101.summary()

    inputs = tf.keras.Input(shape=(32,32,3))
    x = resnet101(inputs,training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(units=10,activation='softmax')(x)
    custom_resnet101 = tf.keras.Model(inputs,outputs)

    return custom_resnet101


def transfer_resnet101v2():
    resnet101v2 = ResNet101V2(include_top=False,weights='imagenet',input_shape=(32,32,3))
    resnet101v2.summary()

    inputs = tf.keras.Input(shape=(32,32,3))
    x = resnet101v2(inputs,training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(units=10,activation='softmax')(x)
    custom_resnet101v2 = tf.keras.Model(inputs,outputs)

    return custom_resnet101v2


def transfer_resnet152():
    resnet152 = ResNet152(include_top=False,weights='imagenet',input_shape=(32,32,3))
    resnet152.summary()

    inputs = tf.keras.Input(shape=(32,32,3))
    x = resnet152(inputs,training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(units=10,activation='softmax')(x)
    custom_resnet152 = tf.keras.Model(inputs,outputs)

    return custom_resnet152


def transfer_resnet152v2():
    resnet152v2 = ResNet152V2(include_top=False,weights='imagenet',input_shape=(32,32,3))
    resnet152v2.summary()

    inputs = tf.keras.Input(shape=(32,32,3))
    x = resnet152v2(inputs,training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(units=10,activation='softmax')(x)
    custom_resnet152v2 = tf.keras.Model(inputs,outputs)

    return custom_resnet152v2


def compile_fit(model):
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['acc']
    )

    log_dir = "./tensorboard/transfer/resnettLogs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(x_train,y_train,
        validation_data = (x_test,y_test),
        epochs=5,
        batch_size = 64,
        callbacks=tensorboard_callback

    )
    return history

resnet50 = transfer_resnet50()
history = compile_fit(resnet50)

resnet50v2 = transfer_resnet50v2()
history = compile_fit(resnet50v2)

resnet101 = transfer_resnet101()
history = compile_fit(resnet101)

resnet101v2 = transfer_resnet101v2()
history = compile_fit(resnet101v2)

resnet152 = transfer_resnet152()
history = compile_fit(resnet152)

resnet152v2 = transfer_resnet152v2()
history = compile_fit(resnet152v2)