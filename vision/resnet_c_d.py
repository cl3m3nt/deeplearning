import tensorflow as tf 
from tensorflow.keras.applications import ResNet50,ResNet50V2,ResNet101,ResNet101V2,ResNet152,ResNet152V2
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow.keras.datasets as datasets
import matplotlib.pyplot as plt
import datetime
import os

# Get Data
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

train_dataset = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)


validation_dataset = image_dataset_from_directory(validation_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE)

# Plot Data
def plot_sample(sample,title):
    plt.imshow(sample)
    plt.title(f'Label:{title}, shape:{sample.shape}')
    plt.show()

def transfer_resnet50():
    resnet50 = ResNet50(include_top=False,weights='imagenet',input_shape=(160,160,3))
    resnet50_preprocess = tf.keras.applications.resnet50.preprocess_input

    inputs = tf.keras.Input(shape=(160,160,3))
    x = resnet50_preprocess(inputs)
    x = resnet50(inputs,training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(units=1,activation='sigmoid')(x)
    custom_resnet50 = tf.keras.Model(inputs,outputs)
    custom_resnet50.summary()

    return custom_resnet50


def transfer_resnet50v2():
    resnet50v2 = ResNet50V2(include_top=False,weights='imagenet',input_shape=(160,160,3))
    resnet50v2_preprocess = tf.keras.applications.resnet50.preprocess_input

    inputs = tf.keras.Input(shape=(160,160,3))
    x = resnet50v2_preprocess(inputs)
    x = resnet50v2(inputs,training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(units=10,activation='sigmoid')(x)
    custom_resnet50v2 = tf.keras.Model(inputs,outputs)
    custom_resnet50v2.summary()

    return custom_resnet50v2


def transfer_resnet101():
    resnet101 = ResNet101(include_top=False,weights='imagenet',input_shape=(160,160,3))
    resnet101_preprocess = tf.keras.applications.resnet50.preprocess_input

    inputs = tf.keras.Input(shape=(160,160,3))
    x = resnet101_preprocess(inputs)
    x = resnet101(inputs,training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(units=1,activation='sigmoid')(x)
    custom_resnet101 = tf.keras.Model(inputs,outputs)
    custom_resnet101.summary()

    return custom_resnet101


def transfer_resnet101v2():
    resnet101v2 = ResNet101V2(include_top=False,weights='imagenet',input_shape=(160,160,3))
    resnet101v2_preprocess = tf.keras.applications.resnet50.preprocess_input

    inputs = tf.keras.Input(shape=(160,160,3))
    x = resnet101v2_preprocess(inputs)
    x = resnet101v2(inputs,training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(units=1,activation='sigmoid')(x)
    custom_resnet101v2 = tf.keras.Model(inputs,outputs)
    custom_resnet101v2.summary()

    return custom_resnet101v2


def transfer_resnet152():
    resnet152 = ResNet152(include_top=False,weights='imagenet',input_shape=(160,160,3))
    resnet152_preprocess = tf.keras.applications.resnet50.preprocess_input


    inputs = tf.keras.Input(shape=(160,160,3))
    x = resnet152_preprocess(inputs)
    x = resnet152(inputs,training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(units=1,activation='sigmoid')(x)
    custom_resnet152 = tf.keras.Model(inputs,outputs)
    custom_resnet152.summary()

    return custom_resnet152


def transfer_resnet152v2():
    resnet152v2 = ResNet152V2(include_top=False,weights='imagenet',input_shape=(160,160,3))
    resnet152v2_preprocess = tf.keras.applications.resnet50.preprocess_input


    inputs = tf.keras.Input(shape=(160,160,3))
    x = resnet152v2_preprocess(inputs)
    x = resnet152v2(inputs,training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(units=1,activation='sigmoid')(x)
    custom_resnet152v2 = tf.keras.Model(inputs,outputs)
    custom_resnet152v2.summary()

    return custom_resnet152v2


def compile_fit(model):
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss = 'binary_crossentropy',
        metrics = ['acc']
    )

    log_dir = "./tensorboard/transfer/resnettLogs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(train_dataset,
        validation_data = validation_dataset,
        epochs=5,
        batch_size = 32
        #callbacks=tensorboard_callback

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