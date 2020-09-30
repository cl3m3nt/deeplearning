import tensorflow as tf 
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow.keras.datasets as datasets
import matplotlib.pyplot as plt 
import datetime

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


def transfer_inceptionv3():
    inceptionv3 = InceptionV3(include_top=False,weights='imagenet',input_shape=(160,160,3))
    inceptionv3_preprocess = tf.keras.applications.inception_v3.preprocess_input

    inputs = tf.keras.Input(shape=(160,160,3))
    x = inceptionv3_preprocess(inputs)
    x = inceptionv3(inputs,training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(units=10,activation='softmax')(x)
    custom_inceptionv3 = tf.keras.Model(inputs,outputs)
    custom_inceptionv3.summary()
    return custom_inceptionv3


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


inceptionv3 = transfer_inceptionv3()
history = compile_fit(inceptionv3)