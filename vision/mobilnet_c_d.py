import tensorflow as tf 
import tensorflow.keras.datasets as datasets
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNet,MobileNetV2
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


def transfer_mobilenet():
    mobilenet=MobileNet(include_top=False,input_shape=(160,160,3),weights='imagenet')
    mobilenet_preprocess = tf.keras.applications.mobilenet.preprocess_input

    inputs = tf.keras.Input(shape=(160,160,3))
    x = mobilenet_preprocess(inputs)
    x = mobilenet(inputs,training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(1,activation='sigmoid')(x)
    custom_mobilenet = tf.keras.Model(inputs,outputs)
    custom_mobilenet.summary()
    return custom_mobilenet


def transfer_mobilenetv2():
    mobilenetv2 = MobileNetV2(include_top=False, input_shape=(160,160,3),weights='imagenet')
    mobilenetv2_preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    inputs = tf.keras.Input(shape=(160,160,3))
    x = mobilenetv2_preprocess(inputs)
    x = mobilenetv2(inputs,training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(1,activation='sigmoid')(x)
    custom_mobilenetv2 = tf.keras.Model(inputs,outputs)
    custom_mobilenetv2.summary()
    return custom_mobilenetv2


def compile_fit(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['acc']
    )


    # Tensorboard logging + Callbacks
    log_dir = "./tensorboard/transfer/mobilenetLogs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(train_dataset,
        validation_data = validation_dataset,
        epochs=5,
        batch_size=32
        #callbacks=tensorboard_callback
    )

    return history


def predict(model,input):
    prediction = model.predict(input)
    return prediction


mobilenet_model = transfer_mobilenet()
history = compile_fit(mobilenet_model)

mobilenetv2_model = transfer_mobilenetv2()
history = compile_fit(mobilenetv2_model)