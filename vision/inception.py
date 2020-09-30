import tensorflow as tf 
from tensorflow.keras.applications import InceptionV3
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


def transfer_inceptionv3():
    inceptionv3 = InceptionV3(include_top=False,weights='imagenet',input_shape=(32,32,3))
    inceptionv3_preprocess = tf.keras.applications.inception_v3.preprocess_input

    inputs = tf.keras.Input(shape=(32,32,3))
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

    history = model.fit(x_train,y_train,
        validation_data = (x_test,y_test),
        epochs=5,
        batch_size = 32
        #callbacks=tensorboard_callback

    )
    return history


inceptionv3 = transfer_inceptionv3()
history = compile_fit(inceptionv3)