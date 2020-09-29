import tensorflow as tf 
import tensorflow.keras.datasets as datasets
from tensorflow.keras.applications import MobileNet,MobileNetV2
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


def transfer_mobilenet():
    mobilenet=MobileNet(include_top=False,input_shape=(32,32,3),weights='imagenet')
    mobilenet.summary()

    inputs = tf.keras.Input(shape=(32,32,3))
    x = mobilenet(inputs,training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10,activation='softmax')(x)
    custom_mobilenet = tf.keras.Model(inputs,outputs)
    return custom_mobilenet


def transfer_mobilenetv2():
    mobilenetv2 = MobileNetV2(include_top=False, input_shape=(32,32,3),weights='imagenet')
    mobilenetv2.summary()

    inputs = tf.keras.Input(shape=(32,32,3))
    x = mobilenetv2(inputs,training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10,activation='softmax')(x)
    custom_mobilenetv2 = tf.keras.Model(inputs,outputs)
    return custom_mobilenetv2


def compile_fit(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss='sparse_categorical_crossentropy',
        metrics=['acc']
    )


    # Tensorboard logging + Callbacks
    log_dir = "./tensorboard/transfer/mobilenetLogs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(x_train,y_train,
        validation_data = (x_test,y_test),
        epochs=5,
        callbacks=tensorboard_callback
    )

    return history


def predict(model,input):
    prediction = model.predict(input)
    return prediction


mobilenet_model = transfer_mobilenet()
history = compile_fit(mobilenet_model)

mobilenetv2_model = transfer_mobilenetv2()
history = compile_fit(mobilenetv2_model)