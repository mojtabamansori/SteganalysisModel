import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, SpatialDropout2D, BatchNormalization, LeakyReLU
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from function_dataload import dataload
from tensorflow.keras import backend as K


# Data Loader Function
def data_loader(file_paths, labels, image_size=(256, 256), batch_size=32, shuffle=True):
    num_samples = len(file_paths)
    indices = np.arange(num_samples)

    if shuffle:
        np.random.shuffle(indices)

    while True:
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]

            batch_files = [file_paths[i] for i in batch_indices]
            batch_labels = [labels[i] for i in batch_indices]

            X_batch = np.array([img_to_array(load_img(file, color_mode='grayscale', target_size=image_size)) for file in batch_files])
            X_batch = X_batch / 255.0  # Normalize pixel values to the range [0, 1]

            y_batch = np.array(batch_labels)

            yield X_batch, y_batch

# Parameters
batch_size = 64
epochs = 100
spatial_dropout = 0.1
bn_momentum = 0.2
epsilon = 0.001
norm_momentum = 0.4
leakyrelu_slope = 0.1  # Corrected the slope to be positive
momentum_sgd = 0.95
learning_rate = 0.001
payload = 0.4

# Data Loading
file_list_cover, file_list_WOW, file_list_UNIWARD = dataload(payload)
labels_cover = [0] * len(file_list_cover)
labels_wow = [1] * len(file_list_WOW)
all_files = file_list_cover + file_list_WOW
all_labels = labels_cover + labels_wow
train_files, test_files, train_labels, test_labels = train_test_split(all_files, all_labels, test_size=0.2, random_state=42)

# Model Definition
model = Sequential()

import tensorflow as tf
from tensorflow.keras import layers, models

srm_weights = np.load('SRM_Kernels.npy')
biasSRM=np.ones(30)

T3 = 3;
def Tanh3(x):
    tanh3 = K.tanh(x)*T3
    return tanh3

def steganalysis_model(num_classes=2):
    input_layer = layers.Input(shape=(256, 256, 1))

    o1 = layers.Conv2D(30, (5, 5), weights=[srm_weights, biasSRM], strides=(1, 1), padding='same', trainable=False,
                      activation=Tanh3, use_bias=True)(input_layer)
    o2 = layers.BatchNormalization(momentum=0.2, epsilon=0.001)(o1)

    o3 = layers.DepthwiseConv2D(kernel_size=(1,1), strides=1, padding='same', depth_multiplier=1, use_bias=False)(o2)
    o3 = layers.LeakyReLU(alpha=-0.1)(o3)
    o4 = layers.BatchNormalization(momentum=0.2, epsilon=0.001)(o3)

    o5 = layers.DepthwiseConv2D(kernel_size=(1, 1), strides=1, padding='same', depth_multiplier=1, use_bias=False)(o4)
    o5 = layers.LeakyReLU(alpha=-0.1)(o5)

    o6 = layers.BatchNormalization(momentum=0.2, epsilon=0.001)(layers.concatenate([o5, o2]))

    x = layers.Conv2D(60, kernel_size=5, strides=1, padding='same')(o6)
    x = layers.LeakyReLU(alpha=-0.1)(x)
    x = layers.BatchNormalization(momentum=0.2, epsilon=0.001)(x)

    x = layers.AvgPool2D(pool_size=2, strides=2)(x)

    x = layers.Conv2D(60, kernel_size=5, strides=1, padding='same')(x)
    x = layers.LeakyReLU(alpha=-0.1)(x)
    x = layers.BatchNormalization(momentum=0.2, epsilon=0.001)(x)

    x = layers.AvgPool2D(pool_size=2, strides=2)(x)

    x = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depth_multiplier=2, use_bias=False)(x)
    x = layers.LeakyReLU(alpha=-0.1)(x)
    x = layers.BatchNormalization(momentum=0.2, epsilon=0.001)(x)

    x = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depth_multiplier=1, use_bias=False)(x)
    x = layers.LeakyReLU(alpha=-0.1)(x)
    x = layers.BatchNormalization(momentum=0.2, epsilon=0.001)(x)

    x = layers.Conv2D(30, kernel_size=5, strides=1, padding='same')(x)
    x = layers.LeakyReLU(alpha=-0.1)(x)
    x = layers.BatchNormalization(momentum=0.2, epsilon=0.001)(x)

    x = layers.Conv2D(30, kernel_size=5, strides=1, padding='same')(x)
    x = layers.LeakyReLU(alpha=-0.1)(x)
    x = layers.BatchNormalization(momentum=0.2, epsilon=0.001)(x)

    x = layers.Conv2D(30, kernel_size=6, strides=1, padding='same')(x)
    x = layers.LeakyReLU(alpha=-0.1)(x)
    x = layers.BatchNormalization(momentum=0.2, epsilon=0.001)(x)

    x = layers.AvgPool2D(pool_size=2, strides=2)(x)

    x = layers.Conv2D(30, kernel_size=5, strides=1, padding='same')(x)
    x = layers.LeakyReLU(alpha=-0.1)(x)
    x = layers.BatchNormalization(momentum=0.2, epsilon=0.001)(x)

    x = layers.Conv2D(30, kernel_size=5, strides=1, padding='same')(x)
    x = layers.LeakyReLU(alpha=-0.1)(x)
    x = layers.BatchNormalization(momentum=0.2, epsilon=0.001)(x)
    x = layers.Flatten()(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

model = steganalysis_model()
model.build((None, 256, 256, 1))  # Adjust input shape based on your data

sgd = SGD(learning_rate=learning_rate, momentum=momentum_sgd)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

# Training
train_loader = data_loader(train_files, train_labels, batch_size=batch_size)
test_loader = data_loader(test_files, test_labels, batch_size=batch_size, shuffle=False)

model.summary()
model.fit(train_loader, epochs=epochs, steps_per_epoch=len(train_files)//batch_size,
          validation_data=test_loader, validation_steps=len(test_files)//batch_size)

# Evaluation
evaluation = model.evaluate(test_loader, steps=len(test_files)//batch_size)
print(f"Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}")
