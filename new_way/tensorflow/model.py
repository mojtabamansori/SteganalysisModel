import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, SpatialDropout2D, BatchNormalization, LeakyReLU
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from function_dataload import dataload
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image = img_to_array(image)
    image = image / 255.0
    return image
def create_tf_dataset(file_lists, labels, batch_size):
    all_files = []
    for i, file_list in enumerate(file_lists):
        all_files.extend([(file_path, i) for file_path in file_list])

    paths, labels = zip(*all_files)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(lambda x, y: (load_and_preprocess_image(x), y))
    dataset = dataset.shuffle(buffer_size=len(paths))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

payload = 0.4
file_list_cover, file_list_WOW, file_list_UNIWARD = dataload(payload)

all_file_lists = [file_list_cover, file_list_WOW]

labels = list(range(len(all_file_lists)))
paths_train, paths_val, labels_train, labels_val = train_test_split(all_file_lists, labels, test_size=0.2, random_state=42)
train_dataset = create_tf_dataset(paths_train, labels_train, batch_size=64)
val_dataset = create_tf_dataset(paths_val, labels_val, batch_size=64)

print("Train dataset size:", len(train_dataset))
print("Validation dataset size:", len(val_dataset))

batch_size = 64
epochs = 100
spatial_dropout = 0.1
bn_momentum = 0.2
epsilon = 0.001
norm_momentum = 0.4
leakyrelu_slope = -0.1
momentum_sgd = 0.95
learning_rate = 0.001

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='linear', kernel_initializer=glorot_normal(), kernel_regularizer=l2(0.01), input_shape=(input_shape)))
model.add(BatchNormalization(momentum=bn_momentum, epsilon=epsilon))
model.add(LeakyReLU(alpha=leakyrelu_slope))
model.add(SpatialDropout2D(spatial_dropout))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='linear', kernel_initializer=glorot_normal(), kernel_regularizer=l2(0.01)))
model.add(BatchNormalization(momentum=bn_momentum, epsilon=epsilon))
model.add(LeakyReLU(alpha=leakyrelu_slope))
model.add(Dense(units=output_dim, activation='softmax', kernel_initializer=glorot_normal(), kernel_regularizer=l2(0.01)))


sgd = SGD(learning_rate=learning_rate, momentum=momentum_sgd)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
