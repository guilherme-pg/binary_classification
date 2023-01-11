
# ~~~~~~~~~~~~~~~ IMPORTS  ~~~~~~~~~~~~~~~

import numpy as np
import os
import random
import tensorflow as tf
from keras.utils import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Resizing, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator
# from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
from Service.general_settings import IMG_SIZE


# ~~~~~~~~~~~~~~~ GENERAL VARIABLES  ~~~~~~~~~~~~~~~

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

dir_path = "C:/Users/guima/Desktop/data_science/Projetos/which_maracatu"

categories = ["maracatu_nation", "maracatu_rural"]

data = []


# ~~~~~~~~~~~~~~~ LOADING DATA  ~~~~~~~~~~~~~~~

for category in categories:
    path = os.path.join(dir_path, category)
    label = categories.index(category)
    
    for img in os.listdir(path):
        image = load_img(os.path.join(path, img), color_mode="grayscale")
        img_array = img_to_array(image)
        data.append([img_array, label])

random.shuffle(data)

X = []
y = []

for features, labels in data:
    X.append(features)
    y.append(labels)


X = np.array(X, dtype=object)
y = np.array(y, dtype=object)

#X = tf.expand_dims(X, axis=-1)


# ~~~~~~~~~~~~~~~ DATA AUGMENTATION  ~~~~~~~~~~~~~~~

# create rescale=1/255. training instantce with data augmentation
data_gen = ImageDataGenerator(
    rotation_range=0.20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.2, 0.5],
    zoom_range=[0.2, 0.4],
    featurewise_center=True,
    featurewise_std_normalization=True
)

# create ImageDataGenerator without data augmentation
train_datagen = ImageDataGenerator(rescale=1/255.)

# create ImageDataGenerator without data augmentation for the test dataset
test_datagen = ImageDataGenerator(rescale=1/255.)

# 
train_data_augmented = train_data_augmented.flow_from_directorydir_path(dir_path, target_size=(IMG_SIZE, IMG_SIZE), batch_size=32, class_mode="binary", shuffle=True)

# create non_augmented train data batches
train_data = train_datagen.flow_from_directory(dir_path, target_size=(IMG_SIZE, IMG_SIZE), batch_size=32, class_mode="binary", shuffle=True)




# train_generator = data_gen.flow(X)


data_gen.fit(X)


# ~~~~~~~~~~~~~~~ TRAIN / TEST  ~~~~~~~~~~~~~~~


# X = X/255


model = Sequential()

model.add(Resizing(IMG_SIZE, IMG_SIZE))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])


fitted_model = model.fit(
    data_gen.flow(X, y, batch_size=32),
    batch_size=32,
    epochs=15
)


axis_x = list(range(1, 16))
print(fitted_model.history.keys())


plt.interactive(False)
plt.plot(axis_x, fitted_model.history['loss'], label='loss', color='red')
plt.plot(axis_x, fitted_model.history['accuracy'], label='accuracy', color='blue')
plt.plot(axis_x, fitted_model.history['val_loss'], label='val_loss', color='orange', alpha=0.4)
plt.plot(axis_x, fitted_model.history['val_accuracy'], label='val_accuracy', color='green', alpha=0.4)
plt.legend(loc="upper left")
plt.show()



# REQUIRE: DATA AUGMENTATION




# REQUIRE: IMPROVE THE MODEL METRICS SCORES