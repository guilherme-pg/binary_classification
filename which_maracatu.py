

# ~~~~~~~~~~~~~~~ IMPORTS  ~~~~~~~~~~~~~~~

import numpy as np
import os
import cv2
import random
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from Service.general_settings import IMG_SIZE


# ~~~~~~~~~~~~~~~ GENERAL VARIABLES  ~~~~~~~~~~~~~~~

dir_path = "C:/Users/guima/Desktop/data_science/Projetos/which_maracatu"

categories = ["maracatu_nation", "maracatu_rural"]

data = []


# ~~~~~~~~~~~~~~~ DATA PROCESSING  ~~~~~~~~~~~~~~~

for category in categories:
    path = os.path.join(dir_path, category)
    label = categories.index(category)
    
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        img_array_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        data.append([img_array_resized, label])
        

random.shuffle(data)

X = []
y = []

for features, labels in data:
    X.append(features)
    y.append(labels)


X = np.array(X)
y = np.array(y)

X = tf.expand_dims(X, axis=-1)


# ~~~~~~~~~~~~~~~ TRAIN / TEST  ~~~~~~~~~~~~~~~


X = X/255


model = Sequential()

model.add(Conv2D(30, (3, 3), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(30, (3, 3), activation='relu'))

model.add(MaxPooling2D((2, 2)))


model.add(Flatten())

model.add(Dense(8, input_shape=X.shape[1:], activation='relu'))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(X, y, batch_size=32, epochs=25, validation_split=0.1)

# REQUIRE: IMPROVE THE MODEL METRICS SCORES