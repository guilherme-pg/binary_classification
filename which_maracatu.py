
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ IMPORTS  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import os
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Resizing, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator
# from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
from Service.general_settings import IMG_SIZE


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GENERAL VARIABLES  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

dir_path = "C:/Users/guilhermevmmpg/Documents/DEV/projetos/binary_classification/"

categories = ["maracatu_nation", "maracatu_rural"]

data = []


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LOADING DATA  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
for category in categories:
    # path to each directory of the 2 types
    path = os.path.join(dir_path, category)
    # labels classification
    label = categories.index(category)
    
    for img in os.listdir(path):
        # loading images from each directory and set than grey
        image = load_img(os.path.join(path, img), color_mode="grayscale")
        img_array = img_to_array(image)
        img_array = tf.image.resize(img_array, (IMG_SIZE, IMG_SIZE))
        data.append([img_array, label])

random.shuffle(data)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SPLITTING DATA  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
X_features = []
y_labels = []

for features, labels in data:
    X_features.append(features)
    y_labels.append(labels)

# X_features = tf.expand_dims(X_features, axis=-1)

X_features = np.array(X_features, dtype=object)
y_labels = np.array(y_labels, dtype=object)

X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels,
                                                    test_size=0.20, shuffle=True, random_state=8)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.20, shuffle=True, random_state=8)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DATA AUGMENTATION  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# create rescale=1/255. training instantce with data augmentation
image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=0.20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.2, 0.5],
    zoom_range=[0.2, 0.4]
)


# rescale all data
test_dataset = ImageDataGenerator(rescale=1./255)


# read the images directly from the directory and augment them
train_generator = image_generator.flow(X_train,
                                       y_train,
                                       batch_size=32,
                                       shuffle=True)

validation_generator = image_generator.flow(X_val,
                                            y_val,
                                            batch_size=32,
                                            shuffle=True)

test_set = test_dataset.flow(X_test,
                             y_test,
                             batch_size=32,
                             shuffle=True)

# image_generator.fit(X_features)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CNN MODEL  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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


#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FITTING MODEL  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
model_fitted = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=2
)


#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ EVALUATING MODEL  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("====================//======================")
val_loss, val_accuracy = model.evaluate(validation_generator)
print("VALIDATION Loss: ", val_loss)
print("VALIDATION Accuracy: ", val_accuracy)
print("====================//======================")
test_loss, test_accuracy = model.evaluate(test_set)
print("TEST Loss: ", test_loss)
print("TEST Accuracy: ", test_accuracy)
print("====================//======================")

#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PLOTTING SCORES  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
axis_x = list(range(1, 16))

plt.interactive(False)
plt.plot(axis_x, model_fitted.history['loss'], label='loss', color='red')
plt.plot(axis_x, model_fitted.history['accuracy'], label='accuracy', color='blue')
plt.plot(axis_x, model_fitted.history['val_loss'], label='val_loss', color='orange', alpha=0.4)
plt.plot(axis_x, model_fitted.history['val_accuracy'], label='val_accuracy', color='green', alpha=0.4)
plt.legend(loc="upper left")
plt.show()
