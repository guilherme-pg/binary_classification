
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
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from Service.general_settings import IMG_SIZE


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GENERAL VARIABLES and CONFIGURATIONS  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

dir_path = "C:/Users/guilhermevmmpg/Documents/DEV/projetos/binary_classification/"

categories = ["bees_ed", "aedes_aegypti_ed"]

data = []

NUM_PARALLEL_EXEC_UNITS = 0

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS,
                                  inter_op_parallelism_threads=2,
                                  allow_soft_placement=True,
                                  device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})


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

X_features = np.array(X_features, dtype=object).astype("float32")
y_labels = np.array(y_labels, dtype=object).astype("float32")

X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels,
                                                    test_size=0.30, shuffle=True, random_state=8)


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.30, shuffle=True, random_state=8)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DATA AUGMENTATION  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# create rescale=1/255. training instantce with data augmentation
image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=0.20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.2, 0.4]
)


# rescale all data
test_image_generator = ImageDataGenerator(rescale=1./255)


# fit augmented data
image_generator.fit(X_train)

# read the images directly from the directory and augment them
train_generator = image_generator.flow(X_train,
                                       y_train,
                                       batch_size=2)

validation_generator = image_generator.flow(X_val,
                                            y_val,
                                            batch_size=2)

test_set = test_image_generator.flow(X_test,
                                     y_test,
                                     batch_size=2)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CNN MODEL  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model = Sequential()

model.add(Resizing(IMG_SIZE, IMG_SIZE))
model.add(Conv2D(8, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])


#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FITTING MODEL  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("train_generator ::::::::::::::::::::::::::::::::::::::: ", type(train_generator))
model_fitted = model.fit(
    train_generator,
    epochs=100,
    verbose=2,
    batch_size=8,
    validation_data=validation_generator,
    validation_steps=3,
    shuffle=True
)


#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ EVALUATING MODEL  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""print('                        //                          ')
valid_loss, valid_accuracy = model.evaluate(validation_generator)
print("VALIDATION Loss: ", valid_loss)
print("VALIDATION Accuracy: ", valid_accuracy)
test_loss, test_accuracy = model.evaluate(test_set)
print("TEST Loss: ", test_loss)
print("TEST Accuracy: ", test_accuracy)

print("score on test: " + str(model.evaluate(X_test, y_test)[1]))
print("score on train: " + str(model.evaluate(X_train, y_train)[1]))"""


#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CONFUSION MATRIX  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PLOTTING SCORES  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(model_fitted.history.keys())

plt.interactive(False)
plt.plot(model_fitted.history['loss'], label='loss', color='red')
plt.plot(model_fitted.history['accuracy'], label='accuracy', color='blue')
plt.plot(model_fitted.history['val_loss'], label='val_loss', color='orange', alpha=0.4)
plt.plot(model_fitted.history['val_accuracy'], label='val_accuracy', color='green', alpha=0.4)
plt.legend(loc="upper left")
plt.savefig("binary_classification_plot.jpg")
plt.show()
