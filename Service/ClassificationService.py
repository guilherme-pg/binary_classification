
import keras
import cv2
import tensorflow as tf
from which_maracatu import model


# check if file is corrupted

# check if file format is allowed





# import model
# predict image
def classify_image(img):
        IMG_SIZE = 100 # constant

        img_array = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        img_array_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

        predictions = model.predict(img_array)
        score = float(predictions[0])

        # RETURN : SCORE RESULTS


