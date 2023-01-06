import cv2
from which_maracatu import model
from general_settings import IMG_SIZE


# import model
# predict image
def classify_image(img):
    # check if file is corrupted
    # check if file is an image

    img_array = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img_array_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

    predictions = model.predict(img_array_resized)
    scores = float(predictions[0])
        # REQUIRE: return multiple scores with analythics
    return scores
    # RETURN : SCORE RESULTS
