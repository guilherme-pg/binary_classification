









# ~~~~~~~~~~~~~~~ IMPORTS  ~~~~~~~~~~~~~~~

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle





# ~~~~~~~~~~~~~~~ GENERAL VARIABLES  ~~~~~~~~~~~~~~~

dir_path = "C:\Users\guima\Desktop\data_science\Projetos\which_maracatu"

IMG_SIZE = 75

categories = ["maracatu_nation", "maracatu_rural"]





# ~~~~~~~~~~~~~~~ DATA PROCESSING  ~~~~~~~~~~~~~~~

for category in categories:
    path = os.path.join(dir_path, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        img_array_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        
        





# ~~~~~~~~~~~~~~~ TRAIN / TEST  ~~~~~~~~~~~~~~~







# ~~~~~~~~~~~~~~~ TEST PERFORMANCE  ~~~~~~~~~~~~~~~








