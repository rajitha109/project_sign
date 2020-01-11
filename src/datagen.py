import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import os

import dataset_builder as db

import keras
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau
from keras.utils import np_utils
from keras import backend as K

import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

from sklearn.model_selection import train_test_split

# create path if not exists
def create_ifnex(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

nb_gen = 20;

#path = 'K:/python/Cam/gest/gestures'

path = 'K:/python/Cam/gestures'


# exit program is path if not exists
def exit_ifnex(directory):
    if not os.path.exists(directory):
        print(directory, 'does not exist')
        exit()

# loads an opencv image from a filepath
def get_img(path):
    image = cv2.imread(path, 0) if db.grayscale else cv2.imread(path, db.channel)
    image = cv2.resize(image, (db.width, db.height))
    image = img_to_array(image)
    image = image.reshape(db.width, db.height, db.channel)
    return image


# use keras to generate more data from existing images



datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    )


classesFolders = os.listdir(path)
for folder in classesFolders:
    files = os.listdir(os.path.join(path, folder))
    for fl in files:
        img = get_img(os.path.join(path, folder, fl))
        img = img.reshape(1, db.width, db.height, db.channel)
        i = 0
        for batch in datagen.flow(img, batch_size=1, save_to_dir=os.path.join(path, folder), save_prefix='genfile', save_format=db.file_format):
            i += 1
            if i > nb_gen:
                break
