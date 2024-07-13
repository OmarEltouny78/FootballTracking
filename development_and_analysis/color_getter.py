import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from PIL import Image
from collections import defaultdict

import tensorflow as tf

image_path='C:\\Users\\omare\\EPLTeacking\\output_images\\Visible numbers\\cropped_image0.jpg'
image=cv2.imread(image_path)
image_resize=cv2.resize(image,(28,56))
image_resize=image_resize/255.0
image2d=np.expand_dims(image_resize,axis=0)
#image2d=image_resize.reshape(28,56,3)
new_model = tf.keras.models.load_model('C:\\Users\\omare\\EPLTeacking\\development_and_analysis\\my_model.h5')

number=np.argmax(new_model.predict(image2d))

print(number)

