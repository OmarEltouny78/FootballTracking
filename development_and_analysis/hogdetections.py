from skimage.feature import hog
from skimage import data, exposure
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random as rng

img=cv2.imread('C:\\Users\\omare\\EPLTeacking\\output_videos\\cropped_img.jpg')
img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

hog_image = hog(img)
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
plt.imshow(hog_image_rescaled)

plt.show()