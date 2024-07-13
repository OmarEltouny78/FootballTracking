import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import easyocr

image_path='C:\\Users\\omare\\EPLTeacking\\output_videos\\cropped_img.jpg'
image=cv2.imread(image_path)
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)



top_half_image=image[int(image.shape[0]/4):int(image.shape[0]/2),:]
print(top_half_image.shape)

plt.imshow(top_half_image)

plt.show()
